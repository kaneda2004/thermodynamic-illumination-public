#!/usr/bin/env python3
"""
RES-287: Fixed-Budget Gain Predictability (Paired Inits)
=======================================================

Goal: Predict two-stage benefit with positive CV R^2 by reducing noise and
using smoother targets than "samples-to-threshold".

Key changes vs prior experiments:
1) Fixed-budget metric (AUC of best-order curve) instead of time-to-target.
2) Paired initial live sets for baseline and two-stage per seed.
3) Replicate averaging to reduce stochastic variance.
4) Early-trajectory features (baseline + stage1) to capture learnability.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Project root resolution (local + GCP)
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    elliptical_slice_sample,
    set_global_seed,
    PRIOR_SIGMA,
    compute_edge_density,
    compute_spectral_coherence,
    compute_compressibility,
    compute_symmetry,
    compute_connected_components,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    n_seeds: int = 100
    n_replicates: int = 5
    n_workers: int = 32
    seed: int = 42

    image_size: int = 32
    order_target: float = 0.55
    exclude_trivial: bool = True

    n_live: int = 100
    budget_total: int = 2000

    n_live_stage1: int = 50
    stage1_budget: int = 300
    pca_components: int = 3
    pca_step_scale: float = 0.5

    early_window: int = 50
    top_k_init: int = 10
    diversity_pairs: int = 200


def generate_initial_live_set(seed: int, n_live: int, image_size: int) -> Dict:
    set_global_seed(seed)
    weights = []
    orders = []
    images = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        weights.append(cppn.get_weights())
        orders.append(order)
        images.append(img)
    return {
        "weights": weights,
        "orders": np.array(orders, dtype=float),
        "images": images,
    }


def build_live_points(weights: List[np.ndarray], images: List[np.ndarray], orders: np.ndarray) -> List[Tuple]:
    live_points = []
    for w, img, order in zip(weights, images, orders):
        cppn = CPPN()
        cppn.set_weights(w)
        live_points.append((cppn, img, float(order)))
    return live_points


def compute_diversity(images: List[np.ndarray], rng: np.random.Generator, n_pairs: int) -> float:
    n = len(images)
    if n < 2:
        return 0.0
    max_pairs = n * (n - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    total = 0.0
    for _ in range(n_pairs):
        i = rng.integers(0, n)
        j = rng.integers(0, n - 1)
        if j >= i:
            j += 1
        total += float(np.mean(images[i] != images[j]))
    return total / n_pairs if n_pairs > 0 else 0.0


def _spectral_ratio(img: np.ndarray) -> float:
    size = img.shape[0]
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    y, x = np.indices((size, size))
    center = np.array([size // 2, size // 2])
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    low = np.sum(radial_profile[: size // 4])
    high = np.sum(radial_profile[size // 4 :])
    return float(low / (high + 1e-8))


def compute_image_feature_stats(images: List[np.ndarray]) -> Dict[str, float]:
    if not images:
        return {
            "density_mean": 0.0,
            "edge_mean": 0.0,
            "coherence_mean": 0.0,
            "compress_mean": 0.0,
            "symmetry_mean": 0.0,
            "components_mean": 0.0,
            "gradient_mean": 0.0,
            "spectral_ratio_mean": 0.0,
        }

    densities = []
    edges = []
    coherence = []
    compress = []
    symmetry = []
    components = []
    gradients = []
    spectral = []

    for img in images:
        densities.append(float(np.mean(img)))
        edges.append(float(compute_edge_density(img)))
        coherence.append(float(compute_spectral_coherence(img)))
        compress.append(float(compute_compressibility(img)))
        symmetry.append(float(compute_symmetry(img)))
        components.append(float(compute_connected_components(img)))
        gy, gx = np.gradient(img)
        gradients.append(float(np.mean(np.sqrt(gx ** 2 + gy ** 2))))
        spectral.append(_spectral_ratio(img))

    return {
        "density_mean": float(np.mean(densities)),
        "edge_mean": float(np.mean(edges)),
        "coherence_mean": float(np.mean(coherence)),
        "compress_mean": float(np.mean(compress)),
        "symmetry_mean": float(np.mean(symmetry)),
        "components_mean": float(np.mean(components)),
        "gradient_mean": float(np.mean(gradients)),
        "spectral_ratio_mean": float(np.mean(spectral)),
    }


def compute_pca_stats(weights_samples: List[np.ndarray], n_components: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(weights_samples) < 2:
        return 0.0, 0.0, None, None
    W = np.array(weights_samples)
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean
    _, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    n_comp = min(n_components, len(S))
    explained_var = float((S[:n_comp] ** 2).sum() / (S ** 2).sum()) if len(S) > 0 else 0.0

    eigvals = S ** 2
    denom = float((eigvals ** 2).sum())
    eff_dim = float((eigvals.sum() ** 2) / denom) if denom > 0 else 0.0
    return explained_var, eff_dim, W_mean, Vt[:n_comp]


def compute_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _init_history_from_orders(orders: np.ndarray) -> Tuple[float, float, int]:
    best_order = -1.0
    best_sum = 0.0
    count = 0
    for order in orders:
        if order > best_order:
            best_order = float(order)
        best_sum += best_order
        count += 1
    return best_order, best_sum, count


def run_baseline_fixed_budget(
    init_cache: Dict,
    config: ExperimentConfig,
    rng_seed: int,
) -> Tuple[Dict, Dict]:
    set_global_seed(rng_seed)
    live_points = build_live_points(
        init_cache["weights"], init_cache["images"], init_cache["orders"]
    )

    n_live = len(live_points)
    best_order, best_sum, count = _init_history_from_orders(init_cache["orders"])
    iterations = max(0, config.budget_total - n_live)

    early_best = []
    early_contractions = []
    early_success = 0

    for iteration in range(iterations):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

        proposal_cppn, proposal_img, proposal_order, _, n_contractions, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            if proposal_order > best_order:
                best_order = float(proposal_order)

        best_sum += best_order
        count += 1

        if iteration < config.early_window:
            early_best.append(best_order)
            early_contractions.append(n_contractions)
            if success:
                early_success += 1

    early_iters = max(1, len(early_best))
    early_stats = {
        "accept_rate": float(early_success / early_iters),
        "contractions_mean": float(np.mean(early_contractions)) if early_contractions else 0.0,
        "best_slope": compute_slope(early_best),
        "best_delta": float(early_best[-1] - early_best[0]) if early_best else 0.0,
        "best_end": float(early_best[-1]) if early_best else float(best_order),
    }

    result = {
        "auc": float(best_sum / max(1, count)),
        "final_best": float(best_order),
        "total_samples": count,
    }
    return result, early_stats


def run_two_stage_fixed_budget(
    init_cache: Dict,
    config: ExperimentConfig,
    rng_seed: int,
) -> Tuple[Dict, Dict]:
    set_global_seed(rng_seed)
    stage1_weights = init_cache["weights"][: config.n_live_stage1]
    stage1_images = init_cache["images"][: config.n_live_stage1]
    stage1_orders = init_cache["orders"][: config.n_live_stage1]

    live_points = build_live_points(stage1_weights, stage1_images, stage1_orders)
    n_live = len(live_points)

    best_order, best_sum, count = _init_history_from_orders(stage1_orders)
    stage1_iters = max(0, config.stage1_budget - n_live)

    collected_weights = [lp[0].get_weights() for lp in live_points]
    stage1_best = []
    stage1_success = 0

    for iteration in range(stage1_iters):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            if proposal_order > best_order:
                best_order = float(proposal_order)
            stage1_success += 1

        best_sum += best_order
        count += 1

        if iteration < config.early_window:
            stage1_best.append(best_order)

    pca_var, eff_dim, pca_mean, pca_components = compute_pca_stats(
        collected_weights, config.pca_components
    )

    stage1_stats = {
        "accept_rate": float(stage1_success / max(1, stage1_iters)),
        "pca_variance": float(pca_var),
        "eff_dim": float(eff_dim),
        "best_slope": compute_slope(stage1_best),
        "best_delta": float(stage1_best[-1] - stage1_best[0]) if stage1_best else 0.0,
        "best_end": float(stage1_best[-1]) if stage1_best else float(best_order),
    }

    stage2_iters = max(0, config.budget_total - config.stage1_budget)
    if pca_mean is not None and pca_components is not None:
        for _ in range(stage2_iters):
            worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
            threshold = live_points[worst_idx][2]
            seed_idx = np.random.randint(0, n_live)

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = pca_components @ (current_w - pca_mean)

            delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * config.pca_step_scale
            proposal_w = pca_mean + (pca_components.T @ (coeffs + delta))

            proposal_cppn = live_points[seed_idx][0].copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(config.image_size)
            proposal_order = order_multiplicative(proposal_img)

            if proposal_order >= threshold:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                if proposal_order > best_order:
                    best_order = float(proposal_order)

            best_sum += best_order
            count += 1
    else:
        for _ in range(stage2_iters):
            best_sum += best_order
            count += 1

    result = {
        "auc": float(best_sum / max(1, count)),
        "final_best": float(best_order),
        "total_samples": count,
    }
    return result, stage1_stats


def run_seed_task(args) -> Dict:
    seed_idx, config = args
    base_seed = config.seed + seed_idx * 1000

    init_cache = generate_initial_live_set(base_seed, config.n_live, config.image_size)
    init_orders = init_cache["orders"]

    trivial = bool(init_orders.max() >= config.order_target)
    if trivial and config.exclude_trivial:
        return {
            "seed": seed_idx,
            "trivial": True,
            "init_max_order": float(init_orders.max()),
        }

    rng = np.random.default_rng(base_seed + 123)
    top_k = min(config.top_k_init, len(init_orders))
    top_idx = np.argsort(init_orders)[-top_k:]
    top_images = [init_cache["images"][i] for i in top_idx]

    init_features = {
        "init_order_max": float(init_orders.max()),
        "init_order_mean": float(init_orders.mean()),
        "init_order_std": float(init_orders.std()),
        "init_order_p95": float(np.percentile(init_orders, 95)),
        "init_order_p99": float(np.percentile(init_orders, 99)),
        "init_topk_mean": float(np.mean(init_orders[top_idx])),
        "init_topk_std": float(np.std(init_orders[top_idx])),
        "init_diversity": compute_diversity(init_cache["images"], rng, config.diversity_pairs),
    }
    init_features.update(compute_image_feature_stats(top_images))

    baseline_aucs = []
    two_stage_aucs = []
    baseline_finals = []
    two_stage_finals = []

    early_stats_accum = {
        "accept_rate": [],
        "contractions_mean": [],
        "best_slope": [],
        "best_delta": [],
        "best_end": [],
    }
    stage1_stats_accum = {
        "accept_rate": [],
        "pca_variance": [],
        "eff_dim": [],
        "best_slope": [],
        "best_delta": [],
        "best_end": [],
    }

    for rep in range(config.n_replicates):
        rep_seed = base_seed + rep + 1
        baseline_result, early = run_baseline_fixed_budget(
            init_cache, config, rep_seed
        )
        two_stage_result, s1 = run_two_stage_fixed_budget(
            init_cache, config, rep_seed + 777
        )

        baseline_aucs.append(baseline_result["auc"])
        two_stage_aucs.append(two_stage_result["auc"])
        baseline_finals.append(baseline_result["final_best"])
        two_stage_finals.append(two_stage_result["final_best"])

        for k in early_stats_accum:
            early_stats_accum[k].append(early.get(k, 0.0))
        for k in stage1_stats_accum:
            stage1_stats_accum[k].append(s1.get(k, 0.0))

    baseline_auc = float(np.mean(baseline_aucs))
    two_stage_auc = float(np.mean(two_stage_aucs))
    baseline_final = float(np.mean(baseline_finals))
    two_stage_final = float(np.mean(two_stage_finals))

    delta_auc = two_stage_auc - baseline_auc
    delta_final = two_stage_final - baseline_final
    ratio_auc = (two_stage_auc / (baseline_auc + 1e-8)) - 1.0

    early_stats = {f"early_{k}": float(np.mean(v)) for k, v in early_stats_accum.items()}
    stage1_stats = {f"stage1_{k}": float(np.mean(v)) for k, v in stage1_stats_accum.items()}

    return {
        "seed": seed_idx,
        "trivial": False,
        "baseline_auc": baseline_auc,
        "two_stage_auc": two_stage_auc,
        "baseline_final": baseline_final,
        "two_stage_final": two_stage_final,
        "delta_auc": float(delta_auc),
        "delta_final": float(delta_final),
        "ratio_auc": float(ratio_auc),
        "features": {
            **init_features,
            **early_stats,
            **stage1_stats,
        },
    }


def evaluate_targets(X: np.ndarray, y: np.ndarray) -> Dict:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge_scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
    ridge.fit(X, y)
    ridge_pred = ridge.predict(X)
    results["ridge"] = {
        "r2_train": float(r2_score(y, ridge_pred)),
        "r2_cv_mean": float(ridge_scores.mean()),
        "r2_cv_std": float(ridge_scores.std()),
        "cv_scores": [float(s) for s in ridge_scores],
    }

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
    rf.fit(X, y)
    rf_pred = rf.predict(X)
    results["rf"] = {
        "r2_train": float(r2_score(y, rf_pred)),
        "r2_cv_mean": float(rf_scores.mean()),
        "r2_cv_std": float(rf_scores.std()),
        "cv_scores": [float(s) for s in rf_scores],
        "feature_importance": None,
    }

    return results


def run_experiment(config: ExperimentConfig) -> Dict:
    logger.info("Starting RES-287 with %d seeds on %d workers", config.n_seeds, config.n_workers)

    with Pool(processes=config.n_workers) as pool:
        rows = pool.map(run_seed_task, [(i, config) for i in range(config.n_seeds)])

    valid_rows = [r for r in rows if not r.get("trivial")]
    if not valid_rows:
        return {"rows": rows, "summary": {"error": "No non-trivial seeds."}}

    feature_names = sorted(valid_rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in feature_names] for r in valid_rows], dtype=float)

    targets = {
        "delta_auc": np.array([r["delta_auc"] for r in valid_rows], dtype=float),
        "delta_final": np.array([r["delta_final"] for r in valid_rows], dtype=float),
        "ratio_auc": np.array([r["ratio_auc"] for r in valid_rows], dtype=float),
    }

    model_results = {}
    best_cv = -1e9
    best_target = None
    best_model = None

    for name, y in targets.items():
        res = evaluate_targets(X, y)
        model_results[name] = res
        for model_name, stats in res.items():
            cv_mean = stats["r2_cv_mean"]
            if cv_mean > best_cv:
                best_cv = cv_mean
                best_target = name
                best_model = model_name

    summary = {
        "n_seeds_total": config.n_seeds,
        "n_seeds_used": len(valid_rows),
        "n_trivial": config.n_seeds - len(valid_rows),
        "mean_delta_auc": float(np.mean(targets["delta_auc"])),
        "mean_delta_final": float(np.mean(targets["delta_final"])),
        "mean_ratio_auc": float(np.mean(targets["ratio_auc"])),
        "best_cv_r2": float(best_cv),
        "best_target": best_target,
        "best_model": best_model,
    }

    # Compute feature importance for best target via RF (fit once).
    if best_target is not None:
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, targets[best_target])
        model_results[best_target]["rf"]["feature_importance"] = {
            name: float(val) for name, val in zip(feature_names, rf.feature_importances_)
        }

    return {
        "config": {
            "n_seeds": config.n_seeds,
            "n_replicates": config.n_replicates,
            "n_workers": config.n_workers,
            "order_target": config.order_target,
            "n_live": config.n_live,
            "budget_total": config.budget_total,
            "n_live_stage1": config.n_live_stage1,
            "stage1_budget": config.stage1_budget,
            "pca_components": config.pca_components,
            "pca_step_scale": config.pca_step_scale,
            "early_window": config.early_window,
            "top_k_init": config.top_k_init,
            "diversity_pairs": config.diversity_pairs,
            "exclude_trivial": config.exclude_trivial,
        },
        "summary": summary,
        "model_results": model_results,
        "feature_names": feature_names,
        "rows": rows,
    }


def main():
    print("=" * 80)
    print("RES-287: Fixed-Budget Gain Predictability (Paired Inits)")
    print("=" * 80)

    config = ExperimentConfig()
    results = run_experiment(config)

    results_dir = project_root / "results" / "fixed_budget_gain"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "res_287_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = results.get("summary", {})
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Seeds used: {summary['n_seeds_used']} (trivial: {summary['n_trivial']})")
    print(f"Mean delta AUC: {summary['mean_delta_auc']:.4f}")
    print(f"Mean delta final: {summary['mean_delta_final']:.4f}")
    print(f"Mean ratio AUC: {summary['mean_ratio_auc']:.4f}")
    print(f"Best CV R^2: {summary['best_cv_r2']:.3f} ({summary['best_target']}, {summary['best_model']})")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
