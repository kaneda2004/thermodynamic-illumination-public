#!/usr/bin/env python3
"""
RES-285: Early-Trajectory Predictability for Two-Stage Speedup (Parallelized)

Goal: Predict log-speedup using early trajectory signals (init stats + early dynamics),
      reducing randomness and achieving positive R² on held-out seeds.

Key ideas:
1) Fix initial live sets per seed (baseline + two-stage use identical init weights).
2) Use a small number of replicates to average out ESS noise.
3) Extract early-trajectory features (acceptance, contractions, best-order slope).
4) Exclude trivial seeds where init already exceeds target.
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
from sklearn.model_selection import cross_val_score
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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    n_seeds: int = 100
    n_replicates: int = 3
    order_target: float = 0.50
    n_workers: int = 32  # Parallel workers

    n_live: int = 100
    baseline_max_iterations: int = 300

    n_live_stage1: int = 50
    stage1_budget: int = 150
    max_iterations_stage2: int = 250
    pca_components: int = 3

    early_window: int = 50
    top_k_init: int = 10
    diversity_pairs: int = 200

    image_size: int = 32
    seed: int = 42
    exclude_trivial: bool = True
    checkpoint_interval: int = 10


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


def compute_image_feature_stats(images: List[np.ndarray]) -> Dict[str, float]:
    if not images:
        return {
            "density_mean": 0.0,
            "edge_mean": 0.0,
            "coherence_mean": 0.0,
            "compress_mean": 0.0,
            "symmetry_mean": 0.0,
        }

    densities = [float(np.mean(img)) for img in images]
    edges = [float(compute_edge_density(img)) for img in images]
    coherence = [float(compute_spectral_coherence(img)) for img in images]
    compress = [float(compute_compressibility(img)) for img in images]
    symmetry = [float(compute_symmetry(img)) for img in images]

    return {
        "density_mean": float(np.mean(densities)),
        "edge_mean": float(np.mean(edges)),
        "coherence_mean": float(np.mean(coherence)),
        "compress_mean": float(np.mean(compress)),
        "symmetry_mean": float(np.mean(symmetry)),
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


def run_baseline_with_stats(
    init_cache: Dict,
    config: ExperimentConfig,
    rng_seed: int,
    early_window: int,
) -> Tuple[Dict, Dict]:
    set_global_seed(rng_seed)
    live_points = build_live_points(
        init_cache["weights"], init_cache["images"], init_cache["orders"]
    )

    n_live = len(live_points)
    best_order = max(lp[2] for lp in live_points)
    total_samples = n_live
    samples_to_target = total_samples if best_order >= config.order_target else None

    early_best = []
    early_contractions = []
    early_success = 0

    for iteration in range(config.baseline_max_iterations):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

        proposal_cppn, proposal_img, proposal_order, _, n_contractions, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        total_samples += 1

        if iteration < early_window:
            early_best.append(best_order)
            early_contractions.append(n_contractions)
            if success:
                early_success += 1

        if best_order >= config.order_target and samples_to_target is None:
            samples_to_target = total_samples
            break

    if samples_to_target is None:
        samples_to_target = total_samples

    early_iters = max(1, len(early_best))
    early_stats = {
        "accept_rate": float(early_success / early_iters),
        "contractions_mean": float(np.mean(early_contractions)) if early_contractions else 0.0,
        "best_slope": compute_slope(early_best),
        "best_delta": float(early_best[-1] - early_best[0]) if early_best else 0.0,
        "best_end": float(early_best[-1]) if early_best else float(best_order),
    }

    result = {
        "total_samples": total_samples,
        "samples_to_target": samples_to_target,
        "success": best_order >= config.order_target,
        "max_order_achieved": float(best_order),
    }
    return result, early_stats


def run_two_stage_with_stats(
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
    best_order = max(lp[2] for lp in live_points)

    total_samples = n_live
    samples_to_target = total_samples if best_order >= config.order_target else None

    collected_weights = [lp[0].get_weights() for lp in live_points]
    stage1_success = 0
    stage1_iters = max(0, config.stage1_budget - n_live)

    for _ in range(stage1_iters):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            best_order = max(best_order, proposal_order)
            stage1_success += 1

        total_samples += 1

        if best_order >= config.order_target and samples_to_target is None:
            samples_to_target = total_samples
            break

    pca_var, eff_dim, pca_mean, pca_components = compute_pca_stats(
        collected_weights, config.pca_components
    )

    stage1_stats = {
        "accept_rate": float(stage1_success / max(1, stage1_iters)),
        "pca_variance": float(pca_var),
        "eff_dim": float(eff_dim),
        "best_end": float(best_order),
    }

    if pca_mean is not None and pca_components is not None and samples_to_target is None:
        for _ in range(config.max_iterations_stage2):
            worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
            threshold = live_points[worst_idx][2]
            seed_idx = np.random.randint(0, n_live)

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = pca_components @ (current_w - pca_mean)

            delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
            proposal_w = pca_mean + (pca_components.T @ (coeffs + delta))

            proposal_cppn = live_points[seed_idx][0].copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(config.image_size)
            proposal_order = order_multiplicative(proposal_img)

            if proposal_order >= threshold:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)

            total_samples += 1

            if best_order >= config.order_target:
                samples_to_target = total_samples
                break

    if samples_to_target is None:
        samples_to_target = total_samples

    result = {
        "total_samples": total_samples,
        "samples_to_target": samples_to_target,
        "success": best_order >= config.order_target,
        "max_order_achieved": float(best_order),
    }
    return result, stage1_stats


def worker_task(args):
    seed_idx, config = args
    try:
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

        # Init feature stats
        rng = np.random.default_rng(base_seed + 123)
        top_k = min(config.top_k_init, len(init_orders))
        top_idx = np.argsort(init_orders)[-top_k:]
        top_images = [init_cache["images"][i] for i in top_idx]

        init_features = {
            "init_order_max": float(init_orders.max()),
            "init_order_mean": float(init_orders.mean()),
            "init_order_std": float(init_orders.std()),
            "init_order_p95": float(np.percentile(init_orders, 95)),
            "init_diversity": compute_diversity(init_cache["images"], rng, config.diversity_pairs),
        }
        init_features.update(compute_image_feature_stats(top_images))

        baseline_samples = []
        two_stage_samples = []
        early_stats = {}
        stage1_stats = {}

        for rep in range(config.n_replicates):
            rep_seed = base_seed + rep + 1
            baseline_result, early = run_baseline_with_stats(
                init_cache, config, rep_seed, config.early_window
            )
            two_stage_result, s1 = run_two_stage_with_stats(
                init_cache, config, rep_seed + 777
            )

            baseline_samples.append(baseline_result["samples_to_target"])
            two_stage_samples.append(two_stage_result["samples_to_target"])

            if rep == 0:
                early_stats = early
                stage1_stats = s1

        baseline_mean = float(np.mean(baseline_samples))
        two_stage_mean = float(np.mean(two_stage_samples))
        speedup = baseline_mean / two_stage_mean if two_stage_mean > 0 else 0.0
        log_speedup = float(np.log(speedup)) if speedup > 0 else 0.0

        return {
            "seed": seed_idx,
            "trivial": False,
            "baseline_mean_samples": baseline_mean,
            "two_stage_mean_samples": two_stage_mean,
            "speedup": speedup,
            "log_speedup": log_speedup,
            "features": {
                **init_features,
                "early_accept_rate": early_stats.get("accept_rate", 0.0),
                "early_contractions_mean": early_stats.get("contractions_mean", 0.0),
                "early_best_slope": early_stats.get("best_slope", 0.0),
                "early_best_delta": early_stats.get("best_delta", 0.0),
                "early_best_end": early_stats.get("best_end", 0.0),
                "stage1_accept_rate": stage1_stats.get("accept_rate", 0.0),
                "stage1_pca_variance": stage1_stats.get("pca_variance", 0.0),
                "stage1_eff_dim": stage1_stats.get("eff_dim", 0.0),
                "stage1_best_end": stage1_stats.get("best_end", 0.0),
            }
        }
    except Exception as e:
        logger.error(f"Error in seed {seed_idx}: {str(e)}")
        return {"seed": seed_idx, "error": str(e)}


def run_experiment(config: ExperimentConfig) -> Dict:
    logger.info(f"Starting RES-285 with {config.n_seeds} seeds on {config.n_workers} workers")
    
    rows = []
    
    with Pool(config.n_workers) as pool:
        args = [(i, config) for i in range(config.n_seeds)]
        results = []
        for i, res in enumerate(pool.imap_unordered(worker_task, args)):
            results.append(res)
            if (i + 1) % config.checkpoint_interval == 0:
                logger.info(f"Progress: {i + 1}/{config.n_seeds} seeds")
    
    # Process results
    for res in results:
        if "error" in res:
            logger.warning(f"Seed {res['seed']} failed: {res['error']}")
            continue
        rows.append(res)

    # Build dataset
    valid_rows = [r for r in rows if not r.get("trivial")]
    if not valid_rows:
        return {"rows": rows, "summary": {"error": "No non-trivial seeds."}}

    feature_names = sorted(valid_rows[0]["features"].keys())
    X = np.array([[r["features"][k] for k in feature_names] for r in valid_rows], dtype=float)
    y = np.array([r["log_speedup"] for r in valid_rows], dtype=float)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    y_pred = rf.predict(X)
    r2_train = float(r2_score(y, y_pred))
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")

    summary = {
        "n_seeds_total": config.n_seeds,
        "n_seeds_used": len(valid_rows),
        "n_trivial": config.n_seeds - len(valid_rows),
        "mean_speedup": float(np.mean([r["speedup"] for r in valid_rows])),
        "r2_train": r2_train,
        "r2_cv_mean": float(cv_scores.mean()),
        "r2_cv_std": float(cv_scores.std()),
    }

    feature_importance = {
        name: float(val) for name, val in zip(feature_names, rf.feature_importances_)
    }

    return {
        "config": {
            "n_seeds": config.n_seeds,
            "n_replicates": config.n_replicates,
            "order_target": config.order_target,
            "n_live": config.n_live,
            "baseline_max_iterations": config.baseline_max_iterations,
            "n_live_stage1": config.n_live_stage1,
            "stage1_budget": config.stage1_budget,
            "max_iterations_stage2": config.max_iterations_stage2,
            "pca_components": config.pca_components,
            "early_window": config.early_window,
            "top_k_init": config.top_k_init,
            "diversity_pairs": config.diversity_pairs,
            "exclude_trivial": config.exclude_trivial,
        },
        "summary": summary,
        "feature_importance": feature_importance,
        "cv_scores": [float(s) for s in cv_scores],
        "feature_names": feature_names,
        "rows": rows,
    }


def main():
    print("=" * 80)
    print("RES-285: Early-Trajectory Predictability (Parallelized)")
    print("=" * 80)

    config = ExperimentConfig()
    results = run_experiment(config)

    results_dir = project_root / "results" / "early_predictability"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "res_285_results.json"
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
    print(f"Mean speedup: {summary['mean_speedup']:.2f}x")
    print(f"R² train: {summary['r2_train']:.3f}")
    print(f"R² CV: {summary['r2_cv_mean']:.3f} ± {summary['r2_cv_std']:.3f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()