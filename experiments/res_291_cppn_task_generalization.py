#!/usr/bin/env python3
"""
RES-291: CPPN Two-Stage Across Alternate Objectives (Continuous)
===============================================================

Hypothesis: Two-stage sampling generalizes across different objective landscapes
within CPPNs (symmetry, spectral coherence, entropy, alignment) when objectives
are continuous and do not saturate at 1.0.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from multiprocessing import Pool

import numpy as np

# Project root resolution (local + GCP)
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from core.thermo_sampler_v3 import CPPN, set_global_seed, PRIOR_SIGMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    n_cppns: int = 40
    n_workers: int = 8
    seed: int = 42

    image_size: int = 32
    n_live: int = 100
    baseline_max_iterations: int = 1200

    n_live_stage1: int = 50
    stage1_budgets: Tuple[int, ...] = (50, 100)
    max_iterations_stage2: int = 300
    pca_components: int = 3

    calibration_samples: int = 1500
    target_hit_baseline: float = 0.2
    anneal_quantile: float = 0.1
    elite_fraction: float = 0.3
    elite_prob: float = 0.7
    pca_step_scale_start: float = 0.6
    pca_step_scale_end: float = 0.2

    objective_names: Tuple[str, ...] = (
        "symmetry",
        "spectral",
        "kolmogorov",
        "ising",
    )


def render_continuous(cppn: CPPN, size: int) -> np.ndarray:
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return np.clip(cppn.activate(x, y), 0.0, 1.0)


def order_symmetry_continuous(img: np.ndarray) -> float:
    h_sym = 1.0 - np.mean(np.abs(img - np.fliplr(img)))
    v_sym = 1.0 - np.mean(np.abs(img - np.flipud(img)))
    rot180_sym = 1.0 - np.mean(np.abs(img - np.rot90(img, 2)))
    if img.shape[0] == img.shape[1]:
        rot90_sym = 1.0 - np.mean(np.abs(img - np.rot90(img)))
        diag_sym = 1.0 - np.mean(np.abs(img - img.T))
    else:
        rot90_sym = 0.0
        diag_sym = 0.0
    return float((h_sym + v_sym + rot90_sym + rot180_sym + diag_sym) / 5)


def order_spectral_continuous(img: np.ndarray) -> float:
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    low_mask = r < (h / 4)
    low_power = np.sum(power[low_mask])
    total_power = np.sum(power) + 1e-10
    return float(low_power / total_power)


def order_ising_continuous(img: np.ndarray) -> float:
    spins = 2 * img.astype(float) - 1
    h_align = np.sum(spins[:, :-1] * spins[:, 1:])
    v_align = np.sum(spins[:-1, :] * spins[1:, :])
    alignment = h_align + v_align
    max_align = 2 * img.size - img.shape[0] - img.shape[1]
    return float((alignment / max_align + 1) / 2)


def order_kolmogorov_continuous(img: np.ndarray) -> float:
    bins = 64
    hist, _ = np.histogram(img, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(bins)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(1.0 - entropy_norm)


def elliptical_slice_sample_continuous(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5,
) -> tuple[CPPN, np.ndarray, float, int, bool]:
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for _ in range(max_restarts):
        nu = np.random.randn(n_params) * PRIOR_SIGMA
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi
        n_contractions = 0

        while n_contractions < max_contractions:
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)
            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = render_continuous(proposal_cppn, image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                return proposal_cppn, proposal_img, proposal_order, total_contractions + n_contractions, True

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

        total_contractions += n_contractions

    current_img = render_continuous(cppn, image_size)
    return cppn, current_img, order_fn(current_img), total_contractions, False


OBJECTIVE_FUNCS: Dict[str, Callable[[np.ndarray], float]] = {
    "symmetry": order_symmetry_continuous,
    "spectral": order_spectral_continuous,
    "kolmogorov": order_kolmogorov_continuous,
    "ising": order_ising_continuous,
}


def calibrate_target(order_fn: Callable, config: ExperimentConfig, seed: int) -> Dict:
    set_global_seed(seed)
    orders = []
    for _ in range(config.calibration_samples):
        cppn = CPPN()
        img = render_continuous(cppn, config.image_size)
        orders.append(order_fn(img))
    orders = np.array(orders, dtype=float)
    target_hit = min(max(config.target_hit_baseline, 1e-4), 0.95)
    p = 1 - (1 - target_hit) ** (1 / config.n_live)
    quantile = 1.0 - p
    target = float(np.quantile(orders, quantile))
    hit_baseline = 1 - (1 - p) ** config.n_live
    hit_stage1 = 1 - (1 - p) ** config.n_live_stage1
    return {
        "target_order": target,
        "order_mean": float(orders.mean()),
        "order_std": float(orders.std()),
        "quantile": float(quantile),
        "p_per_draw": float(p),
        "target_hit_baseline": float(target_hit),
        "p_hit_baseline": float(hit_baseline),
        "p_hit_stage1": float(hit_stage1),
    }


def run_baseline_single_stage(order_fn: Callable, target_order: float, config: ExperimentConfig) -> Dict:
    live_points = []
    best_order = 0.0
    for _ in range(config.n_live):
        cppn = CPPN()
        img = render_continuous(cppn, config.image_size)
        order = order_fn(img)
        live_points.append((cppn, img, order))
        best_order = max(best_order, order)

    total_samples = config.n_live
    samples_to_target = total_samples if best_order >= target_order else None

    for _ in range(config.baseline_max_iterations):
        worst_idx = min(range(config.n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, config.n_live)

        proposal_cppn, proposal_img, proposal_order, _, success = elliptical_slice_sample_continuous(
            live_points[seed_idx][0], threshold, config.image_size, order_fn
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        total_samples += 1

        if best_order >= target_order and samples_to_target is None:
            samples_to_target = total_samples
            break

    if samples_to_target is None:
        samples_to_target = total_samples

    return {
        "total_samples": total_samples,
        "samples_to_target": samples_to_target,
        "max_order_achieved": float(best_order),
        "success": bool(best_order >= target_order),
    }


def run_two_stage_sampling(
    order_fn: Callable,
    target_order: float,
    stage1_budget: int,
    config: ExperimentConfig,
) -> Dict:
    n_live_stage1 = config.n_live_stage1
    live_points = []
    best_order = 0.0
    collected_weights = []
    samples_at_target = None
    init_hit = False
    stage1_hit = False
    stage2_used = False

    for _ in range(n_live_stage1):
        cppn = CPPN()
        img = render_continuous(cppn, config.image_size)
        order = order_fn(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        best_order = max(best_order, order)

    total_samples = n_live_stage1
    if best_order >= target_order:
        samples_at_target = total_samples
        init_hit = True
        stage1_hit = True

    stage1_iters = max(0, stage1_budget - n_live_stage1)
    for _ in range(stage1_iters):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_cppn, proposal_img, proposal_order, _, success = elliptical_slice_sample_continuous(
            live_points[seed_idx][0], threshold, config.image_size, order_fn
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            best_order = max(best_order, proposal_order)

        total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples
            stage1_hit = True
            break

    if samples_at_target is None and len(collected_weights) >= 2:
        stage2_used = True
        W = np.array(collected_weights)
        W_mean = W.mean(axis=0)
        W_centered = W - W_mean
        _, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
        n_comp = min(config.pca_components, len(S))
        pca_components_mat = Vt[:n_comp]
        pca_mean = W_mean

        for iteration in range(config.max_iterations_stage2):
            orders = np.array([lp[2] for lp in live_points], dtype=float)
            q = config.anneal_quantile * (iteration + 1) / max(1, config.max_iterations_stage2)
            threshold = float(np.quantile(orders, q))
            worst_idx = int(np.argmin(orders))

            if np.random.rand() < config.elite_prob:
                elite_count = max(1, int(config.elite_fraction * n_live_stage1))
                elite_idx = np.argsort(orders)[-elite_count:]
                seed_idx = int(np.random.choice(elite_idx))
            else:
                seed_idx = int(np.random.randint(0, n_live_stage1))

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = pca_components_mat @ (current_w - pca_mean)
            t = (iteration + 1) / max(1, config.max_iterations_stage2)
            step_scale = (config.pca_step_scale_start * (1 - t)) + (config.pca_step_scale_end * t)
            delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * step_scale
            proposal_w = pca_mean + (pca_components_mat.T @ (coeffs + delta))

            proposal_cppn = live_points[seed_idx][0].copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = render_continuous(proposal_cppn, config.image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)

            total_samples += 1

            if best_order >= target_order:
                samples_at_target = total_samples
                break

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        "total_samples": total_samples,
        "samples_to_target": samples_at_target,
        "max_order_achieved": float(best_order),
        "success": bool(best_order >= target_order),
        "init_hit": init_hit,
        "stage1_hit": stage1_hit,
        "stage2_used": stage2_used,
    }


def worker_task(args) -> Dict:
    cppn_id, objective_name, target_order, config = args
    set_global_seed(config.seed + cppn_id * 1000)
    order_fn = OBJECTIVE_FUNCS[objective_name]

    baseline = run_baseline_single_stage(order_fn, target_order, config)
    variants = {}
    for budget in config.stage1_budgets:
        key = f"budget_{budget}"
        result = run_two_stage_sampling(order_fn, target_order, budget, config)
        speedup = baseline["samples_to_target"] / result["samples_to_target"] if result["samples_to_target"] > 0 else 0.0
        result["speedup"] = float(speedup)
        variants[key] = result

    best_speedup = max(v["speedup"] for v in variants.values()) if variants else 0.0
    return {
        "cppn_id": cppn_id,
        "baseline": baseline,
        "variants": variants,
        "best_speedup": float(best_speedup),
    }


def analyze_objective(rows: List[Dict]) -> Dict:
    all_speedups = []
    best_speedups = []
    variant_keys = list(rows[0]["variants"].keys()) if rows else []

    for row in rows:
        for key in variant_keys:
            all_speedups.append(row["variants"][key]["speedup"])
        best_speedups.append(row["best_speedup"])

    summary = {
        "n_total": len(rows),
        "speedup_mean": float(np.mean(best_speedups)) if best_speedups else 0.0,
        "speedup_std": float(np.std(best_speedups)) if len(best_speedups) > 1 else 0.0,
        "speedup_min": float(np.min(best_speedups)) if best_speedups else 0.0,
        "speedup_max": float(np.max(best_speedups)) if best_speedups else 0.0,
        "all_speedups": all_speedups,
        "best_speedups": best_speedups,
    }

    per_variant = {}
    for key in variant_keys:
        speedups = [r["variants"][key]["speedup"] for r in rows]
        init_hits = sum(1 for r in rows if r["variants"][key].get("init_hit"))
        stage1_hits = sum(1 for r in rows if r["variants"][key].get("stage1_hit"))
        stage2_used = sum(1 for r in rows if r["variants"][key].get("stage2_used"))
        stage2_speedups = [r["variants"][key]["speedup"] for r in rows if r["variants"][key].get("stage2_used")]
        per_variant[key] = {
            "speedup_mean": float(np.mean(speedups)) if speedups else 0.0,
            "speedup_std": float(np.std(speedups)) if len(speedups) > 1 else 0.0,
            "speedup_min": float(np.min(speedups)) if speedups else 0.0,
            "speedup_max": float(np.max(speedups)) if speedups else 0.0,
            "init_hit_count": int(init_hits),
            "stage1_hit_count": int(stage1_hits),
            "stage2_used_count": int(stage2_used),
            "stage2_speedup_mean": float(np.mean(stage2_speedups)) if stage2_speedups else 0.0,
        }

    summary["variants"] = per_variant
    return summary


def run_experiment(config: ExperimentConfig) -> Dict:
    results = {
        "config": {
            "n_cppns": config.n_cppns,
            "n_workers": config.n_workers,
            "image_size": config.image_size,
            "n_live": config.n_live,
            "baseline_max_iterations": config.baseline_max_iterations,
            "n_live_stage1": config.n_live_stage1,
            "stage1_budgets": list(config.stage1_budgets),
            "max_iterations_stage2": config.max_iterations_stage2,
            "pca_components": config.pca_components,
            "calibration_samples": config.calibration_samples,
            "target_hit_baseline": config.target_hit_baseline,
            "anneal_quantile": config.anneal_quantile,
            "elite_fraction": config.elite_fraction,
            "elite_prob": config.elite_prob,
            "pca_step_scale_start": config.pca_step_scale_start,
            "pca_step_scale_end": config.pca_step_scale_end,
            "objective_names": list(config.objective_names),
        },
        "calibration": {},
        "results": {},
    }

    for idx, objective_name in enumerate(config.objective_names):
        order_fn = OBJECTIVE_FUNCS[objective_name]
        calibration = calibrate_target(order_fn, config, config.seed + idx * 100)
        target_order = calibration["target_order"]

        args = [
            (cppn_id, objective_name, target_order, config)
            for cppn_id in range(config.n_cppns)
        ]
        with Pool(processes=config.n_workers) as pool:
            rows = pool.map(worker_task, args)

        results["calibration"][objective_name] = calibration
        results["results"][objective_name] = {
            "rows": rows,
            "summary": analyze_objective(rows),
        }
        logger.info("Objective %s done: mean speedup %.2f×", objective_name, results["results"][objective_name]["summary"]["speedup_mean"])

    return results


def main():
    print("=" * 80)
    print("RES-291: CPPN Alternate Objectives")
    print("=" * 80)

    config = ExperimentConfig()
    results = run_experiment(config)

    results_dir = project_root / "results" / "cppn_task_generalization"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "res_291_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
