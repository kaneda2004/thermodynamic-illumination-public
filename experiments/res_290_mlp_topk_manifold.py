#!/usr/bin/env python3
"""
RES-290: MLP Two-Stage with Top-K Manifold Learning
===================================================

Hypothesis: For high-D MLPs, PCA should be fit on top-K highest-order
stage-1 samples rather than all samples. This should improve Stage 2.
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

from core.thermo_sampler_v3 import (
    order_multiplicative,
    set_global_seed,
    PRIOR_SIGMA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    n_generators: int = 30
    n_workers: int = 6
    seed: int = 42

    image_size: int = 32
    n_live: int = 100
    baseline_max_iterations: int = 1200

    n_live_stage1: int = 50
    stage1_budget: int = 200
    max_iterations_stage2: int = 300

    top_k_values: Tuple[int, ...] = (25, 50, 100)
    pca_components_values: Tuple[int, ...] = (10, 20, 50)

    calibration_samples: int = 1500
    target_quantile: float = 0.995


class CoordMLP:
    """Coordinate-based MLP (matches CPPN interface)."""

    def __init__(self):
        self.W1 = np.random.randn(4, 64) * PRIOR_SIGMA
        self.b1 = np.random.randn(64) * PRIOR_SIGMA
        self.W2 = np.random.randn(64, 32) * PRIOR_SIGMA
        self.b2 = np.random.randn(32) * PRIOR_SIGMA
        self.W3 = np.random.randn(32, 1) * PRIOR_SIGMA
        self.b3 = np.random.randn(1) * PRIOR_SIGMA

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = np.stack([x, y, r, bias], axis=-1)
        original_shape = inputs.shape[:-1]
        inputs_flat = inputs.reshape(-1, 4)

        h1 = np.tanh(inputs_flat @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = 1 / (1 + np.exp(-np.clip(h2 @ self.W3 + self.b3, -10, 10)))
        return out.reshape(original_shape)

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3,
        ])

    def set_weights(self, w: np.ndarray):
        idx = 0
        self.W1 = w[idx:idx+256].reshape(4, 64)
        idx += 256
        self.b1 = w[idx:idx+64]
        idx += 64
        self.W2 = w[idx:idx+2048].reshape(64, 32)
        idx += 2048
        self.b2 = w[idx:idx+32]
        idx += 32
        self.W3 = w[idx:idx+32].reshape(32, 1)
        idx += 32
        self.b3 = w[idx:idx+1]

    def copy(self) -> "CoordMLP":
        clone = CoordMLP.__new__(CoordMLP)
        clone.W1 = self.W1.copy()
        clone.b1 = self.b1.copy()
        clone.W2 = self.W2.copy()
        clone.b2 = self.b2.copy()
        clone.W3 = self.W3.copy()
        clone.b3 = self.b3.copy()
        return clone


def elliptical_slice_sample_generic(
    generator,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5,
) -> Tuple:
    current_w = generator.get_weights()
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
            proposal_gen = generator.copy()
            proposal_gen.set_weights(proposal_w)
            proposal_img = proposal_gen.render(image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                return proposal_gen, proposal_img, proposal_order, total_contractions + n_contractions, True

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi
            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

        total_contractions += n_contractions

    current_img = generator.render(image_size)
    return generator, current_img, order_fn(current_img), total_contractions, False


def calibrate_target(config: ExperimentConfig, seed: int) -> Dict:
    set_global_seed(seed)
    orders = []
    for _ in range(config.calibration_samples):
        gen = CoordMLP()
        img = gen.render(config.image_size)
        orders.append(order_multiplicative(img))
    orders = np.array(orders, dtype=float)
    target = float(np.quantile(orders, config.target_quantile))
    p = 1.0 - config.target_quantile
    hit_baseline = 1 - (1 - p) ** config.n_live
    hit_stage1 = 1 - (1 - p) ** config.n_live_stage1
    return {
        "target_order": target,
        "order_mean": float(orders.mean()),
        "order_std": float(orders.std()),
        "quantile": config.target_quantile,
        "p_hit_baseline": float(hit_baseline),
        "p_hit_stage1": float(hit_stage1),
    }


def run_baseline_single_stage(target_order: float, config: ExperimentConfig) -> Dict:
    live_points = []
    best_order = 0.0
    for _ in range(config.n_live):
        gen = CoordMLP()
        img = gen.render(config.image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        best_order = max(best_order, order)

    total_samples = config.n_live
    samples_to_target = total_samples if best_order >= target_order else None

    for _ in range(config.baseline_max_iterations):
        worst_idx = min(range(config.n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, config.n_live)

        proposal_gen, proposal_img, proposal_order, _, success = elliptical_slice_sample_generic(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
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


def run_two_stage_topk(
    target_order: float,
    top_k: int,
    pca_components: int,
    config: ExperimentConfig,
) -> Dict:
    n_live_stage1 = config.n_live_stage1
    live_points = []
    best_order = 0.0
    collected = []
    samples_at_target = None

    for _ in range(n_live_stage1):
        gen = CoordMLP()
        img = gen.render(config.image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        collected.append((gen.get_weights(), order))
        best_order = max(best_order, order)

    total_samples = n_live_stage1
    samples_at_target = total_samples if best_order >= target_order else None

    stage1_iters = max(0, config.stage1_budget - n_live_stage1)
    for _ in range(stage1_iters):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_gen, proposal_img, proposal_order, _, success = elliptical_slice_sample_generic(
            live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected.append((proposal_gen.get_weights(), proposal_order))

        total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples
            break

    if samples_at_target is None and collected:
        collected_sorted = sorted(collected, key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(collected_sorted))
        top_weights = [w for w, _ in collected_sorted[:top_k]]

        if len(top_weights) >= 2:
            W = np.array(top_weights)
            W_mean = W.mean(axis=0)
            W_centered = W - W_mean
            _, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
            n_comp = min(pca_components, len(S))
            pca_components_mat = Vt[:n_comp]
            pca_mean = W_mean
        else:
            pca_mean = pca_components_mat = None

        if pca_mean is not None:
            for _ in range(config.max_iterations_stage2):
                worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
                threshold = live_points[worst_idx][2]
                seed_idx = np.random.randint(0, n_live_stage1)

                current_w = live_points[seed_idx][0].get_weights()
                coeffs = pca_components_mat @ (current_w - pca_mean)
                delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                proposal_w = pca_mean + (pca_components_mat.T @ (coeffs + delta))

                proposal_gen = live_points[seed_idx][0].copy()
                proposal_gen.set_weights(proposal_w)
                proposal_img = proposal_gen.render(config.image_size)
                proposal_order = order_multiplicative(proposal_img)

                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
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
        "top_k_used": int(min(top_k, len(collected))),
    }


def worker_task(args) -> Dict:
    gen_id, target_order, config = args
    set_global_seed(config.seed + gen_id * 1000)
    baseline = run_baseline_single_stage(target_order, config)

    variants = {}
    for top_k in config.top_k_values:
        for pca_components in config.pca_components_values:
            key = f"topk_{top_k}_pca_{pca_components}"
            result = run_two_stage_topk(target_order, top_k, pca_components, config)
            speedup = baseline["samples_to_target"] / result["samples_to_target"] if result["samples_to_target"] > 0 else 0.0
            result["speedup"] = float(speedup)
            variants[key] = result

    return {
        "gen_id": gen_id,
        "baseline": baseline,
        "variants": variants,
    }


def analyze_variants(rows: List[Dict]) -> Dict:
    variant_keys = list(rows[0]["variants"].keys()) if rows else []
    summary = {}
    for key in variant_keys:
        speedups = [r["variants"][key]["speedup"] for r in rows]
        success = sum(1 for r in rows if r["variants"][key]["success"])
        summary[key] = {
            "speedup_mean": float(np.mean(speedups)) if speedups else 0.0,
            "speedup_std": float(np.std(speedups)) if len(speedups) > 1 else 0.0,
            "speedup_min": float(np.min(speedups)) if speedups else 0.0,
            "speedup_max": float(np.max(speedups)) if speedups else 0.0,
            "success_count": success,
            "speedups": speedups,
        }
    return summary


def run_experiment(config: ExperimentConfig) -> Dict:
    calibration = calibrate_target(config, config.seed + 7)
    target_order = calibration["target_order"]

    args = [(gen_id, target_order, config) for gen_id in range(config.n_generators)]
    with Pool(processes=config.n_workers) as pool:
        rows = pool.map(worker_task, args)

    return {
        "config": {
            "n_generators": config.n_generators,
            "n_workers": config.n_workers,
            "image_size": config.image_size,
            "n_live": config.n_live,
            "baseline_max_iterations": config.baseline_max_iterations,
            "n_live_stage1": config.n_live_stage1,
            "stage1_budget": config.stage1_budget,
            "max_iterations_stage2": config.max_iterations_stage2,
            "top_k_values": list(config.top_k_values),
            "pca_components_values": list(config.pca_components_values),
            "calibration_samples": config.calibration_samples,
            "target_quantile": config.target_quantile,
        },
        "calibration": calibration,
        "rows": rows,
        "summary": analyze_variants(rows),
    }


def main():
    print("=" * 80)
    print("RES-290: MLP Top-K Manifold")
    print("=" * 80)

    config = ExperimentConfig()
    results = run_experiment(config)

    results_dir = project_root / "results" / "mlp_topk_manifold"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "res_290_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
