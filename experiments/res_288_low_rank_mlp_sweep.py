#!/usr/bin/env python3
"""
RES-288: Two-Stage Sampling vs Low-Rank MLPs
============================================

Hypothesis: Two-stage sampling helps when intrinsic dimension is low.
We test LowRankCoordMLP with rank sweep and calibrated targets.
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
    stage1_budget: int = 150
    max_iterations_stage2: int = 300

    rank_values: Tuple[int, ...] = (2, 4, 8, 16)
    pca_components_max: int = 20
    pca_components_scale: int = 2

    calibration_samples: int = 1200
    target_quantile: float = 0.995


class LowRankCoordMLP:
    """
    Low-rank coordinate MLP: (x, y, r, bias) -> hidden(64) -> hidden(32) -> output(1)
    Each weight matrix W is factorized as A @ B with rank r.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.A1 = np.random.randn(4, rank) * PRIOR_SIGMA
        self.B1 = np.random.randn(rank, 64) * PRIOR_SIGMA
        self.b1 = np.random.randn(64) * PRIOR_SIGMA

        self.A2 = np.random.randn(64, rank) * PRIOR_SIGMA
        self.B2 = np.random.randn(rank, 32) * PRIOR_SIGMA
        self.b2 = np.random.randn(32) * PRIOR_SIGMA

        self.A3 = np.random.randn(32, rank) * PRIOR_SIGMA
        self.B3 = np.random.randn(rank, 1) * PRIOR_SIGMA
        self.b3 = np.random.randn(1) * PRIOR_SIGMA

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = np.stack([x, y, r, bias], axis=-1)
        original_shape = inputs.shape[:-1]
        inputs_flat = inputs.reshape(-1, 4)

        W1 = self.A1 @ self.B1
        h1 = np.tanh(inputs_flat @ W1 + self.b1)

        W2 = self.A2 @ self.B2
        h2 = np.tanh(h1 @ W2 + self.b2)

        W3 = self.A3 @ self.B3
        out = 1 / (1 + np.exp(-np.clip(h2 @ W3 + self.b3, -10, 10)))

        return out.reshape(original_shape)

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.A1.flatten(), self.B1.flatten(), self.b1,
            self.A2.flatten(), self.B2.flatten(), self.b2,
            self.A3.flatten(), self.B3.flatten(), self.b3,
        ])

    def set_weights(self, w: np.ndarray):
        idx = 0
        n = 4 * self.rank
        self.A1 = w[idx:idx+n].reshape(4, self.rank)
        idx += n
        n = self.rank * 64
        self.B1 = w[idx:idx+n].reshape(self.rank, 64)
        idx += n
        self.b1 = w[idx:idx+64]
        idx += 64

        n = 64 * self.rank
        self.A2 = w[idx:idx+n].reshape(64, self.rank)
        idx += n
        n = self.rank * 32
        self.B2 = w[idx:idx+n].reshape(self.rank, 32)
        idx += n
        self.b2 = w[idx:idx+32]
        idx += 32

        n = 32 * self.rank
        self.A3 = w[idx:idx+n].reshape(32, self.rank)
        idx += n
        n = self.rank * 1
        self.B3 = w[idx:idx+n].reshape(self.rank, 1)
        idx += n
        self.b3 = w[idx:idx+1]

    def copy(self) -> "LowRankCoordMLP":
        clone = LowRankCoordMLP.__new__(LowRankCoordMLP)
        clone.rank = self.rank
        clone.A1 = self.A1.copy()
        clone.B1 = self.B1.copy()
        clone.b1 = self.b1.copy()
        clone.A2 = self.A2.copy()
        clone.B2 = self.B2.copy()
        clone.b2 = self.b2.copy()
        clone.A3 = self.A3.copy()
        clone.B3 = self.B3.copy()
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


def run_baseline_single_stage(
    generator_factory: Callable[[], LowRankCoordMLP],
    target_order: float,
    config: ExperimentConfig,
) -> Dict:
    live_points = []
    best_order = 0.0
    n_live = config.n_live

    for _ in range(n_live):
        gen = generator_factory()
        img = gen.render(config.image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        best_order = max(best_order, order)

    total_samples = n_live
    samples_to_target = total_samples if best_order >= target_order else None

    for _ in range(config.baseline_max_iterations):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

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


def run_two_stage_sampling(
    generator_factory: Callable[[], LowRankCoordMLP],
    target_order: float,
    pca_components: int,
    config: ExperimentConfig,
) -> Dict:
    n_live_stage1 = config.n_live_stage1
    live_points = []
    best_order = 0.0
    collected_weights = []
    samples_at_target = None

    for _ in range(n_live_stage1):
        gen = generator_factory()
        img = gen.render(config.image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        collected_weights.append(gen.get_weights())
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
            collected_weights.append(proposal_gen.get_weights())

        total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples
            break

    if samples_at_target is None and len(collected_weights) >= 2:
        W = np.array(collected_weights)
        W_mean = W.mean(axis=0)
        W_centered = W - W_mean
        _, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
        n_comp = min(pca_components, len(S))
        pca_components_mat = Vt[:n_comp]
        pca_mean = W_mean

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
    }


def calibrate_target(
    generator_factory: Callable[[], LowRankCoordMLP],
    config: ExperimentConfig,
    seed: int,
) -> Dict:
    set_global_seed(seed)
    orders = []
    for _ in range(config.calibration_samples):
        gen = generator_factory()
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


def worker_task(args) -> Dict:
    rank, gen_id, target_order, pca_components, config = args
    set_global_seed(config.seed + rank * 10000 + gen_id * 100)
    factory = lambda: LowRankCoordMLP(rank)

    baseline = run_baseline_single_stage(factory, target_order, config)
    two_stage = run_two_stage_sampling(factory, target_order, pca_components, config)

    speedup = baseline["samples_to_target"] / two_stage["samples_to_target"] if two_stage["samples_to_target"] > 0 else 0.0

    return {
        "rank": rank,
        "gen_id": gen_id,
        "baseline": baseline,
        "two_stage": two_stage,
        "speedup": float(speedup),
    }


def analyze_rank_results(rows: List[Dict]) -> Dict:
    speedups = [r["speedup"] for r in rows]
    success_baseline = sum(1 for r in rows if r["baseline"]["success"])
    success_two_stage = sum(1 for r in rows if r["two_stage"]["success"])
    return {
        "n": len(rows),
        "speedup_mean": float(np.mean(speedups)) if speedups else 0.0,
        "speedup_std": float(np.std(speedups)) if len(speedups) > 1 else 0.0,
        "speedup_min": float(np.min(speedups)) if speedups else 0.0,
        "speedup_max": float(np.max(speedups)) if speedups else 0.0,
        "baseline_success": success_baseline,
        "two_stage_success": success_two_stage,
        "speedups": speedups,
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    results = {
        "config": {
            "n_generators": config.n_generators,
            "n_workers": config.n_workers,
            "image_size": config.image_size,
            "n_live": config.n_live,
            "baseline_max_iterations": config.baseline_max_iterations,
            "n_live_stage1": config.n_live_stage1,
            "stage1_budget": config.stage1_budget,
            "max_iterations_stage2": config.max_iterations_stage2,
            "rank_values": list(config.rank_values),
            "pca_components_max": config.pca_components_max,
            "pca_components_scale": config.pca_components_scale,
            "calibration_samples": config.calibration_samples,
            "target_quantile": config.target_quantile,
        },
        "calibration": {},
        "results_by_rank": {},
    }

    for rank in config.rank_values:
        pca_components = min(config.pca_components_max, rank * config.pca_components_scale)
        calibration = calibrate_target(lambda: LowRankCoordMLP(rank), config, config.seed + rank)
        target_order = calibration["target_order"]

        args = [
            (rank, gen_id, target_order, pca_components, config)
            for gen_id in range(config.n_generators)
        ]
        with Pool(processes=config.n_workers) as pool:
            rows = pool.map(worker_task, args)

        results["calibration"][str(rank)] = {
            **calibration,
            "pca_components": pca_components,
        }
        results["results_by_rank"][str(rank)] = {
            "rows": rows,
            "summary": analyze_rank_results(rows),
        }
        logger.info("Rank %d done: mean speedup %.2f×", rank, results["results_by_rank"][str(rank)]["summary"]["speedup_mean"])

    return results


def main():
    print("=" * 80)
    print("RES-288: Low-Rank MLP Sweep")
    print("=" * 80)

    config = ExperimentConfig()
    results = run_experiment(config)

    results_dir = project_root / "results" / "low_rank_mlp_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "res_288_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
