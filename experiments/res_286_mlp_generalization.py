#!/usr/bin/env python3
"""
RES-286: Two-Stage Sampling Generalization to MLPs
===================================================

Hypothesis: Two-stage sampling (explore → learn manifold → exploit) works for
neural network priors beyond CPPNs. If MLPs show similar speedup, the technique
generalizes to any structured weight space.

Method:
1. Create coordinate-based MLP (input: x,y → output: grayscale) matching CPPN interface
2. Run same two-stage experiment as RES-283 with MLPs instead of CPPNs
3. Compare speedup: MLP vs CPPN

Key Question: Is the speedup CPPN-specific or does it generalize?

Expected:
- If MLP speedup ≈ CPPN speedup (2-4×): Technique generalizes
- If MLP speedup ≈ 1×: Effect is CPPN-specific

Runtime: ~15-20 min on M4 Max (20 MLPs, 4 workers)
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging
from multiprocessing import Pool

# ML for analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Ensure project root is in path
local_path = Path('/Users/matt/Development/monochrome_noise_converger')
if local_path.exists():
    project_root = local_path
else:
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Project imports
from core.thermo_sampler_v3 import (
    order_multiplicative, set_global_seed, PRIOR_SIGMA
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# COORDINATE-BASED MLP (matches CPPN interface)
# ============================================================================

class CoordMLP:
    """
    Coordinate-based MLP for structured image generation.
    Same interface as CPPN: render(), get_weights(), set_weights(), copy()

    Architecture: (x, y, r, bias) → hidden(64) → hidden(32) → output(1)
    Activations: tanh hidden, sigmoid output (like CPPN)
    Total params: 4*64 + 64 + 64*32 + 32 + 32*1 + 1 = 256 + 64 + 2048 + 32 + 32 + 1 = 2433
    """

    def __init__(self):
        # Layer 1: 4 inputs (x, y, r, bias) → 64 hidden
        self.W1 = np.random.randn(4, 64) * PRIOR_SIGMA
        self.b1 = np.random.randn(64) * PRIOR_SIGMA

        # Layer 2: 64 → 32 hidden
        self.W2 = np.random.randn(64, 32) * PRIOR_SIGMA
        self.b2 = np.random.randn(32) * PRIOR_SIGMA

        # Output: 32 → 1
        self.W3 = np.random.randn(32, 1) * PRIOR_SIGMA
        self.b3 = np.random.randn(1) * PRIOR_SIGMA

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Forward pass for coordinate grid."""
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)

        # Stack inputs: (H, W, 4)
        inputs = np.stack([x, y, r, bias], axis=-1)
        original_shape = inputs.shape[:-1]

        # Flatten for matrix multiply: (H*W, 4)
        inputs_flat = inputs.reshape(-1, 4)

        # Layer 1: tanh
        h1 = np.tanh(inputs_flat @ self.W1 + self.b1)

        # Layer 2: tanh
        h2 = np.tanh(h1 @ self.W2 + self.b2)

        # Output: sigmoid
        out = 1 / (1 + np.exp(-np.clip(h2 @ self.W3 + self.b3, -10, 10)))

        return out.reshape(original_shape)

    def render(self, size: int = 32) -> np.ndarray:
        """Render binary image (threshold at 0.5)."""
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into single vector."""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_weights(self, w: np.ndarray):
        """Set weights from flattened vector."""
        idx = 0

        # W1: 4*64 = 256
        self.W1 = w[idx:idx+256].reshape(4, 64)
        idx += 256

        # b1: 64
        self.b1 = w[idx:idx+64]
        idx += 64

        # W2: 64*32 = 2048
        self.W2 = w[idx:idx+2048].reshape(64, 32)
        idx += 2048

        # b2: 32
        self.b2 = w[idx:idx+32]
        idx += 32

        # W3: 32*1 = 32
        self.W3 = w[idx:idx+32].reshape(32, 1)
        idx += 32

        # b3: 1
        self.b3 = w[idx:idx+1]

    def copy(self) -> 'CoordMLP':
        """Deep copy."""
        new_mlp = CoordMLP.__new__(CoordMLP)
        new_mlp.W1 = self.W1.copy()
        new_mlp.b1 = self.b1.copy()
        new_mlp.W2 = self.W2.copy()
        new_mlp.b2 = self.b2.copy()
        new_mlp.W3 = self.W3.copy()
        new_mlp.b3 = self.b3.copy()
        return new_mlp


# ============================================================================
# ELLIPTICAL SLICE SAMPLING (adapted for any generator with CPPN interface)
# ============================================================================

def elliptical_slice_sample_generic(
    generator,  # Any object with get_weights, set_weights, copy, render
    threshold: float,
    image_size: int,
    order_fn,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple:
    """
    Elliptical slice sampling for any generator matching CPPN interface.
    Returns: (generator, image, order, n_contractions, success)
    """
    current_w = generator.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for restart in range(max_restarts):
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
                return (proposal_gen, proposal_img, proposal_order, total_contractions + n_contractions, True)

            n_contractions += 1
            total_contractions += 1

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi
            phi = np.random.uniform(phi_min, phi_max)

    # Failed - return original
    original_img = generator.render(image_size)
    original_order = order_fn(original_img)
    return (generator, original_img, original_order, total_contractions, False)


# ============================================================================
# EXPERIMENT CONFIG
# ============================================================================

@dataclass
class ExperimentConfig:
    n_generators: int = 20              # 20 for quick test
    order_target: float = 0.40          # Same as RES-283
    baseline_n_live: int = 100
    baseline_max_iterations: int = 1000
    stage1_budgets: list = None
    strategies: list = None
    n_workers: int = 4
    pca_components: int = 3
    max_iterations_stage2: int = 500
    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.stage1_budgets is None:
            self.stage1_budgets = [50, 100]
        if self.strategies is None:
            self.strategies = ['uniform']


# ============================================================================
# SAMPLING FUNCTIONS (same logic as RES-283, but generator-agnostic)
# ============================================================================

def run_baseline_single_stage(generator_class, target_order, image_size, n_live, max_iterations) -> Dict:
    """Run baseline nested sampling with fresh random generators."""
    live_points = []
    best_order = 0.0
    samples_to_target = None

    for _ in range(n_live):
        gen = generator_class()  # New random generator from prior
        img = gen.render(image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_to_target = n_live

    for iteration in range(max_iterations):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live)

        proposal_gen, proposal_img, proposal_order, _, success = elliptical_slice_sample_generic(
            live_points[seed_idx][0], worst_order, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    if samples_to_target is None:
        samples_to_target = n_live + max_iterations

    return {
        'total_samples': samples_to_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }


def run_two_stage_sampling(generator_class, target_order, image_size, stage1_budget,
                           max_iterations_stage2, strategy='uniform', pca_components=3) -> Dict:
    """Run two-stage sampling with PCA manifold learning."""
    n_live_stage1 = 50
    live_points = []
    best_order = 0.0
    collected_weights = []
    samples_at_target = None

    # Stage 1: Exploration
    for _ in range(n_live_stage1):
        gen = generator_class()
        img = gen.render(image_size)
        order = order_multiplicative(img)
        live_points.append((gen, img, order))
        collected_weights.append(gen.get_weights())
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    stage1_samples = n_live_stage1

    for iteration in range(max(0, stage1_budget - n_live_stage1)):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_gen, proposal_img, proposal_order, _, success = elliptical_slice_sample_generic(
            live_points[seed_idx][0], worst_order, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_gen.get_weights())
            stage1_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples

    # PCA on collected weights
    if len(collected_weights) >= 2:
        W = np.array(collected_weights)
        W_mean = W.mean(axis=0)
        W_centered = W - W_mean
        U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
        n_comp = min(pca_components, len(S))
        pca_components_mat = Vt[:n_comp]
        pca_mean = W_mean
    else:
        pca_mean = pca_components_mat = None

    total_samples = stage1_samples

    # Stage 2: Manifold-constrained sampling
    if pca_mean is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            seed_idx = np.random.randint(0, n_live_stage1)

            current_w = live_points[seed_idx][0].get_weights()
            w_centered = current_w - pca_mean
            coeffs = pca_components_mat @ w_centered

            delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
            new_coeffs = coeffs + delta_coeffs
            proposal_w = pca_mean + (pca_components_mat.T @ new_coeffs)

            proposal_gen = live_points[seed_idx][0].copy()
            proposal_gen.set_weights(proposal_w)
            proposal_img = proposal_gen.render(image_size)
            proposal_order = order_multiplicative(proposal_img)

            if proposal_order >= worst_order:
                live_points[worst_idx] = (proposal_gen, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)

            total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'stage1_samples': stage1_samples,
        'stage2_samples': total_samples - stage1_samples,
        'total_samples': total_samples,
        'samples_to_target': samples_at_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }


# ============================================================================
# WORKER AND PARALLEL EXECUTION
# ============================================================================

def worker_task(args: Tuple) -> Dict:
    """Worker task for parallel execution."""
    gen_id, generator_class_name, config = args

    # Get the generator class
    if generator_class_name == 'CoordMLP':
        generator_class = CoordMLP
    else:
        from core.thermo_sampler_v3 import CPPN
        generator_class = CPPN

    try:
        set_global_seed(gen_id)

        baseline = run_baseline_single_stage(
            generator_class, config.order_target, config.image_size,
            config.baseline_n_live, config.baseline_max_iterations
        )

        variants_results = {}
        for budget in config.stage1_budgets:
            for strategy in config.strategies:
                key = f"budget_{budget}_strategy_{strategy}"
                result = run_two_stage_sampling(
                    generator_class, config.order_target, config.image_size,
                    budget, config.max_iterations_stage2, strategy=strategy
                )
                result['speedup'] = baseline['total_samples'] / result['samples_to_target'] if result['samples_to_target'] > 0 else 0
                variants_results[key] = result

        return {
            'gen_id': gen_id,
            'generator_type': generator_class_name,
            'baseline_samples': baseline['total_samples'],
            'baseline_success': baseline['success'],
            'variants': variants_results,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in {generator_class_name} {gen_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'gen_id': gen_id, 'generator_type': generator_class_name, 'error': str(e), 'success': False}


def run_parallel_experiments(generator_class_name: str, config: ExperimentConfig) -> List[Dict]:
    """Run experiments in parallel."""
    logger.info(f"Running {config.n_generators} {generator_class_name}s with {config.n_workers} workers")

    jobs = [(gen_id, generator_class_name, config) for gen_id in range(config.n_generators)]
    results = []

    with Pool(config.n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_task, jobs)):
            results.append(result)
            if (i + 1) % 5 == 0:
                logger.info(f"  Progress: {i+1}/{config.n_generators} {generator_class_name}s")

    return results


def analyze_results(results: List[Dict], generator_type: str) -> Dict:
    """Analyze speedup results."""
    speedups = []
    best_speedups = []

    for result in results:
        if not result['success']:
            continue

        best_speedup = 0
        for variant_key, variant_result in result['variants'].items():
            speedup = variant_result['speedup']
            speedups.append(speedup)
            best_speedup = max(best_speedup, speedup)

        best_speedups.append(best_speedup)

    if not speedups:
        return {'error': 'No successful results'}

    return {
        'generator_type': generator_type,
        'n_successful': len([r for r in results if r['success']]),
        'n_total': len(results),
        'speedup_mean': float(np.mean(best_speedups)),
        'speedup_std': float(np.std(best_speedups)) if len(best_speedups) > 1 else 0,
        'speedup_min': float(np.min(best_speedups)),
        'speedup_max': float(np.max(best_speedups)),
        'speedup_median': float(np.median(best_speedups)),
        'all_speedups': speedups,
        'best_speedups': best_speedups
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("RES-286: Two-Stage Sampling Generalization (MLP vs CPPN)")
    print("=" * 80)

    config = ExperimentConfig()
    start_time = time.time()

    # Run MLP experiments
    logger.info("\n[1/3] Running MLP experiments...")
    mlp_start = time.time()
    mlp_results = run_parallel_experiments('CoordMLP', config)
    mlp_time = time.time() - mlp_start
    mlp_analysis = analyze_results(mlp_results, 'CoordMLP')
    logger.info(f"  MLP done in {mlp_time:.1f}s - Mean speedup: {mlp_analysis.get('speedup_mean', 0):.2f}×")

    # Run CPPN experiments (for comparison)
    logger.info("\n[2/3] Running CPPN experiments...")
    cppn_start = time.time()
    cppn_results = run_parallel_experiments('CPPN', config)
    cppn_time = time.time() - cppn_start
    cppn_analysis = analyze_results(cppn_results, 'CPPN')
    logger.info(f"  CPPN done in {cppn_time:.1f}s - Mean speedup: {cppn_analysis.get('speedup_mean', 0):.2f}×")

    # Save results
    logger.info("\n[3/3] Saving results...")
    results_dir = project_root / "results" / "mlp_generalization"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'config': {
            'n_generators': config.n_generators,
            'order_target': config.order_target,
            'stage1_budgets': config.stage1_budgets,
            'image_size': config.image_size
        },
        'mlp': mlp_analysis,
        'cppn': cppn_analysis,
        'comparison': {
            'mlp_speedup': mlp_analysis.get('speedup_mean', 0),
            'cppn_speedup': cppn_analysis.get('speedup_mean', 0),
            'ratio': mlp_analysis.get('speedup_mean', 0) / max(cppn_analysis.get('speedup_mean', 0), 0.01),
            'generalizes': mlp_analysis.get('speedup_mean', 0) > 1.5  # If MLP speedup > 1.5×, it generalizes
        },
        'timing': {
            'mlp_seconds': mlp_time,
            'cppn_seconds': cppn_time,
            'total_seconds': time.time() - start_time
        }
    }

    with open(results_dir / "res_286_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Generator':<15} {'Mean Speedup':<15} {'Std':<10} {'Min-Max':<15}")
    print("-" * 55)
    print(f"{'MLP':<15} {mlp_analysis.get('speedup_mean', 0):>10.2f}× {mlp_analysis.get('speedup_std', 0):>10.2f} {mlp_analysis.get('speedup_min', 0):.1f}-{mlp_analysis.get('speedup_max', 0):.1f}×")
    print(f"{'CPPN':<15} {cppn_analysis.get('speedup_mean', 0):>10.2f}× {cppn_analysis.get('speedup_std', 0):>10.2f} {cppn_analysis.get('speedup_min', 0):.1f}-{cppn_analysis.get('speedup_max', 0):.1f}×")

    print(f"\nConclusion: ", end="")
    if output['comparison']['generalizes']:
        print(f"✓ Two-stage sampling GENERALIZES (MLP speedup = {mlp_analysis.get('speedup_mean', 0):.2f}×)")
    else:
        print(f"✗ Two-stage sampling may be CPPN-specific (MLP speedup = {mlp_analysis.get('speedup_mean', 0):.2f}×)")

    print(f"\nTotal runtime: {output['timing']['total_seconds']:.1f}s")
    print(f"Results saved to: {results_dir / 'res_286_results.json'}")


if __name__ == "__main__":
    main()
