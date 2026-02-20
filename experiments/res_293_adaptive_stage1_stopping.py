#!/usr/bin/env python3
"""
RES-293: Adaptive Stage 1 Stopping for Two-Stage Sampling

Hypothesis: Reducing Stage 1 from fixed 150 samples to adaptive (30-150) based on
PCA variance stabilization will improve mean speedup from 3.74× toward theoretical
ceiling by reducing overhead for "easy" CPPNs.

Method:
1. Track explained variance ratio during Stage 1 exploration
2. Stop early when variance stabilizes (>90% explained, or delta < 1%)
3. Compare against fixed 150-sample baseline
4. Measure speedup distribution across 50 CPPNs

Expected outcome: Mean speedup increases from 3.74× to 5-8× by eliminating
unnecessary Stage 1 samples for CPPNs with quickly-discoverable manifolds.

Usage:
    cd /Users/matt/Development/monochrome_noise_converger
    uv run python experiments/res_293_adaptive_stage1_stopping.py
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import zlib

# Ensure project root is in path
PROJECT_ROOT = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_CPPNS = 50
TARGET_ORDER = 0.40  # Harder target - requires actual sampling
MAX_ITERATIONS = 500
RANDOM_SEED = 42

# Adaptive stopping parameters to test
STOPPING_CONFIGS = [
    {'name': 'fixed_150', 'min_samples': 150, 'max_samples': 150, 'variance_threshold': 1.0},
    {'name': 'fixed_100', 'min_samples': 100, 'max_samples': 100, 'variance_threshold': 1.0},
    {'name': 'fixed_50', 'min_samples': 50, 'max_samples': 50, 'variance_threshold': 1.0},
    {'name': 'adaptive_90', 'min_samples': 30, 'max_samples': 150, 'variance_threshold': 0.90},
    {'name': 'adaptive_95', 'min_samples': 30, 'max_samples': 150, 'variance_threshold': 0.95},
    {'name': 'adaptive_delta', 'min_samples': 30, 'max_samples': 150, 'variance_threshold': 0.90, 'delta_threshold': 0.01},
]


# =============================================================================
# ORDER METRIC
# =============================================================================

def compute_order(img: np.ndarray) -> float:
    """Multiplicative order metric."""
    if img.size == 0:
        return 0.0

    # Density gate
    density = np.mean(img)
    density_gate = np.exp(-((density - 0.5) ** 2) / (2 * 0.25 ** 2))

    # Edge density
    binary = (img > 0.5).astype(np.uint8)
    padded = np.pad(binary, 1, mode='edge')
    edges = 0
    for di, dj in [(0, 1), (1, 0)]:
        shifted = padded[1+di:1+di+img.shape[0], 1+dj:1+dj+img.shape[1]]
        edges += np.sum(binary != shifted)
    edge_density = edges / (2 * img.size)
    edge_gate = np.exp(-((edge_density - 0.15) ** 2) / (2 * 0.08 ** 2))

    # Spectral coherence
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (min(h, w) / 4)
    coherence = np.sum(power[low_mask]) / (np.sum(power) + 1e-10)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Compressibility
    tiled = np.tile(binary, (2, 2))
    packed = np.packbits(tiled.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    compressibility = max(0, min(1, 1 - (len(compressed) * 8 / tiled.size)))
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    return float(min(1.0, density_gate * edge_gate * coherence_gate * compress_gate))


# =============================================================================
# CPPN IMPLEMENTATION
# =============================================================================

class CPPN:
    """Compositional Pattern Producing Network."""

    ACTIVATIONS = {
        'sin': lambda x: np.sin(x * np.pi),
        'cos': lambda x: np.cos(x * np.pi),
        'gauss': lambda x: np.exp(-x**2 * 2),
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10))),
        'tanh': np.tanh,
    }

    def __init__(self, size: int = 64):
        self.size = size
        self.n_inputs = 4
        self.weights = None
        self.activations = None

    def set_weights(self, weights: np.ndarray):
        """Set weights from flat vector."""
        idx = 0
        self.w1 = weights[idx:idx+32].reshape(4, 8); idx += 32
        self.b1 = weights[idx:idx+8]; idx += 8
        self.w2 = weights[idx:idx+64].reshape(8, 8); idx += 64
        self.b2 = weights[idx:idx+8]; idx += 8
        self.w3 = weights[idx:idx+8].reshape(8, 1); idx += 8
        self.b3 = weights[idx:idx+1]; idx += 1

        # Use deterministic activations based on weight values
        act_indices = (weights[:16] * 1000).astype(int) % 5
        act_names = list(self.ACTIVATIONS.keys())
        self.activations = [act_names[i] for i in act_indices]

    def get_weight_dim(self) -> int:
        return 32 + 8 + 64 + 8 + 8 + 1  # 121 parameters

    def generate(self) -> np.ndarray:
        """Generate image from current weights."""
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = np.stack([x.flatten(), y.flatten(), r.flatten(), bias.flatten()], axis=1)

        h1 = inputs @ self.w1 + self.b1
        for i in range(8):
            h1[:, i] = self.ACTIVATIONS[self.activations[i % len(self.activations)]](h1[:, i])

        h2 = h1 @ self.w2 + self.b2
        for i in range(8):
            h2[:, i] = self.ACTIVATIONS[self.activations[(8 + i) % len(self.activations)]](h2[:, i])

        out = self.ACTIVATIONS['sigmoid'](h2 @ self.w3 + self.b3)
        return out.reshape(self.size, self.size)


# =============================================================================
# NESTED SAMPLING WITH ADAPTIVE STAGE 1
# =============================================================================

@dataclass
class SamplingResult:
    """Results from one sampling run."""
    config_name: str
    cppn_id: int
    stage1_samples: int
    stage2_samples: int
    total_samples: int
    final_order: float
    target_reached: bool
    explained_variance: float
    speedup_vs_baseline: float
    runtime_seconds: float


def ellipsoidal_sample(points: np.ndarray, expansion: float = 1.5) -> np.ndarray:
    """Sample from ellipsoid bounding the points."""
    mean = np.mean(points, axis=0)
    centered = points - mean

    # Compute covariance with regularization
    cov = np.cov(centered.T) + 1e-6 * np.eye(centered.shape[1])

    # Sample from expanded ellipsoid
    L = np.linalg.cholesky(cov)
    u = np.random.randn(centered.shape[1])
    u = u / np.linalg.norm(u) * np.random.uniform(0, 1) ** (1/centered.shape[1])

    return mean + expansion * L @ u


def run_nested_sampling(
    cppn: CPPN,
    target_order: float,
    stage1_config: dict,
    max_iterations: int = 500,
    n_live: int = 50,
) -> Tuple[int, int, float, float, bool]:
    """
    Run two-stage nested sampling with configurable Stage 1 stopping.

    Returns: (stage1_samples, stage2_samples, final_order, explained_variance, target_reached)
    """
    dim = cppn.get_weight_dim()

    # Initialize live points
    live_points = np.random.randn(n_live, dim)
    live_orders = np.zeros(n_live)

    for i in range(n_live):
        cppn.set_weights(live_points[i])
        img = cppn.generate()
        live_orders[i] = compute_order((img > 0.5).astype(np.uint8))

    # Stage 1: Exploration with adaptive stopping
    stage1_samples = 0
    all_samples = [live_points.copy()]
    prev_variance = 0.0
    explained_variance = 0.0

    min_samples = stage1_config['min_samples']
    max_samples = stage1_config['max_samples']
    variance_threshold = stage1_config['variance_threshold']
    delta_threshold = stage1_config.get('delta_threshold', 0.0)

    for iteration in range(max_samples):
        # Find worst point
        worst_idx = np.argmin(live_orders)
        threshold = live_orders[worst_idx]

        # Check if target reached during Stage 1
        if np.max(live_orders) >= target_order:
            return stage1_samples, 0, float(np.max(live_orders)), explained_variance, True

        # Sample new point above threshold
        for _ in range(100):  # Max attempts
            if len(all_samples) > 1:
                all_points = np.vstack(all_samples)
                new_point = ellipsoidal_sample(all_points)
            else:
                new_point = np.random.randn(dim)

            cppn.set_weights(new_point)
            img = cppn.generate()
            new_order = compute_order((img > 0.5).astype(np.uint8))

            if new_order > threshold:
                live_points[worst_idx] = new_point
                live_orders[worst_idx] = new_order
                all_samples.append(new_point.reshape(1, -1))
                stage1_samples += 1
                break

        # Check adaptive stopping condition
        if iteration >= min_samples:
            all_points = np.vstack(all_samples)

            # PCA to check explained variance
            centered = all_points - np.mean(all_points, axis=0)
            try:
                _, s, _ = np.linalg.svd(centered, full_matrices=False)
                total_var = np.sum(s**2)
                explained_variance = np.sum(s[:2]**2) / total_var if total_var > 0 else 0
            except:
                explained_variance = 0

            # Check stopping conditions
            if explained_variance >= variance_threshold:
                break

            if delta_threshold > 0 and abs(explained_variance - prev_variance) < delta_threshold:
                break

            prev_variance = explained_variance

    # Compute PCA basis for Stage 2
    all_points = np.vstack(all_samples)
    centered = all_points - np.mean(all_points, axis=0)
    mean_point = np.mean(all_points, axis=0)

    try:
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_basis = Vt[:2]  # Top 2 components
    except:
        pca_basis = np.eye(dim)[:2]

    # Stage 2: Manifold-constrained sampling
    stage2_samples = 0

    for iteration in range(max_iterations - stage1_samples):
        # Find worst point
        worst_idx = np.argmin(live_orders)
        threshold = live_orders[worst_idx]

        # Check if target reached
        if np.max(live_orders) >= target_order:
            return stage1_samples, stage2_samples, float(np.max(live_orders)), explained_variance, True

        # Sample in PCA subspace
        for _ in range(100):
            # Project to 2D, perturb, project back
            coeffs = np.random.randn(2) * 0.5
            new_point = mean_point + coeffs @ pca_basis

            # Add small full-space noise
            new_point += np.random.randn(dim) * 0.1

            cppn.set_weights(new_point)
            img = cppn.generate()
            new_order = compute_order((img > 0.5).astype(np.uint8))

            if new_order > threshold:
                live_points[worst_idx] = new_point
                live_orders[worst_idx] = new_order
                stage2_samples += 1
                break

    return stage1_samples, stage2_samples, float(np.max(live_orders)), explained_variance, np.max(live_orders) >= target_order


def run_baseline(cppn: CPPN, target_order: float, max_iterations: int = 500) -> int:
    """Run single-stage nested sampling (baseline for speedup calculation)."""
    dim = cppn.get_weight_dim()
    n_live = 50

    live_points = np.random.randn(n_live, dim)
    live_orders = np.zeros(n_live)

    for i in range(n_live):
        cppn.set_weights(live_points[i])
        img = cppn.generate()
        live_orders[i] = compute_order((img > 0.5).astype(np.uint8))

    samples = 0
    all_samples = [live_points.copy()]

    for iteration in range(max_iterations):
        if np.max(live_orders) >= target_order:
            return samples

        worst_idx = np.argmin(live_orders)
        threshold = live_orders[worst_idx]

        for _ in range(100):
            if len(all_samples) > 1:
                all_points = np.vstack(all_samples)
                new_point = ellipsoidal_sample(all_points)
            else:
                new_point = np.random.randn(dim)

            cppn.set_weights(new_point)
            img = cppn.generate()
            new_order = compute_order((img > 0.5).astype(np.uint8))

            if new_order > threshold:
                live_points[worst_idx] = new_point
                live_orders[worst_idx] = new_order
                all_samples.append(new_point.reshape(1, -1))
                samples += 1
                break

    return samples


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("RES-293: Adaptive Stage 1 Stopping")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    output_dir = PROJECT_ROOT / 'results' / 'adaptive_stage1_stopping'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run experiment for each CPPN
    for cppn_id in range(N_CPPNS):
        print(f"\n--- CPPN {cppn_id + 1}/{N_CPPNS} ---")

        # Create CPPN with deterministic seed
        np.random.seed(RANDOM_SEED + cppn_id * 1000)
        cppn = CPPN(size=64)

        # Run baseline first
        np.random.seed(RANDOM_SEED + cppn_id * 1000 + 999)
        start = time.time()
        baseline_samples = run_baseline(cppn, TARGET_ORDER, MAX_ITERATIONS)
        baseline_time = time.time() - start
        print(f"  Baseline: {baseline_samples} samples ({baseline_time:.1f}s)")

        # Run each stopping configuration
        for config in STOPPING_CONFIGS:
            np.random.seed(RANDOM_SEED + cppn_id * 1000 + hash(config['name']) % 1000)

            start = time.time()
            s1, s2, final_order, exp_var, reached = run_nested_sampling(
                cppn, TARGET_ORDER, config, MAX_ITERATIONS
            )
            runtime = time.time() - start

            total = s1 + s2
            speedup = baseline_samples / total if total > 0 else 0

            result = SamplingResult(
                config_name=config['name'],
                cppn_id=cppn_id,
                stage1_samples=s1,
                stage2_samples=s2,
                total_samples=total,
                final_order=final_order,
                target_reached=reached,
                explained_variance=exp_var,
                speedup_vs_baseline=speedup,
                runtime_seconds=runtime,
            )
            results.append(asdict(result))

            print(f"  {config['name']:15s}: S1={s1:3d}, S2={s2:3d}, total={total:3d}, "
                  f"speedup={speedup:.2f}×, var={exp_var:.2f}")

    # Aggregate results by config
    print("\n" + "=" * 70)
    print("SUMMARY BY CONFIGURATION")
    print("=" * 70)

    summary = {}
    for config in STOPPING_CONFIGS:
        name = config['name']
        config_results = [r for r in results if r['config_name'] == name]

        speedups = [r['speedup_vs_baseline'] for r in config_results]
        s1_samples = [r['stage1_samples'] for r in config_results]
        total_samples = [r['total_samples'] for r in config_results]
        success_rate = np.mean([r['target_reached'] for r in config_results])

        summary[name] = {
            'speedup_mean': float(np.mean(speedups)),
            'speedup_std': float(np.std(speedups)),
            'speedup_min': float(np.min(speedups)),
            'speedup_max': float(np.max(speedups)),
            'stage1_mean': float(np.mean(s1_samples)),
            'total_mean': float(np.mean(total_samples)),
            'success_rate': float(success_rate),
        }

        print(f"\n{name}:")
        print(f"  Speedup: {summary[name]['speedup_mean']:.2f}× ± {summary[name]['speedup_std']:.2f}")
        print(f"  Range: {summary[name]['speedup_min']:.2f}× - {summary[name]['speedup_max']:.2f}×")
        print(f"  Stage 1 mean: {summary[name]['stage1_mean']:.1f} samples")
        print(f"  Total mean: {summary[name]['total_mean']:.1f} samples")
        print(f"  Success rate: {summary[name]['success_rate']:.1%}")

    # Save results
    output = {
        'experiment_id': 'RES-293',
        'hypothesis': 'Adaptive Stage 1 stopping improves speedup by reducing overhead',
        'config': {
            'n_cppns': N_CPPNS,
            'target_order': TARGET_ORDER,
            'max_iterations': MAX_ITERATIONS,
            'stopping_configs': STOPPING_CONFIGS,
        },
        'summary': summary,
        'detailed_results': results,
    }

    results_file = output_dir / 'res_293_results.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Determine best configuration
    best_config = max(summary.keys(), key=lambda k: summary[k]['speedup_mean'])
    baseline_speedup = summary['fixed_150']['speedup_mean']
    best_speedup = summary[best_config]['speedup_mean']

    if baseline_speedup > 0:
        improvement = (best_speedup - baseline_speedup) / baseline_speedup * 100
    else:
        improvement = 0.0

    print(f"\n{'=' * 70}")
    print(f"CONCLUSION")
    print(f"{'=' * 70}")
    print(f"Best configuration: {best_config}")
    print(f"Baseline (fixed_150) speedup: {baseline_speedup:.2f}×")
    print(f"Best speedup: {best_speedup:.2f}×")
    print(f"Improvement: {improvement:+.1f}%")

    if improvement > 20:
        print("✓ VALIDATED: Adaptive stopping significantly improves speedup")
    elif improvement > 5:
        print("~ PARTIAL: Adaptive stopping provides modest improvement")
    else:
        print("✗ REFUTED: Adaptive stopping does not improve speedup")


if __name__ == '__main__':
    main()
