#!/usr/bin/env python3
"""
Coordinate Warping Experiment (RES-102)

HYPOTHESIS: Nonlinear coordinate warping (log, sqrt, square transformations
applied to x,y coordinates before CPPN processing) produces systematically
different order distributions than linear coordinates.

Some transformations may enable higher order by:
- Creating more natural frequency distributions (log warping)
- Expanding/compressing coordinate space non-uniformly
- Introducing implicit bias toward certain pattern scales

NULL HYPOTHESIS: All coordinate warping transformations produce statistically
identical order distributions.

Warping Types Tested:
1. identity: (x, y) - baseline linear coordinates
2. sqrt_warp: (sign(x)*sqrt(|x|), sign(y)*sqrt(|y|)) - expands center
3. square_warp: (sign(x)*x^2, sign(y)*y^2) - compresses center
4. log_warp: (sign(x)*log(1+|x|), sign(y)*log(1+|y|)) - subtle expansion
5. exp_warp: (sign(x)*(exp(|x|)-1), sign(y)*(exp(|y|)-1)) - exponential growth
6. sinh_warp: (sinh(x), sinh(y)) - smooth nonlinear (hyperbolic)

Method:
- Generate N=400 samples per warping type with random CPPN weights
- Apply warping to coordinates BEFORE feeding to standard CPPN
- Compute order_multiplicative for each sample
- Primary test: Kruskal-Wallis H-test across all warpings
- Secondary test: Pairwise Mann-Whitney U with Bonferroni correction
- Effect size: Cohen's d for significant pairs
"""

import sys
import os
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    ACTIVATIONS, PRIOR_SIGMA, order_multiplicative,
    compute_edge_density, compute_symmetry, compute_spectral_coherence, compute_compressibility
)


# ============================================================================
# COORDINATE WARPING FUNCTIONS
# ============================================================================

def identity_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """No transformation - baseline."""
    return x, y

def sqrt_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Square root warping - expands center, compresses edges."""
    x_warp = np.sign(x) * np.sqrt(np.abs(x))
    y_warp = np.sign(y) * np.sqrt(np.abs(y))
    return x_warp, y_warp

def square_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Square warping - compresses center, expands edges."""
    x_warp = np.sign(x) * (x ** 2)
    y_warp = np.sign(y) * (y ** 2)
    return x_warp, y_warp

def log_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Logarithmic warping - gentle expansion of center."""
    x_warp = np.sign(x) * np.log1p(np.abs(x))
    y_warp = np.sign(y) * np.log1p(np.abs(y))
    return x_warp, y_warp

def exp_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exponential warping - strong compression of center."""
    # Scale to prevent explosion: expm1(1) ~ 1.72
    x_warp = np.sign(x) * np.expm1(np.abs(x))
    y_warp = np.sign(y) * np.expm1(np.abs(y))
    return x_warp, y_warp

def sinh_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hyperbolic sine warping - smooth S-curve nonlinearity."""
    x_warp = np.sinh(x)
    y_warp = np.sinh(y)
    return x_warp, y_warp

def tanh_warp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hyperbolic tangent warping - compresses extremes."""
    # Apply stronger scaling to make effect more visible
    x_warp = np.tanh(x * 2)
    y_warp = np.tanh(y * 2)
    return x_warp, y_warp


# All warping functions
WARPINGS = {
    'identity': identity_warp,
    'sqrt': sqrt_warp,
    'square': square_warp,
    'log': log_warp,
    'exp': exp_warp,
    'sinh': sinh_warp,
    'tanh': tanh_warp,
}


# ============================================================================
# WARPED CPPN
# ============================================================================

@dataclass
class WarpedCPPN:
    """A CPPN with configurable coordinate warping applied before processing."""
    warp_type: str
    warp_fn: Callable
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4

    def __post_init__(self):
        """Initialize nodes and connections."""
        if not self.nodes:
            # Standard CPPN structure: x, y, r, bias -> output
            self.nodes = [
                {'id': 0, 'activation': 'identity', 'bias': 0.0},  # x
                {'id': 1, 'activation': 'identity', 'bias': 0.0},  # y
                {'id': 2, 'activation': 'identity', 'bias': 0.0},  # r
                {'id': 3, 'activation': 'identity', 'bias': 0.0},  # bias
                {'id': 4, 'activation': 'sigmoid', 'bias': np.random.randn() * PRIOR_SIGMA},
            ]
            self.connections = [
                {'from_id': i, 'to_id': 4, 'weight': np.random.randn() * PRIOR_SIGMA}
                for i in range(4)
            ]

    def render(self, size: int = 32) -> np.ndarray:
        """Render image with coordinate warping applied."""
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)

        # Apply coordinate warping
        x_warp, y_warp = self.warp_fn(x, y)

        # Compute r from warped coordinates
        r = np.sqrt(x_warp**2 + y_warp**2)
        bias = np.ones_like(x)

        # Input values
        values = {0: x_warp, 1: y_warp, 2: r, 3: bias}

        # Compute output
        output_node = self.nodes[-1]
        total = np.zeros_like(x) + output_node['bias']

        for conn in self.connections:
            total += values[conn['from_id']] * conn['weight']

        # Apply sigmoid
        output = 1 / (1 + np.exp(-np.clip(total, -10, 10)))

        return (output > 0.5).astype(np.uint8)


def create_warped_cppn(warp_type: str, seed: int = None) -> WarpedCPPN:
    """Create a CPPN with specified coordinate warping."""
    if seed is not None:
        np.random.seed(seed)

    if warp_type not in WARPINGS:
        raise ValueError(f"Unknown warping: {warp_type}. Choose from {list(WARPINGS.keys())}")

    return WarpedCPPN(
        warp_type=warp_type,
        warp_fn=WARPINGS[warp_type]
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_features(img: np.ndarray) -> dict:
    """Compute all feature metrics for an image."""
    return {
        'order': order_multiplicative(img),
        'edge_density': compute_edge_density(img),
        'symmetry': compute_symmetry(img),
        'spectral_coherence': compute_spectral_coherence(img),
        'compressibility': compute_compressibility(img),
    }


def run_experiment(n_samples: int = 400, image_size: int = 32, base_seed: int = 42):
    """
    Run the coordinate warping experiment.
    """
    print("=" * 70)
    print("COORDINATE WARPING EXPERIMENT (RES-102)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Nonlinear coordinate warping produces distinct order distributions.")
    print("            Some transformations may enable higher order than linear baseline.")
    print()
    print("NULL (H0): All coordinate warpings produce identical order distributions.")
    print()
    print(f"Parameters: n_samples={n_samples}, image_size={image_size}")
    print()
    print("Warpings tested:")
    for name in WARPINGS.keys():
        print(f"  {name}")
    print()

    # Storage for results
    results_by_warp = {warp: [] for warp in WARPINGS.keys()}

    # Generate samples for each warping type
    print("Generating samples...")
    warp_list = list(WARPINGS.keys())

    for warp_idx, warp_type in enumerate(warp_list):
        print(f"  {warp_type}...", end="", flush=True)

        for i in range(n_samples):
            # Use unique seed for each sample
            seed = base_seed * 1000 + warp_idx * n_samples + i
            cppn = create_warped_cppn(warp_type, seed=seed)
            img = cppn.render(image_size)
            features = compute_features(img)
            features['warp_type'] = warp_type
            results_by_warp[warp_type].append(features)

        print(f" done ({n_samples} samples)")

    print()

    # Compute summary statistics per warping
    print("-" * 70)
    print("SUMMARY STATISTICS BY WARPING TYPE")
    print("-" * 70)
    print(f"{'Warping':<12} {'Order':>10} {'Order_std':>10} {'EdgeDens':>10} {'SpectCoh':>10}")
    print("-" * 70)

    summary_stats = {}
    order_by_warp = {}

    for warp_type in warp_list:
        features = results_by_warp[warp_type]
        orders = np.array([f['order'] for f in features])
        order_by_warp[warp_type] = orders

        summary_stats[warp_type] = {
            'order_mean': float(np.mean(orders)),
            'order_std': float(np.std(orders)),
            'order_median': float(np.median(orders)),
            'order_q25': float(np.percentile(orders, 25)),
            'order_q75': float(np.percentile(orders, 75)),
            'edge_density_mean': float(np.mean([f['edge_density'] for f in features])),
            'symmetry_mean': float(np.mean([f['symmetry'] for f in features])),
            'spectral_coherence_mean': float(np.mean([f['spectral_coherence'] for f in features])),
            'compressibility_mean': float(np.mean([f['compressibility'] for f in features])),
        }
        s = summary_stats[warp_type]
        print(f"{warp_type:<12} {s['order_mean']:>10.4f} {s['order_std']:>10.4f} "
              f"{s['edge_density_mean']:>10.4f} {s['spectral_coherence_mean']:>10.4f}")

    print()

    # PRIMARY TEST: Kruskal-Wallis across all warpings
    print("=" * 70)
    print("PRIMARY HYPOTHESIS TEST: Kruskal-Wallis H-test (all warpings)")
    print("=" * 70)
    print()

    groups = [order_by_warp[w] for w in warp_list]
    H_stat, p_value = kruskal(*groups)

    print(f"Kruskal-Wallis H-statistic: {H_stat:.2f}")
    print(f"P-value: {p_value:.2e}")
    print()

    primary_significant = p_value < 0.01
    if primary_significant:
        print("RESULT: Order distributions DIFFER significantly across warpings (p < 0.01)")
    else:
        print("RESULT: No significant difference in order distributions (p >= 0.01)")

    print()

    # SECONDARY TEST: Pairwise Mann-Whitney U with Bonferroni correction
    print("=" * 70)
    print("SECONDARY ANALYSIS: Pairwise Mann-Whitney U (Bonferroni corrected)")
    print("=" * 70)
    print()

    n_comparisons = len(warp_list) * (len(warp_list) - 1) // 2
    alpha_corrected = 0.01 / n_comparisons

    print(f"Number of comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")
    print()

    pairwise_results = []

    print(f"{'Comparison':<25} {'U':>10} {'p-value':>12} {'Cohen d':>10} {'Sig':>6}")
    print("-" * 70)

    for i, w1 in enumerate(warp_list):
        for j, w2 in enumerate(warp_list):
            if i >= j:
                continue

            U_stat, p_val = mannwhitneyu(order_by_warp[w1], order_by_warp[w2],
                                         alternative='two-sided')
            d = cohens_d(order_by_warp[w1], order_by_warp[w2])

            significant = p_val < alpha_corrected and abs(d) > 0.5
            sig_marker = "***" if significant else ""

            pairwise_results.append({
                'warp1': w1,
                'warp2': w2,
                'U': float(U_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': bool(significant)
            })

            comparison = f"{w1} vs {w2}"
            print(f"{comparison:<25} {U_stat:>10.1f} {p_val:>12.2e} {d:>10.3f} {sig_marker:>6}")

    print()

    # Count significant pairs
    n_significant = sum(1 for r in pairwise_results if r['significant'])
    print(f"Significant pairs (p < {alpha_corrected:.4f}, |d| > 0.5): {n_significant}/{n_comparisons}")
    print()

    # Find best and worst warpings
    print("=" * 70)
    print("WARPINGS RANKED BY MEAN ORDER")
    print("=" * 70)
    sorted_warpings = sorted(warp_list, key=lambda w: summary_stats[w]['order_mean'], reverse=True)

    for i, warp in enumerate(sorted_warpings):
        s = summary_stats[warp]
        print(f"{i+1}. {warp:<12}: mean={s['order_mean']:.4f}, std={s['order_std']:.4f}, "
              f"median={s['order_median']:.4f}")

    print()

    # Test specific hypothesis: best vs identity baseline
    print("=" * 70)
    print("SPECIFIC HYPOTHESIS: Best warping vs Identity baseline")
    print("=" * 70)

    best_warp = sorted_warpings[0]
    if best_warp != 'identity':
        U_best_id, p_best_id = mannwhitneyu(order_by_warp[best_warp],
                                             order_by_warp['identity'],
                                             alternative='greater')
        d_best_id = cohens_d(order_by_warp[best_warp], order_by_warp['identity'])

        print(f"Best warping ({best_warp}) mean order: {summary_stats[best_warp]['order_mean']:.4f}")
        print(f"Identity mean order:                   {summary_stats['identity']['order_mean']:.4f}")
        print(f"Mann-Whitney U (one-sided, {best_warp} > identity): {U_best_id:.1f}")
        print(f"P-value: {p_best_id:.2e}")
        print(f"Cohen's d: {d_best_id:.3f}")
        print()

        best_beats_identity = p_best_id < 0.01 and d_best_id > 0.5
        if best_beats_identity:
            print(f"RESULT: {best_warp} warping produces HIGHER order than identity (VALIDATED)")
        elif d_best_id > 0 and p_best_id < 0.05:
            print(f"RESULT: Weak evidence for {best_warp} > identity (marginal)")
        else:
            print(f"RESULT: No evidence that {best_warp} is better than identity")
    else:
        print("Identity is the best warping - no nonlinear transformation improves order.")
        best_beats_identity = False
        p_best_id = 1.0
        d_best_id = 0.0

    print()

    # Compute effect size relative to identity for all warpings
    print("=" * 70)
    print("EFFECT SIZE VS IDENTITY BASELINE")
    print("=" * 70)

    vs_identity = {}
    for warp_type in warp_list:
        if warp_type == 'identity':
            continue
        d = cohens_d(order_by_warp[warp_type], order_by_warp['identity'])
        vs_identity[warp_type] = d
        effect_desc = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        direction = "higher" if d > 0 else "lower"
        print(f"{warp_type:<12}: d={d:>7.3f} ({direction}, {effect_desc})")

    print()

    # Determine overall status
    overall_validated = primary_significant and n_significant > 0

    # Prepare final results
    final_results = {
        'experiment': 'coordinate_warping',
        'experiment_id': 'RES-102',
        'hypothesis': 'Nonlinear coordinate warping produces distinct order distributions',
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'base_seed': base_seed,
            'warpings': list(WARPINGS.keys()),
        },
        'primary_test': {
            'test': 'kruskal_wallis',
            'H_statistic': float(H_stat),
            'p_value': float(p_value),
            'significant': bool(primary_significant),
        },
        'secondary_tests': {
            'pairwise_comparisons': pairwise_results,
            'n_significant': n_significant,
            'n_comparisons': n_comparisons,
            'bonferroni_alpha': float(alpha_corrected),
        },
        'best_vs_identity': {
            'best_warp': best_warp,
            'cohens_d': float(d_best_id) if best_warp != 'identity' else 0.0,
            'p_value': float(p_best_id) if best_warp != 'identity' else 1.0,
            'beats_identity': bool(best_beats_identity) if best_warp != 'identity' else False,
        },
        'effect_vs_identity': vs_identity,
        'summary_by_warp': summary_stats,
        'rankings': {
            'by_mean_order': sorted_warpings,
        },
        'status': 'validated' if overall_validated else 'refuted' if not primary_significant else 'inconclusive',
        'confidence': 'high' if overall_validated or (not primary_significant and p_value > 0.1) else 'medium',
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'coordinate_warping'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'coordinate_warping_results.json'

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")
    print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"Primary hypothesis (warpings produce different order distributions):")
    print(f"  Kruskal-Wallis H: {H_stat:.2f}")
    print(f"  P-value: {p_value:.2e} (threshold: 0.01)")
    print(f"  Status: {'SIGNIFICANT' if primary_significant else 'NOT SIGNIFICANT'}")
    print()
    print(f"Pairwise comparisons:")
    print(f"  Significant pairs: {n_significant}/{n_comparisons}")
    print()
    print(f"Best warping by mean order: {sorted_warpings[0]} ({summary_stats[sorted_warpings[0]]['order_mean']:.4f})")
    print(f"Worst warping by mean order: {sorted_warpings[-1]} ({summary_stats[sorted_warpings[-1]]['order_mean']:.4f})")
    print()
    print(f"Overall status: {final_results['status'].upper()}")

    return final_results


if __name__ == "__main__":
    results = run_experiment(n_samples=400, image_size=32, base_seed=42)
