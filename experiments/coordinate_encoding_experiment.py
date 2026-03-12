#!/usr/bin/env python3
"""
Coordinate Encoding Experiment (RES-023)

HYPOTHESIS: Different coordinate encodings for CPPN inputs produce distinct
order distributions. Specifically:
- Sinusoidal positional encoding (inspired by transformers) will produce
  higher mean order than standard Cartesian coordinates
- The order distribution width (std) varies systematically with encoding type

NULL HYPOTHESIS: All coordinate encodings produce statistically identical
order distributions.

Encoding Types Tested:
1. Cartesian: (x, y, bias) - minimal 3-input
2. Standard: (x, y, r, bias) - current default
3. Polar: (r, theta, bias) - pure polar coordinates
4. Sinusoidal: (sin(x*pi), cos(x*pi), sin(y*pi), cos(y*pi), r, bias) - 6 inputs

Method:
- Generate N=300 samples per encoding type with random CPPN weights
- Compute order_multiplicative for each sample
- Primary test: Kruskal-Wallis H-test across all encodings
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
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    ACTIVATIONS, PRIOR_SIGMA, order_multiplicative,
    compute_edge_density, compute_symmetry, compute_spectral_coherence, compute_compressibility
)


# ============================================================================
# COORDINATE ENCODING DEFINITIONS
# ============================================================================

@dataclass
class EncodedCPPN:
    """A CPPN with configurable coordinate encoding."""
    encoding_type: str
    input_names: list
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    output_activation: str = 'sigmoid'

    def __post_init__(self):
        """Initialize nodes and connections based on encoding."""
        n_inputs = len(self.input_names)

        # Input nodes (id: 0 to n_inputs-1)
        self.nodes = []
        for i in range(n_inputs):
            self.nodes.append({'id': i, 'activation': 'identity', 'bias': 0.0})

        # Output node
        output_id = n_inputs
        self.nodes.append({
            'id': output_id,
            'activation': self.output_activation,
            'bias': np.random.randn() * PRIOR_SIGMA
        })

        # Connections: all inputs -> output
        self.connections = []
        for i in range(n_inputs):
            self.connections.append({
                'from_id': i,
                'to_id': output_id,
                'weight': np.random.randn() * PRIOR_SIGMA
            })

        self.output_id = output_id
        self.n_inputs = n_inputs

    def activate(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the network given input arrays."""
        values = {}

        # Set input values
        for i, name in enumerate(self.input_names):
            values[i] = input_dict[name]

        # Compute output
        output_node = self.nodes[-1]
        total = np.zeros_like(list(input_dict.values())[0]) + output_node['bias']

        for conn in self.connections:
            total += values[conn['from_id']] * conn['weight']

        # Apply output activation
        return ACTIVATIONS[output_node['activation']](total)

    def render(self, size: int = 32) -> np.ndarray:
        """Render image using this encoding's coordinate system."""
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)

        # Compute all possible coordinate features
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)  # Range: [-pi, pi]
        theta_norm = theta / np.pi  # Normalize to [-1, 1]
        bias = np.ones_like(x)

        # Sinusoidal positional encoding
        sin_x = np.sin(x * np.pi)
        cos_x = np.cos(x * np.pi)
        sin_y = np.sin(y * np.pi)
        cos_y = np.cos(y * np.pi)

        # Build input dictionary based on encoding type
        all_coords = {
            'x': x, 'y': y, 'r': r, 'theta': theta_norm, 'bias': bias,
            'sin_x': sin_x, 'cos_x': cos_x, 'sin_y': sin_y, 'cos_y': cos_y
        }

        input_dict = {name: all_coords[name] for name in self.input_names}

        return (self.activate(input_dict) > 0.5).astype(np.uint8)


# Encoding configurations
ENCODINGS = {
    'cartesian': ['x', 'y', 'bias'],
    'standard': ['x', 'y', 'r', 'bias'],
    'polar': ['r', 'theta', 'bias'],
    'sinusoidal': ['sin_x', 'cos_x', 'sin_y', 'cos_y', 'r', 'bias'],
    'augmented': ['x', 'y', 'r', 'theta', 'bias'],  # Cartesian + full polar
}


def create_encoded_cppn(encoding_type: str, seed: int = None) -> EncodedCPPN:
    """Create a CPPN with specified coordinate encoding."""
    if seed is not None:
        np.random.seed(seed)

    if encoding_type not in ENCODINGS:
        raise ValueError(f"Unknown encoding: {encoding_type}. Choose from {list(ENCODINGS.keys())}")

    return EncodedCPPN(
        encoding_type=encoding_type,
        input_names=ENCODINGS[encoding_type]
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


def run_experiment(n_samples: int = 300, image_size: int = 32, base_seed: int = 42):
    """
    Run the coordinate encoding experiment.

    Args:
        n_samples: Number of samples per encoding type
        image_size: Size of generated images
        base_seed: Base random seed for reproducibility
    """
    print("=" * 70)
    print("COORDINATE ENCODING EXPERIMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Different coordinate encodings produce distinct order distributions.")
    print("            Sinusoidal positional encoding will produce higher mean order.")
    print()
    print("NULL (H0): All coordinate encodings produce identical order distributions.")
    print()
    print(f"Parameters: n_samples={n_samples}, image_size={image_size}")
    print()
    print("Encodings tested:")
    for enc, inputs in ENCODINGS.items():
        print(f"  {enc}: {inputs}")
    print()

    # Storage for results
    results_by_encoding = {enc: [] for enc in ENCODINGS.keys()}

    # Generate samples for each encoding type
    print("Generating samples...")
    encoding_list = list(ENCODINGS.keys())

    for enc_idx, encoding in enumerate(encoding_list):
        print(f"  {encoding} ({len(ENCODINGS[encoding])} inputs)...", end="", flush=True)

        for i in range(n_samples):
            # Use unique seed for each sample
            seed = base_seed * 1000 + enc_idx * n_samples + i
            cppn = create_encoded_cppn(encoding, seed=seed)
            img = cppn.render(image_size)
            features = compute_features(img)
            features['encoding'] = encoding
            results_by_encoding[encoding].append(features)

        print(f" done ({n_samples} samples)")

    print()

    # Compute summary statistics per encoding
    print("-" * 70)
    print("SUMMARY STATISTICS BY ENCODING TYPE")
    print("-" * 70)
    print(f"{'Encoding':<12} {'Order':>10} {'Order_std':>10} {'EdgeDens':>10} {'SpectCoh':>10}")
    print("-" * 70)

    summary_stats = {}
    order_by_encoding = {}

    for encoding in encoding_list:
        features = results_by_encoding[encoding]
        orders = np.array([f['order'] for f in features])
        order_by_encoding[encoding] = orders

        summary_stats[encoding] = {
            'order_mean': float(np.mean(orders)),
            'order_std': float(np.std(orders)),
            'order_median': float(np.median(orders)),
            'order_q25': float(np.percentile(orders, 25)),
            'order_q75': float(np.percentile(orders, 75)),
            'edge_density_mean': float(np.mean([f['edge_density'] for f in features])),
            'symmetry_mean': float(np.mean([f['symmetry'] for f in features])),
            'spectral_coherence_mean': float(np.mean([f['spectral_coherence'] for f in features])),
            'compressibility_mean': float(np.mean([f['compressibility'] for f in features])),
            'n_inputs': len(ENCODINGS[encoding]),
        }
        s = summary_stats[encoding]
        print(f"{encoding:<12} {s['order_mean']:>10.4f} {s['order_std']:>10.4f} "
              f"{s['edge_density_mean']:>10.4f} {s['spectral_coherence_mean']:>10.4f}")

    print()

    # PRIMARY TEST: Kruskal-Wallis across all encodings
    print("=" * 70)
    print("PRIMARY HYPOTHESIS TEST: Kruskal-Wallis H-test (all encodings)")
    print("=" * 70)
    print()

    groups = [order_by_encoding[enc] for enc in encoding_list]
    H_stat, p_value = kruskal(*groups)

    print(f"Kruskal-Wallis H-statistic: {H_stat:.2f}")
    print(f"P-value: {p_value:.2e}")
    print()

    primary_significant = p_value < 0.01
    if primary_significant:
        print("RESULT: Order distributions DIFFER significantly across encodings (p < 0.01)")
    else:
        print("RESULT: No significant difference in order distributions (p >= 0.01)")

    print()

    # SECONDARY TEST: Pairwise Mann-Whitney U with Bonferroni correction
    print("=" * 70)
    print("SECONDARY ANALYSIS: Pairwise Mann-Whitney U (Bonferroni corrected)")
    print("=" * 70)
    print()

    n_comparisons = len(encoding_list) * (len(encoding_list) - 1) // 2
    alpha_corrected = 0.01 / n_comparisons

    print(f"Number of comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")
    print()

    pairwise_results = []

    print(f"{'Comparison':<30} {'U':>10} {'p-value':>12} {'Cohen d':>10} {'Sig':>6}")
    print("-" * 70)

    for i, enc1 in enumerate(encoding_list):
        for j, enc2 in enumerate(encoding_list):
            if i >= j:
                continue

            U_stat, p_val = mannwhitneyu(order_by_encoding[enc1], order_by_encoding[enc2],
                                         alternative='two-sided')
            d = cohens_d(order_by_encoding[enc1], order_by_encoding[enc2])

            significant = p_val < alpha_corrected and abs(d) > 0.5
            sig_marker = "***" if significant else ""

            pairwise_results.append({
                'enc1': enc1,
                'enc2': enc2,
                'U': float(U_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': bool(significant)
            })

            comparison = f"{enc1} vs {enc2}"
            print(f"{comparison:<30} {U_stat:>10.1f} {p_val:>12.2e} {d:>10.3f} {sig_marker:>6}")

    print()

    # Count significant pairs
    n_significant = sum(1 for r in pairwise_results if r['significant'])
    print(f"Significant pairs (p < {alpha_corrected:.4f}, |d| > 0.5): {n_significant}/{n_comparisons}")
    print()

    # Find best and worst encodings
    print("=" * 70)
    print("ENCODINGS RANKED BY MEAN ORDER")
    print("=" * 70)
    sorted_encodings = sorted(encoding_list, key=lambda e: summary_stats[e]['order_mean'], reverse=True)

    for i, enc in enumerate(sorted_encodings):
        s = summary_stats[enc]
        print(f"{i+1}. {enc:<12}: mean={s['order_mean']:.4f}, std={s['order_std']:.4f}, "
              f"median={s['order_median']:.4f}, inputs={s['n_inputs']}")

    print()

    # Test specific hypothesis: sinusoidal vs standard
    print("=" * 70)
    print("SPECIFIC HYPOTHESIS: Sinusoidal vs Standard encoding")
    print("=" * 70)

    U_sin_std, p_sin_std = mannwhitneyu(order_by_encoding['sinusoidal'],
                                         order_by_encoding['standard'],
                                         alternative='greater')  # One-sided: sinusoidal > standard
    d_sin_std = cohens_d(order_by_encoding['sinusoidal'], order_by_encoding['standard'])

    print(f"Sinusoidal mean order: {summary_stats['sinusoidal']['order_mean']:.4f}")
    print(f"Standard mean order:   {summary_stats['standard']['order_mean']:.4f}")
    print(f"Mann-Whitney U (one-sided, sinusoidal > standard): {U_sin_std:.1f}")
    print(f"P-value: {p_sin_std:.2e}")
    print(f"Cohen's d: {d_sin_std:.3f}")
    print()

    sinusoidal_better = p_sin_std < 0.01 and d_sin_std > 0.5
    if sinusoidal_better:
        print("RESULT: Sinusoidal encoding produces HIGHER order than standard (VALIDATED)")
    elif d_sin_std > 0 and p_sin_std < 0.05:
        print("RESULT: Weak evidence for sinusoidal > standard (marginal)")
    else:
        print("RESULT: No evidence that sinusoidal encoding is better than standard")

    print()

    # Determine overall status
    # Validated if: overall Kruskal-Wallis significant AND at least one pairwise comparison significant
    overall_validated = primary_significant and n_significant > 0

    # Prepare final results
    final_results = {
        'experiment': 'coordinate_encoding',
        'hypothesis': 'Different coordinate encodings produce distinct order distributions',
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'base_seed': base_seed,
            'encodings': ENCODINGS,
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
        'sinusoidal_vs_standard': {
            'U_statistic': float(U_sin_std),
            'p_value_one_sided': float(p_sin_std),
            'cohens_d': float(d_sin_std),
            'sinusoidal_better': bool(sinusoidal_better),
        },
        'summary_by_encoding': summary_stats,
        'rankings': {
            'by_mean_order': sorted_encodings,
        },
        'status': 'validated' if overall_validated else 'refuted' if primary_significant else 'inconclusive',
        'confidence': 'high' if overall_validated or (not primary_significant and p_value > 0.1) else 'medium',
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'coordinate_encoding'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'coordinate_encoding_results.json'

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")
    print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"Primary hypothesis (encodings produce different order distributions):")
    print(f"  Kruskal-Wallis H: {H_stat:.2f}")
    print(f"  P-value: {p_value:.2e} (threshold: 0.01)")
    print(f"  Status: {'SIGNIFICANT' if primary_significant else 'NOT SIGNIFICANT'}")
    print()
    print(f"Pairwise comparisons:")
    print(f"  Significant pairs: {n_significant}/{n_comparisons}")
    print()
    print(f"Best encoding by mean order: {sorted_encodings[0]} ({summary_stats[sorted_encodings[0]]['order_mean']:.4f})")
    print(f"Worst encoding by mean order: {sorted_encodings[-1]} ({summary_stats[sorted_encodings[-1]]['order_mean']:.4f})")
    print()
    print(f"Overall status: {final_results['status'].upper()}")

    return final_results


if __name__ == "__main__":
    results = run_experiment(n_samples=300, image_size=32, base_seed=42)
