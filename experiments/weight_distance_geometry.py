#!/usr/bin/env python3
"""
RES-167: Weight Distance Geometry
Test whether high-order CPPN weight vectors cluster more tightly (smaller pairwise
Euclidean distances) than low-order CPPNs.

Hypothesis: High-order CPPN weight vectors have smaller pairwise Euclidean distances
than low-order CPPNs.

This is distinct from RES-144 which tested angular clustering (cosine similarity).
Here we test the actual Euclidean distance in weight space.
"""

import numpy as np
import json
import os
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative


def generate_samples(n_samples=2000, seed=42):
    """Generate CPPN samples and compute order scores."""
    np.random.seed(seed)

    samples = []
    for i in range(n_samples):
        cppn = CPPN()  # Fresh CPPN with random weights from prior
        img = cppn.render(32)
        order = order_multiplicative(img)
        weights = cppn.get_weights()
        samples.append({
            'weights': weights,
            'order': order
        })

    orders = [s['order'] for s in samples]
    print(f"Order distribution: min={min(orders):.4f}, max={max(orders):.4f}, median={np.median(orders):.4f}")
    return samples


def analyze_weight_distances(samples, high_threshold=0.3, low_threshold=0.05):
    """
    Analyze pairwise Euclidean distances within high-order vs low-order groups.

    Returns dict with metrics and effect sizes.
    """
    # Separate high and low order samples
    high_order = [s for s in samples if s['order'] >= high_threshold]
    low_order = [s for s in samples if s['order'] <= low_threshold]

    print(f"High-order samples (order >= {high_threshold}): {len(high_order)}")
    print(f"Low-order samples (order <= {low_threshold}): {len(low_order)}")

    if len(high_order) < 10 or len(low_order) < 10:
        return None

    # Get weight matrices
    high_weights = np.array([s['weights'] for s in high_order])
    low_weights = np.array([s['weights'] for s in low_order])

    # Compute pairwise Euclidean distances
    high_distances = pdist(high_weights, metric='euclidean')
    low_distances = pdist(low_weights, metric='euclidean')

    # Statistics
    high_mean = np.mean(high_distances)
    low_mean = np.mean(low_distances)
    high_std = np.std(high_distances)
    low_std = np.std(low_distances)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_std**2 + low_std**2) / 2)
    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

    # Statistical test (Mann-Whitney U for non-normal distance distributions)
    stat, p_value = stats.mannwhitneyu(high_distances, low_distances, alternative='two-sided')

    # Also compute L2 norms to compare with RES-037
    high_norms = np.linalg.norm(high_weights, axis=1)
    low_norms = np.linalg.norm(low_weights, axis=1)

    norm_d = (np.mean(high_norms) - np.mean(low_norms)) / np.sqrt((np.std(high_norms)**2 + np.std(low_norms)**2) / 2)

    return {
        'n_high': len(high_order),
        'n_low': len(low_order),
        'high_mean_dist': high_mean,
        'low_mean_dist': low_mean,
        'high_std_dist': high_std,
        'low_std_dist': low_std,
        'cohens_d': cohens_d,
        'p_value': p_value,
        'high_mean_norm': float(np.mean(high_norms)),
        'low_mean_norm': float(np.mean(low_norms)),
        'norm_cohens_d': norm_d,
        # Additional: correlation between order and distance to origin
        'high_distances': high_distances.tolist()[:100],  # Sample for storage
        'low_distances': low_distances.tolist()[:100],
    }


def correlation_analysis(samples):
    """
    Compute correlation between order and weight properties.
    """
    orders = np.array([s['order'] for s in samples])
    weights = np.array([s['weights'] for s in samples])

    # L2 norm
    norms = np.linalg.norm(weights, axis=1)

    # Compute distance to mean weight vector
    mean_weights = np.mean(weights, axis=0)
    dist_to_mean = np.linalg.norm(weights - mean_weights, axis=1)

    # Correlations
    r_norm, p_norm = stats.pearsonr(orders, norms)
    r_dist, p_dist = stats.pearsonr(orders, dist_to_mean)

    return {
        'norm_correlation': r_norm,
        'norm_p': p_norm,
        'dist_to_mean_correlation': r_dist,
        'dist_to_mean_p': p_dist
    }


def main():
    print("RES-167: Weight Distance Geometry")
    print("=" * 60)

    # Generate samples - larger N for robust statistics
    print("\nGenerating CPPN samples...")
    samples = generate_samples(n_samples=2000, seed=42)

    # Analyze distances - use percentiles for thresholds
    orders = [s['order'] for s in samples]
    high_threshold = np.percentile(orders, 90)  # Top 10%
    low_threshold = np.percentile(orders, 30)   # Bottom 30%
    print(f"\nUsing percentile thresholds: high>={high_threshold:.4f}, low<={low_threshold:.4f}")

    print("\nAnalyzing pairwise weight distances...")
    results = analyze_weight_distances(samples, high_threshold=high_threshold, low_threshold=low_threshold)

    if results is None:
        print("ERROR: Not enough samples in high/low groups")
        return

    print(f"\nHigh-order group: {results['n_high']} samples")
    print(f"Low-order group: {results['n_low']} samples")

    print(f"\nPairwise Euclidean distances:")
    print(f"  High-order: {results['high_mean_dist']:.4f} +/- {results['high_std_dist']:.4f}")
    print(f"  Low-order: {results['low_mean_dist']:.4f} +/- {results['low_std_dist']:.4f}")
    print(f"  Cohen's d: {results['cohens_d']:.4f}")
    print(f"  p-value: {results['p_value']:.2e}")

    print(f"\nL2 norms (for comparison with RES-037):")
    print(f"  High-order: {results['high_mean_norm']:.4f}")
    print(f"  Low-order: {results['low_mean_norm']:.4f}")
    print(f"  Cohen's d: {results['norm_cohens_d']:.4f}")

    # Correlation analysis
    print("\nCorrelation analysis...")
    corr = correlation_analysis(samples)
    print(f"  Order vs L2 norm: r={corr['norm_correlation']:.3f}, p={corr['norm_p']:.2e}")
    print(f"  Order vs dist-to-mean: r={corr['dist_to_mean_correlation']:.3f}, p={corr['dist_to_mean_p']:.2e}")

    # Determine outcome
    if results['p_value'] < 0.01 and abs(results['cohens_d']) > 0.5:
        if results['cohens_d'] < 0:
            status = 'validated'
            summary = f"High-order CPPNs cluster more tightly (d={results['cohens_d']:.2f})"
        else:
            status = 'refuted'
            summary = f"High-order CPPNs spread MORE widely (d={results['cohens_d']:.2f}), opposite hypothesis"
    else:
        status = 'inconclusive' if results['p_value'] < 0.05 else 'refuted'
        summary = f"No significant difference in pairwise distances (d={results['cohens_d']:.2f}, p={results['p_value']:.2e})"

    print(f"\nConclusion: {status.upper()}")
    print(f"Summary: {summary}")

    # Save results
    output_dir = '/Users/matt/Development/monochrome_noise_converger/results/weight_distance_geometry'
    os.makedirs(output_dir, exist_ok=True)

    # Remove large arrays for JSON
    results_clean = {k: v for k, v in results.items() if not k.endswith('distances')}
    results_clean.update(corr)
    results_clean['status'] = status
    results_clean['summary'] = summary

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return status, summary, results_clean


if __name__ == '__main__':
    main()
