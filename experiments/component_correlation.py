#!/usr/bin/env python3
"""
RES-077: Order component correlation analysis.

Hypothesis: Order components exhibit hierarchical correlation structure
with density-compress as primary axis and edge-coherence as secondary axis.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    compute_compressibility,
    compute_edge_density,
    compute_spectral_coherence,
    CPPN
)


def compute_all_components(img: np.ndarray) -> dict:
    """Extract all 4 order components from an image."""
    return {
        'density': np.mean(img),
        'edge': compute_edge_density(img),
        'coherence': compute_spectral_coherence(img),
        'compress': compute_compressibility(img)
    }


def generate_samples(n_samples: int = 500, img_size: int = 64) -> list:
    """Generate diverse samples from CPPN prior."""
    samples = []
    for _ in range(n_samples):
        cppn = CPPN()  # Fresh random CPPN each time
        img = cppn.render(img_size)
        samples.append(compute_all_components(img))
    return samples


def analyze_correlations(samples: list) -> dict:
    """Compute correlation matrix and identify structure."""
    components = ['density', 'edge', 'coherence', 'compress']
    n = len(samples)

    # Extract arrays
    data = {c: np.array([s[c] for s in samples]) for c in components}

    # Correlation matrix
    corr_matrix = {}
    pvalue_matrix = {}

    pairs = []
    for i, c1 in enumerate(components):
        for c2 in components[i+1:]:
            r, p = stats.pearsonr(data[c1], data[c2])
            corr_matrix[f"{c1}-{c2}"] = r
            pvalue_matrix[f"{c1}-{c2}"] = p
            pairs.append((c1, c2, r, p))

    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        'correlations': corr_matrix,
        'pvalues': pvalue_matrix,
        'ranked_pairs': pairs,
        'n_samples': n
    }


def main():
    print("RES-077: Order Component Correlation Analysis")
    print("=" * 50)

    # Generate samples
    print("\nGenerating 500 CPPN samples...")
    samples = generate_samples(n_samples=500, img_size=64)

    # Analyze correlations
    print("Computing correlations...")
    results = analyze_correlations(samples)

    # Report
    print("\n" + "=" * 50)
    print("CORRELATION RANKINGS (by |r|):")
    print("-" * 50)

    for c1, c2, r, p in results['ranked_pairs']:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {c1:10} - {c2:10}: r={r:+.3f}  p={p:.2e} {sig}")

    # Test hypothesis
    print("\n" + "=" * 50)
    print("HYPOTHESIS TEST:")

    density_compress = abs(results['correlations'].get('density-compress', 0))
    edge_coherence = abs(results['correlations'].get('edge-coherence', 0))

    # Original hypothesis: density-compress > 0.8, edge-coherence > 0.6
    h1_pass = density_compress > 0.8
    h2_pass = edge_coherence > 0.6

    print(f"  |r(density,compress)| = {density_compress:.3f} {'> 0.8 PASS' if h1_pass else '< 0.8 FAIL'}")
    print(f"  |r(edge,coherence)|   = {edge_coherence:.3f} {'> 0.6 PASS' if h2_pass else '< 0.6 FAIL'}")

    # Alternative: find actual structure
    print("\n" + "=" * 50)
    print("ACTUAL STRUCTURE:")

    strongest = results['ranked_pairs'][0]
    second = results['ranked_pairs'][1]

    print(f"  Primary axis:   {strongest[0]}-{strongest[1]} (r={strongest[2]:.3f})")
    print(f"  Secondary axis: {second[0]}-{second[1]} (r={second[2]:.3f})")

    # Determine verdict
    all_sig = all(p < 0.01 for _, _, _, p in results['ranked_pairs'])
    max_corr = max(abs(r) for _, _, r, _ in results['ranked_pairs'])

    if h1_pass and h2_pass:
        status = "VALIDATED"
        summary = f"Hierarchical structure confirmed: density-compress r={density_compress:.2f}, edge-coherence r={edge_coherence:.2f}"
    elif max_corr > 0.5:
        status = "PARTIAL"
        summary = f"Correlation structure exists but differs from hypothesis. Strongest: {strongest[0]}-{strongest[1]} r={strongest[2]:.2f}"
    else:
        status = "REFUTED"
        summary = f"Components are largely independent. Max |r|={max_corr:.2f}"

    print(f"\nVERDICT: {status}")
    print(f"SUMMARY: {summary}")

    # Return for log manager
    return {
        'status': status,
        'summary': summary,
        'metrics': {
            'n_samples': results['n_samples'],
            'density_compress_r': float(results['correlations'].get('density-compress', 0)),
            'edge_coherence_r': float(results['correlations'].get('edge-coherence', 0)),
            'max_abs_r': float(max_corr),
            'all_pairs_sig': bool(all_sig)
        }
    }


if __name__ == "__main__":
    result = main()
    print(f"\n{result}")
