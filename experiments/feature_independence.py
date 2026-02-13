#!/usr/bin/env python3
"""Test statistical independence of order components (density, edge_density, coherence)."""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_compressibility, compute_edge_density, compute_spectral_coherence

def mutual_information_discrete(x, y, bins=20):
    """Estimate MI via histogram discretization."""
    c_xy = np.histogram2d(x, y, bins)[0]
    c_x = np.histogram(x, bins)[0]
    c_y = np.histogram(y, bins)[0]

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_xy = c_xy / c_xy.sum() + eps
    p_x = c_x / c_x.sum() + eps
    p_y = c_y / c_y.sum() + eps

    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i,j] > eps:
                mi += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
    return max(0, mi)

def main():
    np.random.seed(42)

    # Generate CPPN samples
    n_samples = 500
    densities = []
    edge_densities = []
    coherences = []

    size = 64

    print(f"Generating {n_samples} CPPN samples...")
    for i in range(n_samples):
        cppn = CPPN()  # Fresh random CPPN each time
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        output = cppn.activate(xx, yy)
        img = (output > 0.5).astype(np.uint8)

        densities.append(compute_compressibility(img))
        edge_densities.append(compute_edge_density(img))
        coherences.append(compute_spectral_coherence(img))
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_samples}")

    densities = np.array(densities)
    edge_densities = np.array(edge_densities)
    coherences = np.array(coherences)

    # Compute mutual information between pairs
    mi_de = mutual_information_discrete(densities, edge_densities)
    mi_dc = mutual_information_discrete(densities, coherences)
    mi_ec = mutual_information_discrete(edge_densities, coherences)

    # Compute Spearman correlations
    r_de, p_de = stats.spearmanr(densities, edge_densities)
    r_dc, p_dc = stats.spearmanr(densities, coherences)
    r_ec, p_ec = stats.spearmanr(edge_densities, coherences)

    # Test via correlation ratio (eta-squared)
    def correlation_ratio(x, y, bins=10):
        """How much variance in y is explained by x bins."""
        bin_idx = np.digitize(x, np.linspace(x.min(), x.max(), bins))
        ss_between = sum(len(y[bin_idx==i]) * (y[bin_idx==i].mean() - y.mean())**2
                         for i in range(1, bins+1) if len(y[bin_idx==i]) > 0)
        ss_total = ((y - y.mean())**2).sum()
        return ss_between / ss_total if ss_total > 0 else 0

    eta_de = correlation_ratio(densities, edge_densities)
    eta_dc = correlation_ratio(densities, coherences)
    eta_ec = correlation_ratio(edge_densities, coherences)

    print("\n=== FEATURE INDEPENDENCE RESULTS ===")
    print(f"\nMutual Information (bits):")
    print(f"  density-edge:     {mi_de:.3f}")
    print(f"  density-coherence: {mi_dc:.3f}")
    print(f"  edge-coherence:   {mi_ec:.3f}")

    print(f"\nSpearman correlations:")
    print(f"  density-edge:     r={r_de:.3f}, p={p_de:.2e}")
    print(f"  density-coherence: r={r_dc:.3f}, p={p_dc:.2e}")
    print(f"  edge-coherence:   r={r_ec:.3f}, p={p_ec:.2e}")

    print(f"\nCorrelation ratios (eta-squared):")
    print(f"  density-edge:     {eta_de:.3f}")
    print(f"  density-coherence: {eta_dc:.3f}")
    print(f"  edge-coherence:   {eta_ec:.3f}")

    # Independence criterion: MI < 0.1 bits for all pairs
    mi_max = max(mi_de, mi_dc, mi_ec)
    independent = mi_max < 0.1

    # Also check correlations
    r_max = max(abs(r_de), abs(r_dc), abs(r_ec))
    low_corr = r_max < 0.3

    print(f"\n=== VERDICT ===")
    print(f"Max MI: {mi_max:.3f} bits (threshold: 0.1)")
    print(f"Max |r|: {r_max:.3f} (threshold: 0.3)")
    print(f"Independent (MI<0.1): {independent}")
    print(f"Low correlation (|r|<0.3): {low_corr}")

    if independent and low_corr:
        print("STATUS: VALIDATED - Components are statistically independent")
    elif not independent:
        print("STATUS: REFUTED - Significant mutual information between components")
    else:
        print("STATUS: REFUTED - Significant correlations between components")

if __name__ == "__main__":
    main()
