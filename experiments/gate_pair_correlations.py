"""
RES-174: Gate Pair Correlation Analysis

HYPOTHESIS: Coherence-compress gate pair correlation exceeds density-edge pair correlation

The order_multiplicative metric has four gates:
  - density_gate: Gaussian bell curve centered at 0.5 (penalizes empty/full)
  - edge_gate: Gaussian bell curve centered at 0.15 (penalizes no edges / too many edges)
  - coherence_gate: Sigmoid at 0.3 (rewards spectral coherence)
  - compress_gate: Piecewise linear (rewards compressibility 0.2-0.8)

Theory: Coherence and compress may be more tightly coupled because both respond
to spatial structure (low-frequency dominance correlates with compressibility).
Density and edge may be more independent (density is just mean, edges depend on
contrast patterns).

METHOD:
1. Generate 500 CPPN images
2. Compute all four gate values for each
3. Compute correlation matrix between gates
4. Test if coherence-compress corr > density-edge corr using Fisher z-test
5. Analyze the full correlation structure
"""

import numpy as np
import json
import os
from scipy import stats
from datetime import datetime
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, gaussian_gate,
    compute_compressibility, compute_edge_density, compute_spectral_coherence
)


def compute_all_gates(img: np.ndarray) -> dict:
    """Compute all four gate values for an image."""
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)

    # Gate 1: Density (bell curve centered at 0.5, sigma=0.25)
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density (bell curve centered at 0.15, sigma=0.08)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)

    # Gate 3: Coherence (spectral, sigmoid centered at 0.3)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    return {
        'density': density,
        'edge_density': edge_density,
        'coherence': coherence,
        'compressibility': compressibility,
        'density_gate': density_gate,
        'edge_gate': edge_gate,
        'coherence_gate': coherence_gate,
        'compress_gate': compress_gate,
        'order': order_multiplicative(img)
    }


def fisher_z_test(r1: float, r2: float, n1: int, n2: int) -> tuple:
    """Fisher z-test for difference between two correlations."""
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def main():
    np.random.seed(42)
    n_samples = 500

    print(f"Generating {n_samples} CPPN images...")

    # Collect gate values
    all_gates = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(64)
        gates = compute_all_gates(img)
        all_gates.append(gates)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    # Convert to arrays
    gate_names = ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate']
    raw_names = ['density', 'edge_density', 'coherence', 'compressibility']

    gate_arrays = {name: np.array([g[name] for g in all_gates]) for name in gate_names}
    raw_arrays = {name: np.array([g[name] for g in all_gates]) for name in raw_names}
    orders = np.array([g['order'] for g in all_gates])

    # Compute correlation matrix between gates
    print("\n=== Gate Correlation Matrix ===")
    n = len(gate_names)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, name_i in enumerate(gate_names):
        for j, name_j in enumerate(gate_names):
            r, p = stats.spearmanr(gate_arrays[name_i], gate_arrays[name_j])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p

    print("\nSpearman correlations between gate values:")
    header = "           " + "  ".join([f"{n[:8]:>10}" for n in gate_names])
    print(header)
    for i, name_i in enumerate(gate_names):
        row = f"{name_i[:10]:10} " + "  ".join([f"{corr_matrix[i,j]:>10.3f}" for j in range(n)])
        print(row)

    # Key hypothesis test: coherence-compress vs density-edge
    r_coherence_compress = corr_matrix[gate_names.index('coherence_gate'), gate_names.index('compress_gate')]
    r_density_edge = corr_matrix[gate_names.index('density_gate'), gate_names.index('edge_gate')]

    print(f"\n=== Primary Hypothesis Test ===")
    print(f"Coherence-Compress correlation: {r_coherence_compress:.4f}")
    print(f"Density-Edge correlation: {r_density_edge:.4f}")

    z_stat, p_value = fisher_z_test(r_coherence_compress, r_density_edge, n_samples, n_samples)
    print(f"Fisher z-test: z={z_stat:.3f}, p={p_value:.6f}")

    # Effect size: difference in correlations
    correlation_diff = r_coherence_compress - r_density_edge

    # Also test with raw (pre-gate) values
    r_coh_comp_raw, _ = stats.spearmanr(raw_arrays['coherence'], raw_arrays['compressibility'])
    r_den_edge_raw, _ = stats.spearmanr(raw_arrays['density'], raw_arrays['edge_density'])

    print(f"\n=== Raw (Pre-Gate) Correlations ===")
    print(f"Coherence-Compressibility: {r_coh_comp_raw:.4f}")
    print(f"Density-EdgeDensity: {r_den_edge_raw:.4f}")

    z_raw, p_raw = fisher_z_test(r_coh_comp_raw, r_den_edge_raw, n_samples, n_samples)
    print(f"Fisher z-test (raw): z={z_raw:.3f}, p={p_raw:.6f}")

    # Full correlation analysis
    print("\n=== All Pairwise Gate Correlations ===")
    pair_corrs = []
    for i in range(n):
        for j in range(i+1, n):
            name = f"{gate_names[i][:4]}-{gate_names[j][:4]}"
            r = corr_matrix[i, j]
            pair_corrs.append((name, r))
            print(f"  {gate_names[i]} - {gate_names[j]}: r={r:.4f}")

    # Gate contributions to order
    print("\n=== Gate Contributions to Order ===")
    gate_order_corrs = {}
    for name in gate_names:
        r, p = stats.spearmanr(gate_arrays[name], orders)
        gate_order_corrs[name] = {'r': r, 'p': p}
        print(f"  {name:15s} vs order: r={r:.4f}, p={p:.2e}")

    # Compute effect size as Cohen's d for the correlation difference
    # Using Fisher z transform
    z_coh_comp = np.arctanh(r_coherence_compress)
    z_den_edge = np.arctanh(r_density_edge)
    se_single = 1/np.sqrt(n_samples - 3)
    cohens_d = (z_coh_comp - z_den_edge) / se_single

    print(f"\n=== Results Summary ===")
    print(f"Effect size (Cohen's d from Fisher z): {cohens_d:.3f}")
    print(f"Correlation difference: {correlation_diff:.4f}")

    # Determine status
    validated = (
        p_value < 0.01 and
        abs(cohens_d) > 0.5 and
        r_coherence_compress > r_density_edge  # Hypothesis direction
    )

    if validated:
        status = "validated"
    elif p_value >= 0.01 or abs(cohens_d) <= 0.5:
        status = "refuted" if r_coherence_compress <= r_density_edge else "inconclusive"
    else:
        status = "inconclusive"

    print(f"\nSTATUS: {status}")
    print(f"  p-value: {p_value:.2e} (threshold: <0.01)")
    print(f"  effect size: {cohens_d:.3f} (threshold: >0.5)")
    print(f"  direction correct: {r_coherence_compress > r_density_edge}")

    # Save results
    results = {
        'experiment_id': 'RES-174',
        'hypothesis': 'Coherence-compress gate pair correlation exceeds density-edge pair correlation',
        'domain': 'gate_interactions',
        'status': status,
        'n_samples': n_samples,
        'primary_result': {
            'coherence_compress_corr': float(r_coherence_compress),
            'density_edge_corr': float(r_density_edge),
            'correlation_diff': float(correlation_diff),
            'fisher_z': float(z_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        },
        'raw_correlations': {
            'coherence_compressibility': float(r_coh_comp_raw),
            'density_edge_density': float(r_den_edge_raw)
        },
        'full_correlation_matrix': {
            'gates': gate_names,
            'matrix': corr_matrix.tolist()
        },
        'gate_order_correlations': {k: {'r': float(v['r']), 'p': float(v['p'])} for k, v in gate_order_corrs.items()},
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    results_dir = '/Users/matt/Development/monochrome_noise_converger/results/gate_pair_correlations'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results.json')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    results = main()
