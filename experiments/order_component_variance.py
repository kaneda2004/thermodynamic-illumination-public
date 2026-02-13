#!/usr/bin/env python3
"""
RES-055: Which component of order_multiplicative() contributes most to variance?

Hypothesis: Coherence gate contributes most to variance due to its steep sigmoid.

Method:
1. Generate diverse CPPN samples spanning order range
2. Compute each component's contribution to final score
3. Calculate variance attribution using variance decomposition
4. Validate with correlation analysis
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, gaussian_gate,
    compute_compressibility, compute_edge_density,
    compute_spectral_coherence, compute_symmetry,
    compute_connected_components
)
from scipy import stats


def decompose_order_components(img: np.ndarray) -> dict:
    """Decompose order_multiplicative into its component gates/bonuses."""
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)
    symmetry = compute_symmetry(img)
    components = compute_connected_components(img)

    # Compute gates
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Compress gate
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Bonuses
    symmetry_bonus = 0.3 * symmetry
    if components == 0:
        component_bonus = 0
    elif components <= 5:
        component_bonus = 0.2 * (components / 5)
    else:
        component_bonus = max(0, 0.2 * (1 - (components - 5) / 20))

    base_score = density_gate * edge_gate * coherence_gate * compress_gate
    final_score = base_score * (1 + symmetry_bonus + component_bonus)

    return {
        'density_gate': density_gate,
        'edge_gate': edge_gate,
        'coherence_gate': coherence_gate,
        'compress_gate': compress_gate,
        'symmetry_bonus': symmetry_bonus,
        'component_bonus': component_bonus,
        'base_score': base_score,
        'final_score': final_score,
        # Raw values for analysis
        'density_raw': density,
        'edge_raw': edge_density,
        'coherence_raw': coherence,
        'compress_raw': compressibility,
        'symmetry_raw': symmetry,
        'components_raw': components
    }


def run_experiment(n_samples=500, image_size=32, seed=42):
    """Run variance decomposition experiment."""
    np.random.seed(seed)

    print(f"Generating {n_samples} CPPN samples at {image_size}x{image_size}...")

    # Collect decomposed components
    all_components = []

    for i in range(n_samples):
        # Random CPPN with random weights
        np.random.seed(i)
        cppn = CPPN()  # Uses default random initialization
        x = np.linspace(-1, 1, image_size)
        y = np.linspace(-1, 1, image_size)
        xx, yy = np.meshgrid(x, y)
        img_raw = cppn.activate(xx, yy)
        # Binarize: threshold at 0.5
        img = (img_raw > 0.5).astype(np.uint8)

        components = decompose_order_components(img)
        all_components.append(components)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    # Convert to arrays
    keys = ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate',
            'symmetry_bonus', 'component_bonus', 'final_score']
    data = {k: np.array([c[k] for c in all_components]) for k in keys}

    # Variance of each gate
    print("\n=== Gate Variances ===")
    gate_vars = {}
    for k in ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate']:
        gate_vars[k] = np.var(data[k])
        print(f"  {k}: var={gate_vars[k]:.6f}, mean={np.mean(data[k]):.4f}, std={np.std(data[k]):.4f}")

    # Variance of bonuses
    print("\n=== Bonus Variances ===")
    for k in ['symmetry_bonus', 'component_bonus']:
        print(f"  {k}: var={np.var(data[k]):.6f}, mean={np.mean(data[k]):.4f}, std={np.std(data[k]):.4f}")

    # Total variance
    total_var = np.var(data['final_score'])
    print(f"\n=== Final Score Variance: {total_var:.6f} ===")

    # Correlation of each gate with final score (variance attribution proxy)
    print("\n=== Correlation with Final Score (R^2 = variance explained) ===")
    correlations = {}
    for k in ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate']:
        r, p = stats.pearsonr(data[k], data['final_score'])
        correlations[k] = {'r': r, 'r2': r**2, 'p': p}
        print(f"  {k}: r={r:.4f}, R^2={r**2:.4f}, p={p:.2e}")

    # Variance decomposition via log-transform (since multiplicative)
    print("\n=== Log-Variance Attribution (multiplicative decomposition) ===")
    # For multiplicative: log(base) = log(d) + log(e) + log(c) + log(comp)
    # Variance of sum = sum of variances + 2*covariances
    log_data = {}
    for k in ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate']:
        vals = data[k]
        # Clip to avoid log(0)
        vals_clipped = np.clip(vals, 1e-10, None)
        log_data[k] = np.log(vals_clipped)

    log_vars = {k: np.var(log_data[k]) for k in log_data}
    total_log_var = sum(log_vars.values())

    print(f"  Total log-variance: {total_log_var:.4f}")
    print("\n  Fractional contribution (log-domain):")
    for k, v in sorted(log_vars.items(), key=lambda x: -x[1]):
        frac = v / total_log_var if total_log_var > 0 else 0
        print(f"    {k}: {frac:.2%} (log-var={v:.4f})")

    # Identify dominant component
    dominant = max(log_vars, key=log_vars.get)
    dominant_frac = log_vars[dominant] / total_log_var if total_log_var > 0 else 0

    # Statistical test: Is dominant significantly larger?
    # Use bootstrap to get CI on fraction
    print("\n=== Bootstrap Analysis ===")
    n_bootstrap = 1000
    bootstrap_fracs = {k: [] for k in log_vars}

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        boot_log_vars = {}
        for k in log_data:
            boot_log_vars[k] = np.var(log_data[k][idx])
        boot_total = sum(boot_log_vars.values())
        for k in boot_log_vars:
            bootstrap_fracs[k].append(boot_log_vars[k] / boot_total if boot_total > 0 else 0)

    print("  Gate contribution fractions (95% CI):")
    results = {}
    for k in sorted(log_vars.keys(), key=lambda x: -log_vars[x]):
        fracs = bootstrap_fracs[k]
        mean_frac = np.mean(fracs)
        ci_low = np.percentile(fracs, 2.5)
        ci_high = np.percentile(fracs, 97.5)
        results[k] = {'mean': mean_frac, 'ci_low': ci_low, 'ci_high': ci_high}
        print(f"    {k}: {mean_frac:.2%} [{ci_low:.2%}, {ci_high:.2%}]")

    # Effect size: How much larger is dominant vs second?
    sorted_keys = sorted(log_vars.keys(), key=lambda x: -log_vars[x])
    first = sorted_keys[0]
    second = sorted_keys[1]
    effect_ratio = results[first]['mean'] / results[second]['mean'] if results[second]['mean'] > 0 else float('inf')

    # Is hypothesis confirmed?
    # Hypothesis: coherence_gate contributes most
    hypothesis_confirmed = (first == 'coherence_gate')

    print(f"\n=== RESULTS ===")
    print(f"  Dominant gate: {first}")
    print(f"  Contribution: {results[first]['mean']:.2%}")
    print(f"  Ratio to second ({second}): {effect_ratio:.2f}x")
    print(f"  Hypothesis (coherence dominant): {'CONFIRMED' if hypothesis_confirmed else 'REFUTED'}")

    # Final summary
    return {
        'dominant_gate': first,
        'dominant_fraction': float(results[first]['mean']),
        'dominant_ci': [float(results[first]['ci_low']), float(results[first]['ci_high'])],
        'second_gate': second,
        'second_fraction': float(results[second]['mean']),
        'effect_ratio': float(effect_ratio),
        'hypothesis_confirmed': bool(hypothesis_confirmed),
        'all_fractions': {k: float(v['mean']) for k, v in results.items()},
        'correlations': {k: float(v['r2']) for k, v in correlations.items()},
        'n_samples': n_samples
    }


if __name__ == '__main__':
    results = run_experiment(n_samples=500, image_size=32)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Dominant: {results['dominant_gate']} ({results['dominant_fraction']:.1%})")
    print(f"Second: {results['second_gate']} ({results['second_fraction']:.1%})")
    print(f"Effect ratio: {results['effect_ratio']:.2f}x")
