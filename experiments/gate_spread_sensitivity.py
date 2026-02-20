"""
RES-197: Gate spread (max-min) predicts order sensitivity to perturbation better than absolute order

Hypothesis: The spread between max and min gate values (max_gate - min_gate) predicts
order sensitivity to perturbation better than the absolute order value.

Rationale: With multicollinearity (RES-174), all gates correlate r>0.86 with order.
But the bottleneck structure (RES-149) suggests gate imbalance matters. If one gate
is near zero while others are high (large spread), perturbations affecting the bottleneck
gate should have outsized impact vs balanced gates (small spread).

Method:
1. Generate diverse CPPNs across order spectrum
2. For each, compute individual gates and spread (max - min)
3. Apply random perturbations and measure order change
4. Correlate: (a) order vs sensitivity, (b) spread vs sensitivity
5. Compare correlation strengths

Success: spread-sensitivity correlation r > order-sensitivity correlation r + 0.1
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative,
    compute_compressibility, compute_edge_density,
    compute_spectral_coherence, gaussian_gate
)
from scipy import stats


def compute_gates(img: np.ndarray) -> dict:
    """Compute individual gate values for an image."""
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)

    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    gates = [density_gate, edge_gate, coherence_gate, compress_gate]
    return {
        'density': density_gate,
        'edge': edge_gate,
        'coherence': coherence_gate,
        'compress': compress_gate,
        'min_gate': min(gates),
        'max_gate': max(gates),
        'spread': max(gates) - min(gates),
        'std': np.std(gates),
        'product': np.prod(gates)
    }


def measure_perturbation_sensitivity(cppn: CPPN, n_perturbations: int = 20, sigma: float = 0.1) -> float:
    """Measure order sensitivity by perturbing weights and measuring order change."""
    base_img = cppn.render(32)
    base_order = order_multiplicative(base_img)

    weights = cppn.get_weights()
    order_changes = []

    for _ in range(n_perturbations):
        perturbed = cppn.copy()
        noise = np.random.randn(len(weights)) * sigma
        perturbed.set_weights(weights + noise)

        perturbed_img = perturbed.render(32)
        perturbed_order = order_multiplicative(perturbed_img)

        order_changes.append(abs(perturbed_order - base_order))

    return np.mean(order_changes)


def main():
    np.random.seed(42)

    n_samples = 500
    print(f"Generating {n_samples} CPPN samples...")

    results = []

    for i in range(n_samples):
        cppn = CPPN()
        # Add some variation by perturbing weights
        weights = cppn.get_weights()
        cppn.set_weights(weights * np.random.uniform(0.1, 3.0, len(weights)))

        img = cppn.render(32)
        order = order_multiplicative(img)
        gates = compute_gates(img)
        sensitivity = measure_perturbation_sensitivity(cppn)

        results.append({
            'order': order,
            'sensitivity': sensitivity,
            **gates
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # Extract arrays
    orders = np.array([r['order'] for r in results])
    sensitivities = np.array([r['sensitivity'] for r in results])
    spreads = np.array([r['spread'] for r in results])
    min_gates = np.array([r['min_gate'] for r in results])
    stds = np.array([r['std'] for r in results])
    products = np.array([r['product'] for r in results])

    print("\n=== CORRELATION ANALYSIS ===")

    # Core correlations
    r_order, p_order = stats.pearsonr(orders, sensitivities)
    r_spread, p_spread = stats.pearsonr(spreads, sensitivities)
    r_min, p_min = stats.pearsonr(min_gates, sensitivities)
    r_std, p_std = stats.pearsonr(stds, sensitivities)

    # Spearman for robustness
    rho_order, _ = stats.spearmanr(orders, sensitivities)
    rho_spread, _ = stats.spearmanr(spreads, sensitivities)

    print(f"\nPearson correlations with sensitivity:")
    print(f"  Order:      r = {r_order:.4f}, p = {p_order:.2e}")
    print(f"  Spread:     r = {r_spread:.4f}, p = {p_spread:.2e}")
    print(f"  Min gate:   r = {r_min:.4f}, p = {p_min:.2e}")
    print(f"  Gate std:   r = {r_std:.4f}, p = {p_std:.2e}")

    print(f"\nSpearman correlations:")
    print(f"  Order:      rho = {rho_order:.4f}")
    print(f"  Spread:     rho = {rho_spread:.4f}")

    # Compare correlation magnitudes
    # Use Fisher z-transformation to test difference
    def fisher_z(r):
        return 0.5 * np.log((1 + r) / (1 - r))

    z_order = fisher_z(r_order)
    z_spread = fisher_z(r_spread)
    se = np.sqrt(2 / (n_samples - 3))
    z_diff = (z_spread - z_order) / se
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    print(f"\n=== HYPOTHESIS TEST ===")
    print(f"Fisher z-test for correlation difference:")
    print(f"  z_order = {z_order:.4f}, z_spread = {z_spread:.4f}")
    print(f"  z_diff = {z_diff:.4f}, p = {p_diff:.2e}")

    # Effect size (Cohen's d for high vs low spread)
    median_spread = np.median(spreads)
    high_spread_sens = sensitivities[spreads > median_spread]
    low_spread_sens = sensitivities[spreads <= median_spread]

    pooled_std = np.sqrt((np.var(high_spread_sens) + np.var(low_spread_sens)) / 2)
    cohen_d = (np.mean(high_spread_sens) - np.mean(low_spread_sens)) / (pooled_std + 1e-10)

    print(f"\nEffect size (high vs low spread):")
    print(f"  High spread sensitivity: {np.mean(high_spread_sens):.4f}")
    print(f"  Low spread sensitivity:  {np.mean(low_spread_sens):.4f}")
    print(f"  Cohen's d = {cohen_d:.4f}")

    # Partial correlation: spread controlling for order
    from numpy.linalg import lstsq

    # Residualize sensitivity on order
    X = np.column_stack([np.ones(n_samples), orders])
    beta = lstsq(X, sensitivities, rcond=None)[0]
    sens_resid = sensitivities - X @ beta

    # Residualize spread on order
    beta_sp = lstsq(X, spreads, rcond=None)[0]
    spread_resid = spreads - X @ beta_sp

    r_partial, p_partial = stats.pearsonr(spread_resid, sens_resid)
    print(f"\nPartial correlation (spread | order):")
    print(f"  r_partial = {r_partial:.4f}, p = {p_partial:.2e}")

    # Regression: sensitivity ~ order + spread
    X_full = np.column_stack([np.ones(n_samples), orders, spreads])
    beta_full = lstsq(X_full, sensitivities, rcond=None)[0]
    pred_full = X_full @ beta_full
    r2_full = 1 - np.var(sensitivities - pred_full) / np.var(sensitivities)

    X_order = np.column_stack([np.ones(n_samples), orders])
    beta_order = lstsq(X_order, sensitivities, rcond=None)[0]
    pred_order = X_order @ beta_order
    r2_order = 1 - np.var(sensitivities - pred_order) / np.var(sensitivities)

    print(f"\nRegression R^2:")
    print(f"  Order only:        R^2 = {r2_order:.4f}")
    print(f"  Order + Spread:    R^2 = {r2_full:.4f}")
    print(f"  Improvement:       +{r2_full - r2_order:.4f}")

    # Test if spread adds predictive power (F-test)
    n, p_full, p_reduced = n_samples, 3, 2
    ss_full = np.sum((sensitivities - pred_full) ** 2)
    ss_reduced = np.sum((sensitivities - pred_order) ** 2)
    f_stat = ((ss_reduced - ss_full) / (p_full - p_reduced)) / (ss_full / (n - p_full))
    p_f = 1 - stats.f.cdf(f_stat, p_full - p_reduced, n - p_full)

    print(f"\nF-test for spread contribution:")
    print(f"  F = {f_stat:.4f}, p = {p_f:.2e}")

    # Summary statistics
    print("\n=== DESCRIPTIVE STATS ===")
    print(f"Order:       mean={np.mean(orders):.3f}, std={np.std(orders):.3f}")
    print(f"Spread:      mean={np.mean(spreads):.3f}, std={np.std(spreads):.3f}")
    print(f"Sensitivity: mean={np.mean(sensitivities):.4f}, std={np.std(sensitivities):.4f}")

    # Correlation between order and spread
    r_os, p_os = stats.pearsonr(orders, spreads)
    print(f"\nOrder-Spread correlation: r = {r_os:.4f}, p = {p_os:.2e}")

    print("\n=== VERDICT ===")
    hypothesis_supported = abs(r_spread) > abs(r_order) + 0.1 and p_spread < 0.01
    if hypothesis_supported:
        print("VALIDATED: Spread predicts sensitivity better than order")
    elif abs(r_spread) > abs(r_order):
        print("PARTIAL: Spread has higher correlation but difference < 0.1")
    else:
        print("REFUTED: Order predicts sensitivity as well or better than spread")

    # Check what actually predicts sensitivity best
    print("\n=== BEST PREDICTOR ===")
    predictors = {
        'order': r_order,
        'spread': r_spread,
        'min_gate': r_min,
        'gate_std': r_std
    }
    best = max(predictors.items(), key=lambda x: abs(x[1]))
    print(f"Best predictor: {best[0]} with r = {best[1]:.4f}")

    return {
        'r_order': r_order,
        'r_spread': r_spread,
        'r_partial': r_partial,
        'cohen_d': cohen_d,
        'p_spread': p_spread,
        'hypothesis_supported': hypothesis_supported
    }


if __name__ == '__main__':
    main()
