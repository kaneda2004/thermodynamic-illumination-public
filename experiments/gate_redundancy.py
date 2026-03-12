#!/usr/bin/env python3
"""
RES-149: Test whether multiplicative gates have high redundancy.

Hypothesis: Multiplicative gates have high redundancy - improving one gate rarely
helps if another is bottleneck.

Method:
1. Generate CPPN images and compute all gate values
2. Identify which gate is the "bottleneck" (minimum gate value) for each image
3. Measure correlation between bottleneck identity and final order score
4. Test: if gates are redundant, images with same bottleneck should have similar
   order scores regardless of other gate values
5. Compute mutual information between each gate pair to quantify redundancy
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, gaussian_gate, compute_compressibility, compute_edge_density,
    compute_spectral_coherence, compute_symmetry, compute_connected_components
)


def compute_all_gates(img: np.ndarray) -> dict:
    """Compute all gate values for an image."""
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)
    symmetry = compute_symmetry(img)
    components = compute_connected_components(img)

    # Gate 1: Density
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)

    # Gate 3: Coherence
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility
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
    order = min(1.0, base_score * (1 + symmetry_bonus + component_bonus))

    return {
        'density_gate': density_gate,
        'edge_gate': edge_gate,
        'coherence_gate': coherence_gate,
        'compress_gate': compress_gate,
        'base_score': base_score,
        'order': order,
        'raw_density': density,
        'raw_edge': edge_density,
        'raw_coherence': coherence,
        'raw_compress': compressibility
    }


def identify_bottleneck(gates: dict) -> str:
    """Identify which gate is the bottleneck (minimum value)."""
    gate_values = {
        'density': gates['density_gate'],
        'edge': gates['edge_gate'],
        'coherence': gates['coherence_gate'],
        'compress': gates['compress_gate']
    }
    return min(gate_values, key=gate_values.get)


def compute_gate_redundancy(data: list) -> dict:
    """Compute pairwise correlations between gates (redundancy measure)."""
    gate_names = ['density_gate', 'edge_gate', 'coherence_gate', 'compress_gate']
    correlations = {}

    for i, g1 in enumerate(gate_names):
        for j, g2 in enumerate(gate_names):
            if i < j:
                v1 = [d[g1] for d in data]
                v2 = [d[g2] for d in data]
                r, p = stats.pearsonr(v1, v2)
                correlations[f'{g1[:4]}-{g2[:4]}'] = {'r': r, 'p': p}

    return correlations


def main():
    np.random.seed(42)
    n_samples = 500
    image_size = 32

    print("RES-149: Gate Redundancy Analysis")
    print("=" * 60)

    # Generate CPPN images and compute gates
    print(f"\nGenerating {n_samples} CPPN images...")
    data = []
    for i in range(n_samples):
        np.random.seed(i)
        cppn = CPPN()  # Dataclass with random weights in __post_init__
        img = cppn.render(image_size)
        gates = compute_all_gates(img)
        gates['bottleneck'] = identify_bottleneck(gates)
        data.append(gates)

    # 1. Bottleneck analysis
    print("\n1. BOTTLENECK ANALYSIS")
    print("-" * 40)
    bottleneck_counts = {}
    bottleneck_orders = {}
    for d in data:
        bn = d['bottleneck']
        bottleneck_counts[bn] = bottleneck_counts.get(bn, 0) + 1
        if bn not in bottleneck_orders:
            bottleneck_orders[bn] = []
        bottleneck_orders[bn].append(d['order'])

    print("Bottleneck frequency:")
    for bn, count in sorted(bottleneck_counts.items(), key=lambda x: -x[1]):
        mean_order = np.mean(bottleneck_orders[bn])
        std_order = np.std(bottleneck_orders[bn])
        print(f"  {bn:12s}: {count:4d} ({100*count/n_samples:5.1f}%), "
              f"mean order = {mean_order:.4f} +/- {std_order:.4f}")

    # 2. Gate correlation (redundancy)
    print("\n2. GATE CORRELATIONS (Redundancy)")
    print("-" * 40)
    correlations = compute_gate_redundancy(data)
    mean_abs_r = np.mean([abs(c['r']) for c in correlations.values()])
    print(f"Mean |r| across gate pairs: {mean_abs_r:.3f}")
    for pair, corr in sorted(correlations.items(), key=lambda x: -abs(x[1]['r'])):
        sig = "***" if corr['p'] < 0.001 else "**" if corr['p'] < 0.01 else "*" if corr['p'] < 0.05 else ""
        print(f"  {pair:20s}: r = {corr['r']:+.3f} {sig}")

    # 3. Bottleneck dominance: how much does min gate explain variance?
    print("\n3. BOTTLENECK DOMINANCE")
    print("-" * 40)
    min_gates = [min(d['density_gate'], d['edge_gate'], d['coherence_gate'], d['compress_gate']) for d in data]
    base_scores = [d['base_score'] for d in data]
    orders = [d['order'] for d in data]

    r_min_order, p_min = stats.pearsonr(min_gates, orders)
    print(f"Correlation(min_gate, order): r = {r_min_order:.3f}, p = {p_min:.2e}")

    # Compare: does min_gate predict order better than product of all gates?
    gate_products = [d['density_gate'] * d['edge_gate'] * d['coherence_gate'] * d['compress_gate'] for d in data]
    r_prod_order, p_prod = stats.pearsonr(gate_products, orders)
    print(f"Correlation(gate_product, order): r = {r_prod_order:.3f}, p = {p_prod:.2e}")

    # 4. Within-bottleneck variance: if gates are redundant, same bottleneck should give similar order
    print("\n4. WITHIN-BOTTLENECK VARIANCE")
    print("-" * 40)
    total_var = np.var(orders)
    within_var = sum(np.var(bottleneck_orders[bn]) * len(bottleneck_orders[bn])
                     for bn in bottleneck_orders) / n_samples
    between_var = total_var - within_var
    eta_squared = between_var / total_var if total_var > 0 else 0

    print(f"Total order variance:   {total_var:.6f}")
    print(f"Within-bottleneck var:  {within_var:.6f}")
    print(f"Between-bottleneck var: {between_var:.6f}")
    print(f"Eta-squared (bottleneck explains): {eta_squared:.3f}")

    # 5. Non-bottleneck gate contribution
    print("\n5. NON-BOTTLENECK GATE CONTRIBUTION")
    print("-" * 40)
    # For each sample, measure how much order changes with non-bottleneck gates
    # If redundancy is high, non-bottleneck gates should have low marginal contribution

    # Regress order on all gates
    from sklearn.linear_model import LinearRegression
    X = np.array([[d['density_gate'], d['edge_gate'], d['coherence_gate'], d['compress_gate']] for d in data])
    y = np.array(orders)
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    print(f"Linear R^2 (all gates -> order): {r2:.3f}")
    print("Gate coefficients:")
    for name, coef in zip(['density', 'edge', 'coherence', 'compress'], reg.coef_):
        print(f"  {name:12s}: {coef:+.4f}")

    # 6. Effect size: difference between bottleneck groups
    print("\n6. EFFECT SIZE")
    print("-" * 40)
    # Compare order distributions between dominant bottleneck types
    sorted_bn = sorted(bottleneck_counts.keys(), key=lambda x: -bottleneck_counts[x])
    if len(sorted_bn) >= 2:
        bn1, bn2 = sorted_bn[0], sorted_bn[1]
        o1, o2 = bottleneck_orders[bn1], bottleneck_orders[bn2]
        pooled_std = np.sqrt((np.var(o1) * (len(o1)-1) + np.var(o2) * (len(o2)-1)) / (len(o1) + len(o2) - 2))
        d = (np.mean(o1) - np.mean(o2)) / pooled_std if pooled_std > 0 else 0
        t_stat, p_val = stats.ttest_ind(o1, o2)
        print(f"Comparing {bn1} vs {bn2}:")
        print(f"  Mean order: {np.mean(o1):.4f} vs {np.mean(o2):.4f}")
        print(f"  Cohen's d = {d:.2f}")
        print(f"  t = {t_stat:.2f}, p = {p_val:.2e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Key finding: is redundancy high (|r| > 0.7) or low?
    high_redundancy = mean_abs_r > 0.5
    bottleneck_dominant = eta_squared > 0.3

    print(f"Mean gate correlation: {mean_abs_r:.3f} ({'HIGH' if high_redundancy else 'LOW'} redundancy)")
    print(f"Bottleneck eta-squared: {eta_squared:.3f} ({'HIGH' if bottleneck_dominant else 'LOW'} dominance)")
    print(f"Min gate -> order correlation: {r_min_order:.3f}")

    # Hypothesis test: redundancy means correlation should be > 0.5 AND bottleneck should explain > 30% variance
    validated = high_redundancy and bottleneck_dominant
    status = "VALIDATED" if validated else "REFUTED"

    print(f"\nHYPOTHESIS: {status}")
    if validated:
        print("Gates are redundant - min gate strongly determines order.")
    else:
        print("Gates are NOT highly redundant - all contribute independently to order.")

    # Compute effect size for redundancy claim
    # d = correlation difference from high redundancy threshold
    effect_size = mean_abs_r  # Use correlation as effect size

    return {
        'status': status.lower(),
        'mean_gate_correlation': mean_abs_r,
        'eta_squared': eta_squared,
        'min_gate_order_r': r_min_order,
        'effect_size': effect_size
    }


if __name__ == "__main__":
    results = main()
