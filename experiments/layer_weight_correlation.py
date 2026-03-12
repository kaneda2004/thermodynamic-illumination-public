#!/usr/bin/env python3
"""
RES-092: Layer-wise Weight Correlations in High-Order CPPNs

Hypothesis: Layer-wise weight correlations differ between high-order and low-order
CPPNs, with successful architectures showing coordinated weight patterns across layers.

Methodology:
- Create 2-layer CPPNs (input->hidden1->hidden2->output)
- Sample many CPPNs and measure their order scores
- Extract weights by layer: input-to-h1, h1-to-h2, h2-to-output
- Compute within-CPPN cross-layer correlations
- Compare correlation patterns between high-order and low-order CPPNs

Related: RES-081 (weight sign patterns), RES-074 (hidden node optimization)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from dataclasses import dataclass, field
from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA
)
from scipy import stats
import random


def create_layered_cppn(n_hidden_per_layer: int = 4) -> tuple[CPPN, dict]:
    """
    Create a 2-layer CPPN with clear layer structure.
    Returns CPPN and dictionary mapping layer pairs to weight indices.
    """
    # Input nodes: x, y, r, bias (ids 0-3)
    # Output node: id 4
    # Hidden layer 1: ids 5, 6, 7, 8
    # Hidden layer 2: ids 9, 10, 11, 12

    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
    ]

    # Add hidden layer 1
    activations = list(ACTIVATIONS.keys())
    h1_ids = []
    for i in range(n_hidden_per_layer):
        hid = 5 + i
        h1_ids.append(hid)
        act = random.choice(activations)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    # Add hidden layer 2
    h2_ids = []
    for i in range(n_hidden_per_layer):
        hid = 5 + n_hidden_per_layer + i
        h2_ids.append(hid)
        act = random.choice(activations)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    connections = []
    layer_indices = {
        'input_to_h1': [],
        'h1_to_h2': [],
        'h2_to_output': []
    }

    conn_idx = 0

    # Input -> Hidden layer 1 (dense connection)
    for inp in [0, 1, 2, 3]:
        for hid in h1_ids:
            connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))
            layer_indices['input_to_h1'].append(conn_idx)
            conn_idx += 1

    # Hidden layer 1 -> Hidden layer 2 (dense connection)
    for h1 in h1_ids:
        for h2 in h2_ids:
            connections.append(Connection(h1, h2, np.random.randn() * PRIOR_SIGMA))
            layer_indices['h1_to_h2'].append(conn_idx)
            conn_idx += 1

    # Hidden layer 2 -> Output (dense connection)
    for h2 in h2_ids:
        connections.append(Connection(h2, 4, np.random.randn() * PRIOR_SIGMA))
        layer_indices['h2_to_output'].append(conn_idx)
        conn_idx += 1

    return CPPN(nodes=nodes, connections=connections), layer_indices


def extract_layer_weights(cppn: CPPN, layer_indices: dict) -> dict:
    """Extract weights organized by layer."""
    all_weights = cppn.get_weights()
    n_conn = len(cppn.connections)

    # Only use connection weights (not biases)
    conn_weights = all_weights[:n_conn]

    return {
        layer: conn_weights[indices] if max(indices) < len(conn_weights) else np.array([])
        for layer, indices in layer_indices.items()
    }


def compute_cross_layer_correlations(layer_weights: dict) -> dict:
    """Compute correlations between layer weight distributions."""
    layers = ['input_to_h1', 'h1_to_h2', 'h2_to_output']
    correlations = {}

    for i, l1 in enumerate(layers):
        for l2 in layers[i+1:]:
            w1 = layer_weights[l1]
            w2 = layer_weights[l2]

            if len(w1) == 0 or len(w2) == 0:
                correlations[f'{l1}_vs_{l2}'] = {
                    'magnitude_corr': np.nan,
                    'sign_agreement': np.nan
                }
                continue

            # Compare magnitude distributions
            # Use mean absolute values as representatives
            mag1 = np.abs(w1)
            mag2 = np.abs(w2)

            # Correlation of magnitude ranks (are layers "scaled" similarly?)
            # We'll look at mean magnitude ratio
            mean_mag1 = np.mean(mag1)
            mean_mag2 = np.mean(mag2)

            # Sign agreement: fraction of same-sign weights
            # Compare to random expectation (0.5)
            signs1 = np.sign(w1)
            signs2 = np.sign(w2)

            # For cross-layer: compare sign distribution statistics
            pos_frac1 = np.mean(signs1 > 0)
            pos_frac2 = np.mean(signs2 > 0)

            correlations[f'{l1}_vs_{l2}'] = {
                'mean_mag_ratio': mean_mag1 / (mean_mag2 + 1e-10),
                'std_mag_ratio': np.std(mag1) / (np.std(mag2) + 1e-10),
                'pos_frac_diff': abs(pos_frac1 - pos_frac2),
                'mean_mag1': mean_mag1,
                'mean_mag2': mean_mag2,
            }

    # Also compute within-CPPN weight statistics
    all_weights = np.concatenate([layer_weights[l] for l in layers if len(layer_weights[l]) > 0])
    correlations['global'] = {
        'total_magnitude': np.sum(np.abs(all_weights)),
        'weight_std': np.std(all_weights),
        'pos_fraction': np.mean(all_weights > 0),
        'kurtosis': stats.kurtosis(all_weights) if len(all_weights) > 4 else 0
    }

    return correlations


def main():
    print("=" * 70)
    print("RES-092: Layer-wise Weight Correlations in High-Order CPPNs")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    n_samples = 500
    n_hidden = 4  # Per layer

    print(f"\nSampling {n_samples} layered CPPNs...")
    print(f"Architecture: 4 inputs -> {n_hidden} hidden -> {n_hidden} hidden -> 1 output")
    print("-" * 70)

    # Collect data
    orders = []
    correlations_list = []
    weight_stats = []

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}...")

        cppn, layer_indices = create_layered_cppn(n_hidden)
        img = cppn.render(32)
        order = order_multiplicative(img)
        orders.append(order)

        layer_weights = extract_layer_weights(cppn, layer_indices)
        corrs = compute_cross_layer_correlations(layer_weights)
        correlations_list.append(corrs)

        # Store layer-wise summary stats
        weight_stats.append({
            'order': order,
            'input_to_h1_mean_mag': np.mean(np.abs(layer_weights['input_to_h1'])),
            'h1_to_h2_mean_mag': np.mean(np.abs(layer_weights['h1_to_h2'])),
            'h2_to_output_mean_mag': np.mean(np.abs(layer_weights['h2_to_output'])),
            'input_to_h1_std': np.std(layer_weights['input_to_h1']),
            'h1_to_h2_std': np.std(layer_weights['h1_to_h2']),
            'h2_to_output_std': np.std(layer_weights['h2_to_output']),
            'total_magnitude': corrs['global']['total_magnitude'],
        })

    orders = np.array(orders)

    # Split into high/low order groups (top/bottom quartiles)
    q25, q75 = np.percentile(orders, [25, 75])
    high_order_mask = orders >= q75
    low_order_mask = orders <= q25

    print(f"\nOrder statistics: mean={np.mean(orders):.4f}, std={np.std(orders):.4f}")
    print(f"Q25={q25:.4f}, Q75={q75:.4f}")
    print(f"High-order group (>= Q75): n={np.sum(high_order_mask)}")
    print(f"Low-order group (<= Q25): n={np.sum(low_order_mask)}")

    # Analyze differences between high and low order groups
    print("\n" + "=" * 70)
    print("LAYER-WISE WEIGHT ANALYSIS")
    print("=" * 70)

    metrics_to_test = [
        'input_to_h1_mean_mag', 'h1_to_h2_mean_mag', 'h2_to_output_mean_mag',
        'input_to_h1_std', 'h1_to_h2_std', 'h2_to_output_std',
        'total_magnitude'
    ]

    significant_results = []

    print("\nComparing High-Order vs Low-Order CPPNs:")
    print("-" * 70)
    print(f"{'Metric':<25} | {'High Mean':>10} | {'Low Mean':>10} | {'Effect d':>10} | {'p-value':>12}")
    print("-" * 70)

    for metric in metrics_to_test:
        high_vals = np.array([ws[metric] for i, ws in enumerate(weight_stats) if high_order_mask[i]])
        low_vals = np.array([ws[metric] for i, ws in enumerate(weight_stats) if low_order_mask[i]])

        high_mean = np.mean(high_vals)
        low_mean = np.mean(low_vals)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(high_vals, low_vals, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt((np.std(high_vals)**2 + np.std(low_vals)**2) / 2)
        effect_d = (high_mean - low_mean) / (pooled_std + 1e-10)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"{metric:<25} | {high_mean:>10.4f} | {low_mean:>10.4f} | {effect_d:>10.3f} | {p_val:>10.2e} {sig}")

        if p_val < 0.01 and abs(effect_d) > 0.5:
            significant_results.append({
                'metric': metric,
                'effect_d': effect_d,
                'p_val': p_val
            })

    # Cross-layer correlation analysis
    print("\n" + "=" * 70)
    print("CROSS-LAYER MAGNITUDE RATIOS")
    print("=" * 70)

    layer_pairs = ['input_to_h1_vs_h1_to_h2', 'h1_to_h2_vs_h2_to_output']

    for pair in layer_pairs:
        high_ratios = np.array([c[pair]['mean_mag_ratio'] for i, c in enumerate(correlations_list) if high_order_mask[i]])
        low_ratios = np.array([c[pair]['mean_mag_ratio'] for i, c in enumerate(correlations_list) if low_order_mask[i]])

        # Clean up infinities
        high_ratios = high_ratios[np.isfinite(high_ratios)]
        low_ratios = low_ratios[np.isfinite(low_ratios)]

        if len(high_ratios) > 5 and len(low_ratios) > 5:
            t_stat, p_val = stats.ttest_ind(high_ratios, low_ratios, equal_var=False)
            pooled_std = np.sqrt((np.std(high_ratios)**2 + np.std(low_ratios)**2) / 2)
            effect_d = (np.mean(high_ratios) - np.mean(low_ratios)) / (pooled_std + 1e-10)

            print(f"\n{pair}:")
            print(f"  High-order mean ratio: {np.mean(high_ratios):.4f} +/- {np.std(high_ratios):.4f}")
            print(f"  Low-order mean ratio: {np.mean(low_ratios):.4f} +/- {np.std(low_ratios):.4f}")
            print(f"  Effect size d={effect_d:.3f}, p={p_val:.2e}")

            if p_val < 0.01 and abs(effect_d) > 0.5:
                significant_results.append({
                    'metric': f'{pair}_ratio',
                    'effect_d': effect_d,
                    'p_val': p_val
                })

    # Correlation with order
    print("\n" + "=" * 70)
    print("CORRELATION WITH ORDER SCORE")
    print("=" * 70)

    order_correlations = []
    for metric in metrics_to_test:
        vals = np.array([ws[metric] for ws in weight_stats])
        r, p = stats.pearsonr(orders, vals)
        print(f"{metric:<25}: r={r:>7.4f}, p={p:.2e}")
        order_correlations.append({'metric': metric, 'r': r, 'p': p})

    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    max_effect = max([abs(sr['effect_d']) for sr in significant_results]) if significant_results else 0
    min_p = min([sr['p_val'] for sr in significant_results]) if significant_results else 1.0

    # Check for any strong correlations with order
    max_order_corr = max([abs(oc['r']) for oc in order_correlations])
    corrs_with_p = [oc['p'] for oc in order_correlations if abs(oc['r']) > 0.05]
    max_corr_p = min(corrs_with_p) if corrs_with_p else 1.0

    if len(significant_results) >= 2 and max_effect > 0.5:
        verdict = "VALIDATED"
        summary = f"High-order CPPNs show distinct layer-wise weight patterns. {len(significant_results)} metrics differ significantly (max d={max_effect:.2f}, min p={min_p:.2e})."
    elif len(significant_results) >= 1 or max_order_corr > 0.2:
        verdict = "INCONCLUSIVE"
        summary = f"Weak evidence for layer-wise differences. {len(significant_results)} significant metrics, max order correlation r={max_order_corr:.3f}."
    else:
        verdict = "REFUTED"
        summary = f"No significant layer-wise weight differences between high and low order CPPNs. Max effect d={max_effect:.2f}."

    print(f"\nStatus: {verdict}")
    print(f"Summary: {summary}")

    # Key metrics for logging
    print(f"\nKey Metrics:")
    print(f"  effect_size: {max_effect:.3f}")
    print(f"  p_value: {min_p:.2e}")
    print(f"  significant_metrics: {len(significant_results)}")
    print(f"  max_order_correlation: {max_order_corr:.4f}")

    return verdict, summary, {
        'max_effect_size': float(max_effect),
        'min_p_value': float(min_p),
        'significant_metrics': len(significant_results),
        'max_order_correlation': float(max_order_corr),
        'n_samples': n_samples
    }


if __name__ == "__main__":
    main()
