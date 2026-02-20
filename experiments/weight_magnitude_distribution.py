"""
RES-097: Weight Magnitude Distribution Analysis

Hypothesis: High-order CPPN networks exhibit heavier-tailed weight magnitude
distributions (larger kurtosis, larger max/mean ratio) compared to low-order networks.

Rationale: If structured images require specific weight configurations, high-order
CPPNs may show distinctive distributional characteristics in their weights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    nested_sampling_v3,
    set_global_seed
)
import json


def compute_weight_statistics(cppn: CPPN) -> dict:
    """Extract statistical properties of weight magnitude distribution."""
    weights = cppn.get_weights()
    abs_weights = np.abs(weights)

    # Basic statistics
    mean_mag = np.mean(abs_weights)
    std_mag = np.std(abs_weights)
    max_mag = np.max(abs_weights)

    # Distribution shape measures
    kurtosis = stats.kurtosis(abs_weights, fisher=True)  # Excess kurtosis (0 for normal)
    skewness = stats.skew(abs_weights)

    # Tail measures
    max_mean_ratio = max_mag / mean_mag if mean_mag > 0 else 0

    # Percentile ratios (tail heaviness)
    p90 = np.percentile(abs_weights, 90)
    p50 = np.percentile(abs_weights, 50)
    p90_p50_ratio = p90 / p50 if p50 > 0 else 0

    # Gini coefficient (measure of inequality in weight magnitudes)
    sorted_weights = np.sort(abs_weights)
    n = len(sorted_weights)
    if n > 1 and np.sum(sorted_weights) > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights) - (n + 1) * np.sum(sorted_weights)) / (n * np.sum(sorted_weights))
    else:
        gini = 0

    return {
        'mean_magnitude': float(mean_mag),
        'std_magnitude': float(std_mag),
        'max_magnitude': float(max_mag),
        'kurtosis': float(kurtosis),
        'skewness': float(skewness),
        'max_mean_ratio': float(max_mean_ratio),
        'p90_p50_ratio': float(p90_p50_ratio),
        'gini': float(gini),
        'n_weights': int(len(weights))
    }


def sample_cppns_by_order(n_samples: int = 500, seed: int = 42) -> list:
    """Sample CPPNs and compute their order and weight statistics."""
    set_global_seed(seed)

    samples = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        weight_stats = compute_weight_statistics(cppn)

        samples.append({
            'order': order,
            **weight_stats
        })

        if (i + 1) % 100 == 0:
            print(f"  Sampled {i + 1}/{n_samples} CPPNs")

    return samples


def analyze_by_order_groups(samples: list, n_groups: int = 5) -> dict:
    """Analyze weight statistics by order quintiles."""
    orders = np.array([s['order'] for s in samples])
    percentiles = np.percentile(orders, np.linspace(0, 100, n_groups + 1))

    groups = []
    for i in range(n_groups):
        mask = (orders >= percentiles[i]) & (orders < percentiles[i + 1])
        if i == n_groups - 1:  # Include max in last group
            mask = (orders >= percentiles[i]) & (orders <= percentiles[i + 1])

        group_samples = [s for s, m in zip(samples, mask) if m]

        if len(group_samples) > 0:
            groups.append({
                'group': i + 1,
                'order_range': (float(percentiles[i]), float(percentiles[i + 1])),
                'n_samples': len(group_samples),
                'mean_order': float(np.mean([s['order'] for s in group_samples])),
                'mean_kurtosis': float(np.mean([s['kurtosis'] for s in group_samples])),
                'std_kurtosis': float(np.std([s['kurtosis'] for s in group_samples])),
                'mean_max_mean_ratio': float(np.mean([s['max_mean_ratio'] for s in group_samples])),
                'std_max_mean_ratio': float(np.std([s['max_mean_ratio'] for s in group_samples])),
                'mean_gini': float(np.mean([s['gini'] for s in group_samples])),
                'std_gini': float(np.std([s['gini'] for s in group_samples])),
                'mean_p90_p50_ratio': float(np.mean([s['p90_p50_ratio'] for s in group_samples])),
                'std_p90_p50_ratio': float(np.std([s['p90_p50_ratio'] for s in group_samples])),
            })

    return groups


def run_correlation_tests(samples: list) -> dict:
    """Test correlations between order and weight distribution metrics."""
    orders = np.array([s['order'] for s in samples])

    metrics = ['kurtosis', 'max_mean_ratio', 'gini', 'p90_p50_ratio', 'skewness']
    results = {}

    for metric in metrics:
        values = np.array([s[metric] for s in samples])

        # Remove any NaN or inf values
        valid_mask = np.isfinite(values) & np.isfinite(orders)
        clean_orders = orders[valid_mask]
        clean_values = values[valid_mask]

        if len(clean_orders) > 10:
            # Pearson correlation
            r, p = stats.pearsonr(clean_orders, clean_values)

            # Spearman rank correlation (more robust to non-linearity)
            rho, p_spearman = stats.spearmanr(clean_orders, clean_values)

            results[metric] = {
                'pearson_r': float(r),
                'pearson_p': float(p),
                'spearman_rho': float(rho),
                'spearman_p': float(p_spearman),
                'n_valid': int(len(clean_orders))
            }

    return results


def run_high_order_sampling(n_iterations: int = 300, n_live: int = 50, seed: int = 123) -> list:
    """Sample high-order CPPNs via nested sampling and extract weight stats from live points."""
    print("\nRunning nested sampling to get high-order CPPNs...")

    dead_points, live_points, summary = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=32,
        order_fn=order_multiplicative,
        sampling_mode="measure",
        track_metrics=False,
        output_dir="experiments/weight_dist_output",
        seed=seed
    )

    # Extract from final live points (highest order)
    # LivePoint has 'cppn' attribute (not 'generator')
    high_order_samples = []
    for lp in live_points:
        weight_stats = compute_weight_statistics(lp.cppn)
        high_order_samples.append({
            'order': lp.order_value,
            **weight_stats
        })

    # Note: DeadPoint doesn't store CPPN, so we can't extract from them
    # This is okay - we have 1000 prior samples + 50 high-order live points

    return high_order_samples


def main():
    print("=" * 70)
    print("RES-097: Weight Magnitude Distribution Analysis")
    print("=" * 70)

    # Part 1: Random CPPN sampling (baseline distribution)
    print("\nPart 1: Sampling random CPPNs from prior...")
    prior_samples = sample_cppns_by_order(n_samples=1000, seed=42)

    # Part 2: Nested sampling for high-order CPPNs
    print("\nPart 2: Nested sampling for high-order CPPNs...")
    nested_samples = run_high_order_sampling(n_iterations=400, n_live=50, seed=123)

    # Combine all samples
    all_samples = prior_samples + nested_samples
    print(f"\nTotal samples: {len(all_samples)}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Group analysis
    print("\n--- Order Quintile Analysis ---")
    groups = analyze_by_order_groups(all_samples, n_groups=5)

    for g in groups:
        print(f"\nGroup {g['group']} (order {g['order_range'][0]:.4f} - {g['order_range'][1]:.4f}):")
        print(f"  n={g['n_samples']}, mean_order={g['mean_order']:.4f}")
        print(f"  kurtosis: {g['mean_kurtosis']:.3f} +/- {g['std_kurtosis']:.3f}")
        print(f"  max/mean ratio: {g['mean_max_mean_ratio']:.3f} +/- {g['std_max_mean_ratio']:.3f}")
        print(f"  gini: {g['mean_gini']:.3f} +/- {g['std_gini']:.3f}")

    # Correlation tests
    print("\n--- Correlation Tests (Order vs Weight Metrics) ---")
    correlations = run_correlation_tests(all_samples)

    primary_metrics = ['kurtosis', 'max_mean_ratio', 'gini']

    for metric in primary_metrics:
        if metric in correlations:
            c = correlations[metric]
            print(f"\n{metric}:")
            print(f"  Pearson r = {c['pearson_r']:.4f}, p = {c['pearson_p']:.2e}")
            print(f"  Spearman rho = {c['spearman_rho']:.4f}, p = {c['spearman_p']:.2e}")

    # Statistical test: Compare lowest vs highest quintile
    print("\n--- Statistical Test: Lowest vs Highest Order Quintile ---")

    low_group = groups[0]
    high_group = groups[-1]

    low_samples = [s for s in all_samples if low_group['order_range'][0] <= s['order'] < low_group['order_range'][1]]
    high_samples = [s for s in all_samples if high_group['order_range'][0] <= s['order'] <= high_group['order_range'][1]]

    test_metrics = ['kurtosis', 'max_mean_ratio', 'gini']
    test_results = {}

    for metric in test_metrics:
        low_vals = [s[metric] for s in low_samples if np.isfinite(s[metric])]
        high_vals = [s[metric] for s in high_samples if np.isfinite(s[metric])]

        if len(low_vals) > 5 and len(high_vals) > 5:
            # Mann-Whitney U test (non-parametric)
            u_stat, p_mw = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(low_vals) + np.var(high_vals)) / 2)
            cohens_d = (np.mean(high_vals) - np.mean(low_vals)) / pooled_std if pooled_std > 0 else 0

            test_results[metric] = {
                'low_mean': float(np.mean(low_vals)),
                'high_mean': float(np.mean(high_vals)),
                'u_statistic': float(u_stat),
                'p_value': float(p_mw),
                'cohens_d': float(cohens_d)
            }

            print(f"\n{metric}:")
            print(f"  Low-order mean: {np.mean(low_vals):.4f}")
            print(f"  High-order mean: {np.mean(high_vals):.4f}")
            print(f"  Mann-Whitney p = {p_mw:.2e}")
            print(f"  Cohen's d = {cohens_d:.3f}")

    # Determine verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check hypothesis: heavier tails (larger kurtosis, larger max/mean ratio)
    kurtosis_result = test_results.get('kurtosis', {})
    max_mean_result = test_results.get('max_mean_ratio', {})
    gini_result = test_results.get('gini', {})

    # Hypothesis is validated if:
    # 1. At least one metric shows significant difference (p < 0.01)
    # 2. Effect size > 0.5
    # 3. Direction is as hypothesized (high-order has heavier tails)

    validated = False
    key_finding = "No significant difference found"
    effect_size = 0
    p_value = 1.0

    # Check kurtosis (heavier tails = higher kurtosis)
    if kurtosis_result:
        if kurtosis_result['p_value'] < 0.01 and abs(kurtosis_result['cohens_d']) > 0.5:
            if kurtosis_result['cohens_d'] > 0:  # High-order has higher kurtosis
                validated = True
                key_finding = "High-order CPPNs have significantly higher kurtosis"
                effect_size = kurtosis_result['cohens_d']
                p_value = kurtosis_result['p_value']

    # Check max/mean ratio
    if max_mean_result:
        if max_mean_result['p_value'] < 0.01 and abs(max_mean_result['cohens_d']) > 0.5:
            if max_mean_result['cohens_d'] > 0:  # High-order has higher ratio
                if not validated:
                    validated = True
                    key_finding = "High-order CPPNs have significantly higher max/mean ratio"
                    effect_size = max_mean_result['cohens_d']
                    p_value = max_mean_result['p_value']
                else:
                    key_finding += " and max/mean ratio"

    # Check if effect is in opposite direction (refuted)
    refuted = False
    if kurtosis_result and max_mean_result:
        if (kurtosis_result['cohens_d'] < -0.5 and kurtosis_result['p_value'] < 0.01):
            refuted = True
            key_finding = "REFUTED: High-order CPPNs have LIGHTER tails (lower kurtosis)"
            effect_size = kurtosis_result['cohens_d']
            p_value = kurtosis_result['p_value']

    if validated:
        status = "VALIDATED"
    elif refuted:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"

    print(f"\nStatus: {status}")
    print(f"Finding: {key_finding}")
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    print(f"P-value: {p_value:.2e}")

    # Save results
    results = {
        'experiment_id': 'RES-097',
        'hypothesis': 'High-order CPPN networks exhibit heavier-tailed weight magnitude distributions',
        'status': status,
        'finding': key_finding,
        'effect_size': effect_size,
        'p_value': p_value,
        'groups': groups,
        'correlations': correlations,
        'tests': test_results,
        'n_prior_samples': len(prior_samples),
        'n_nested_samples': len(nested_samples)
    }

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/weight_magnitude_distribution.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to experiments/results/weight_magnitude_distribution.json")

    return status, effect_size, p_value


if __name__ == "__main__":
    main()
