#!/usr/bin/env python3
"""
Experiment: Role of biases in CPPN image order.

Hypothesis (RES-059): Node biases in CPPNs shift mean pixel density toward 0.5,
and bias magnitude correlates with image order.

Approach:
1. Generate many CPPN images with standard architecture
2. For each: record output node bias, pixel density, and order
3. Test: Does |bias| predict |density - 0.5|?
4. Control: Generate CPPN with bias=0 and compare density/order distributions
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, PRIOR_SIGMA


def analyze_bias_density_relationship(n_samples: int = 1000, seed: int = 42):
    """Analyze how biases affect density and order."""
    np.random.seed(seed)

    results = {
        'standard': {'biases': [], 'densities': [], 'orders': [], 'density_deviations': []},
        'zero_bias': {'densities': [], 'orders': [], 'density_deviations': []},
        'optimal_bias': {'densities': [], 'orders': [], 'density_deviations': []}
    }

    size = 32

    print(f"Generating {n_samples} CPPN samples for each condition...")

    # Standard CPPNs with biases from prior
    for i in range(n_samples):
        cppn = CPPN()  # Fresh random CPPN
        output_node = next(n for n in cppn.nodes if n.id == cppn.output_id)
        bias = output_node.bias

        img = cppn.render(size)  # Returns binarized uint8
        density = np.mean(img)
        order = order_multiplicative(img)

        results['standard']['biases'].append(bias)
        results['standard']['densities'].append(density)
        results['standard']['orders'].append(order)
        results['standard']['density_deviations'].append(abs(density - 0.5))

    # Control: Zero-bias CPPNs
    print("Generating zero-bias control samples...")
    for i in range(n_samples):
        cppn = CPPN()
        # Zero out all biases
        for node in cppn.nodes:
            if node.id not in cppn.input_ids:
                node.bias = 0.0

        img = cppn.render(size)
        density = np.mean(img)
        order = order_multiplicative(img)

        results['zero_bias']['densities'].append(density)
        results['zero_bias']['orders'].append(order)
        results['zero_bias']['density_deviations'].append(abs(density - 0.5))

    # Optimal bias: Adjust bias to center density at 0.5
    print("Generating optimal-bias samples...")
    for i in range(n_samples):
        cppn = CPPN()

        # First evaluate with current bias
        img = cppn.render(size)
        density = np.mean(img)

        # Simple heuristic: if density is off from 0.5, adjust bias
        # Sigmoid output: bias shift changes mean output
        output_node = next(n for n in cppn.nodes if n.id == cppn.output_id)

        # Binary search for optimal bias
        best_bias = output_node.bias
        best_density_dev = abs(density - 0.5)

        for test_bias in np.linspace(-3, 3, 50):
            output_node.bias = test_bias
            test_img = cppn.render(size)
            test_density = np.mean(test_img)
            dev = abs(test_density - 0.5)
            if dev < best_density_dev:
                best_density_dev = dev
                best_bias = test_bias

        output_node.bias = best_bias
        img = cppn.render(size)
        density = np.mean(img)
        order = order_multiplicative(img)

        results['optimal_bias']['densities'].append(density)
        results['optimal_bias']['orders'].append(order)
        results['optimal_bias']['density_deviations'].append(abs(density - 0.5))

    return results


def statistical_analysis(results):
    """Compute statistics for the experiment."""
    analysis = {}

    # 1. Correlation between bias and density deviation
    biases = np.array(results['standard']['biases'])
    density_devs = np.array(results['standard']['density_deviations'])

    # Does larger |bias| correlate with smaller |density - 0.5|?
    # We test: bias should help CENTER density
    # If bias = 0 and density is far from 0.5, bias should be able to fix it
    r_bias_dev, p_bias_dev = stats.pearsonr(np.abs(biases), density_devs)
    analysis['bias_deviation_correlation'] = {
        'r': float(r_bias_dev),
        'p': float(p_bias_dev),
        'interpretation': 'negative r means bias helps center density'
    }

    # 2. Compare density distributions
    std_densities = np.array(results['standard']['densities'])
    zero_densities = np.array(results['zero_bias']['densities'])
    opt_densities = np.array(results['optimal_bias']['densities'])

    analysis['density_stats'] = {
        'standard': {
            'mean': float(np.mean(std_densities)),
            'std': float(np.std(std_densities)),
            'mean_deviation_from_0.5': float(np.mean(np.abs(std_densities - 0.5)))
        },
        'zero_bias': {
            'mean': float(np.mean(zero_densities)),
            'std': float(np.std(zero_densities)),
            'mean_deviation_from_0.5': float(np.mean(np.abs(zero_densities - 0.5)))
        },
        'optimal_bias': {
            'mean': float(np.mean(opt_densities)),
            'std': float(np.std(opt_densities)),
            'mean_deviation_from_0.5': float(np.mean(np.abs(opt_densities - 0.5)))
        }
    }

    # 3. Compare order distributions
    std_orders = np.array(results['standard']['orders'])
    zero_orders = np.array(results['zero_bias']['orders'])
    opt_orders = np.array(results['optimal_bias']['orders'])

    # Mann-Whitney U test: standard vs zero-bias orders
    u_stat_vs_zero, p_vs_zero = stats.mannwhitneyu(std_orders, zero_orders, alternative='two-sided')
    # Effect size (rank-biserial correlation)
    n1, n2 = len(std_orders), len(zero_orders)
    effect_vs_zero = 1 - (2 * u_stat_vs_zero) / (n1 * n2)

    # Standard vs optimal
    u_stat_vs_opt, p_vs_opt = stats.mannwhitneyu(std_orders, opt_orders, alternative='two-sided')
    effect_vs_opt = 1 - (2 * u_stat_vs_opt) / (n1 * n2)

    analysis['order_comparison'] = {
        'standard_mean': float(np.mean(std_orders)),
        'standard_median': float(np.median(std_orders)),
        'standard_high_order_frac': float(np.mean(std_orders > 0.3)),
        'zero_bias_mean': float(np.mean(zero_orders)),
        'zero_bias_median': float(np.median(zero_orders)),
        'zero_bias_high_order_frac': float(np.mean(zero_orders > 0.3)),
        'optimal_bias_mean': float(np.mean(opt_orders)),
        'optimal_bias_median': float(np.median(opt_orders)),
        'optimal_bias_high_order_frac': float(np.mean(opt_orders > 0.3)),
        'standard_vs_zero': {
            'u_statistic': float(u_stat_vs_zero),
            'p_value': float(p_vs_zero),
            'effect_size': float(effect_vs_zero)
        },
        'standard_vs_optimal': {
            'u_statistic': float(u_stat_vs_opt),
            'p_value': float(p_vs_opt),
            'effect_size': float(effect_vs_opt)
        }
    }

    # 4. Key question: Does bias help density gate?
    # Compute density gate values
    def density_gate(d):
        return np.exp(-((d - 0.5) ** 2) / (2 * 0.25 ** 2))

    std_gates = density_gate(std_densities)
    zero_gates = density_gate(zero_densities)
    opt_gates = density_gate(opt_densities)

    t_stat_gate, p_gate = stats.ttest_ind(std_gates, zero_gates)
    cohens_d_gate = (np.mean(std_gates) - np.mean(zero_gates)) / np.sqrt(
        (np.var(std_gates) + np.var(zero_gates)) / 2
    )

    analysis['density_gate_comparison'] = {
        'standard_mean_gate': float(np.mean(std_gates)),
        'zero_bias_mean_gate': float(np.mean(zero_gates)),
        'optimal_bias_mean_gate': float(np.mean(opt_gates)),
        't_statistic': float(t_stat_gate),
        'p_value': float(p_gate),
        'cohens_d': float(cohens_d_gate)
    }

    # 5. Correlation: bias with order
    r_bias_order, p_bias_order = stats.pearsonr(biases, std_orders)
    r_abs_bias_order, p_abs_bias_order = stats.pearsonr(np.abs(biases), std_orders)

    analysis['bias_order_correlation'] = {
        'bias_vs_order_r': float(r_bias_order),
        'bias_vs_order_p': float(p_bias_order),
        'abs_bias_vs_order_r': float(r_abs_bias_order),
        'abs_bias_vs_order_p': float(p_abs_bias_order)
    }

    return analysis


def main():
    print("=" * 60)
    print("RES-059: Bias Role Experiment")
    print("=" * 60)
    print()

    # Run experiment
    results = analyze_bias_density_relationship(n_samples=1000, seed=42)

    # Statistical analysis
    analysis = statistical_analysis(results)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n1. DENSITY STATISTICS:")
    for cond, stats_dict in analysis['density_stats'].items():
        print(f"  {cond}:")
        print(f"    Mean density: {stats_dict['mean']:.4f}")
        print(f"    Std density: {stats_dict['std']:.4f}")
        print(f"    Mean |density - 0.5|: {stats_dict['mean_deviation_from_0.5']:.4f}")

    print("\n2. DENSITY GATE COMPARISON:")
    dgc = analysis['density_gate_comparison']
    print(f"  Standard mean gate: {dgc['standard_mean_gate']:.4f}")
    print(f"  Zero-bias mean gate: {dgc['zero_bias_mean_gate']:.4f}")
    print(f"  Optimal-bias mean gate: {dgc['optimal_bias_mean_gate']:.4f}")
    print(f"  Standard vs Zero: t={dgc['t_statistic']:.2f}, p={dgc['p_value']:.2e}, Cohen's d={dgc['cohens_d']:.3f}")

    print("\n3. ORDER COMPARISON:")
    oc = analysis['order_comparison']
    print(f"  Standard: mean={oc['standard_mean']:.4f}, high-order frac={oc['standard_high_order_frac']:.4f}")
    print(f"  Zero-bias: mean={oc['zero_bias_mean']:.4f}, high-order frac={oc['zero_bias_high_order_frac']:.4f}")
    print(f"  Optimal-bias: mean={oc['optimal_bias_mean']:.4f}, high-order frac={oc['optimal_bias_high_order_frac']:.4f}")
    print(f"  Standard vs Zero: p={oc['standard_vs_zero']['p_value']:.2e}, effect={oc['standard_vs_zero']['effect_size']:.3f}")
    print(f"  Standard vs Optimal: p={oc['standard_vs_optimal']['p_value']:.2e}, effect={oc['standard_vs_optimal']['effect_size']:.3f}")

    print("\n4. BIAS-ORDER CORRELATION:")
    boc = analysis['bias_order_correlation']
    print(f"  bias vs order: r={boc['bias_vs_order_r']:.4f}, p={boc['bias_vs_order_p']:.2e}")
    print(f"  |bias| vs order: r={boc['abs_bias_vs_order_r']:.4f}, p={boc['abs_bias_vs_order_p']:.2e}")

    # Determine verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Key metrics for decision
    dgc = analysis['density_gate_comparison']
    oc = analysis['order_comparison']

    # Does bias help density gate? (Cohen's d > 0.5 = medium effect)
    bias_helps_gate = dgc['cohens_d'] > 0.5 and dgc['p_value'] < 0.01

    # Does bias help order?
    bias_helps_order = (oc['standard_vs_zero']['effect_size'] > 0.2 and
                        oc['standard_vs_zero']['p_value'] < 0.01)

    # Optimal bias gives significant boost?
    optimal_helps = (oc['standard_vs_optimal']['effect_size'] > 0.2 and
                     oc['standard_vs_optimal']['p_value'] < 0.01)

    if bias_helps_gate and bias_helps_order:
        status = "VALIDATED"
        summary = f"Biases significantly improve density gate (d={dgc['cohens_d']:.2f}) and order (effect={oc['standard_vs_zero']['effect_size']:.2f})"
    elif bias_helps_gate or bias_helps_order:
        status = "PARTIALLY_VALIDATED"
        summary = f"Mixed evidence: gate effect d={dgc['cohens_d']:.2f}, order effect={oc['standard_vs_zero']['effect_size']:.2f}"
    else:
        status = "REFUTED"
        summary = f"Biases do not significantly affect density or order (gate d={dgc['cohens_d']:.2f}, order effect={oc['standard_vs_zero']['effect_size']:.2f})"

    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "bias_role"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'hypothesis': 'Node biases shift density toward 0.5 and correlate with order',
        'status': status,
        'summary': summary,
        'analysis': analysis,
        'metrics': {
            'effect_size': float(dgc['cohens_d']),
            'p_value': float(dgc['p_value']),
            'order_effect': float(oc['standard_vs_zero']['effect_size']),
            'order_p_value': float(oc['standard_vs_zero']['p_value'])
        }
    }

    with open(output_dir / "bias_role_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'bias_role_results.json'}")

    return output


if __name__ == "__main__":
    main()
