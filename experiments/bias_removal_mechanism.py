"""
RES-135: Investigate why zero-bias CPPNs produce higher-order images.

Hypothesis: Zero-bias CPPNs produce higher-order images because the bias term
adds constant offsets that shift activations away from the steepest gradients
of activation functions (tanh/sin), reducing local contrast and structure.

Mechanism theory:
- Activation functions like tanh/sigmoid have max gradient at x=0
- Bias shifts the input distribution away from 0
- This reduces the sensitivity of the output to input variations
- Lower sensitivity = lower contrast = lower order

Test approach:
1. Generate many CPPN pairs: with-bias vs zero-bias
2. Measure activation gradient statistics
3. Correlate bias magnitude with activation gradient steepness
4. Verify that lower gradients correlate with lower order
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, ACTIVATIONS, Node, Connection, PRIOR_SIGMA


def activation_gradient(func_name: str, x: np.ndarray) -> np.ndarray:
    """Compute numerical gradient of activation function."""
    eps = 1e-5
    func = ACTIVATIONS[func_name]
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def measure_activation_steepness(cppn: CPPN, size: int = 32) -> dict:
    """
    Measure how much of the activation happens in high-gradient regions.

    Returns statistics about where activations land on the activation curves.
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias_input = np.ones_like(x)

    # Track pre-activation values and gradients for each non-input node
    values = {0: x.flatten(), 1: y.flatten(), 2: r.flatten(), 3: bias_input.flatten()}

    pre_activations = []
    gradients = []

    for nid in cppn._get_eval_order():
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros(size*size) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        # Compute gradient at this pre-activation
        grad = activation_gradient(node.activation, total)

        pre_activations.append(total)
        gradients.append(grad)

        values[nid] = ACTIVATIONS[node.activation](total)

    # Aggregate statistics
    all_pre = np.concatenate(pre_activations)
    all_grad = np.concatenate(gradients)

    return {
        'mean_gradient': np.mean(all_grad),
        'median_gradient': np.median(all_grad),
        'mean_abs_pre_activation': np.mean(np.abs(all_pre)),
        'gradient_variance': np.var(all_grad),
        'high_gradient_fraction': np.mean(all_grad > 0.5),  # Fraction in high-slope region
    }


def create_paired_cppns(n_pairs: int, seed: int = 42) -> list:
    """Create pairs of CPPNs: one normal, one with all biases set to zero."""
    np.random.seed(seed)
    pairs = []

    for i in range(n_pairs):
        # Create normal CPPN
        cppn = CPPN()  # Uses default initialization with biases from prior

        # Create zero-bias version
        cppn_zero = cppn.copy()
        for node in cppn_zero.nodes:
            if node.id not in cppn_zero.input_ids:  # Don't touch input nodes
                node.bias = 0.0

        pairs.append((cppn, cppn_zero))

    return pairs


def main():
    print("=" * 70)
    print("RES-135: Bias Removal Mechanism Investigation")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 500

    # Generate paired samples
    pairs = create_paired_cppns(n_samples)

    results = {
        'normal': {'orders': [], 'gradients': [], 'high_grad_frac': [], 'abs_pre': []},
        'zero_bias': {'orders': [], 'gradients': [], 'high_grad_frac': [], 'abs_pre': []},
    }

    print(f"\nGenerating {n_samples} paired CPPN samples...")

    for i, (cppn_normal, cppn_zero) in enumerate(pairs):
        # Render images
        img_normal = cppn_normal.render(32)
        img_zero = cppn_zero.render(32)

        # Compute orders
        order_normal = order_multiplicative(img_normal)
        order_zero = order_multiplicative(img_zero)

        # Measure activation steepness
        stats_normal = measure_activation_steepness(cppn_normal, 32)
        stats_zero = measure_activation_steepness(cppn_zero, 32)

        results['normal']['orders'].append(order_normal)
        results['normal']['gradients'].append(stats_normal['mean_gradient'])
        results['normal']['high_grad_frac'].append(stats_normal['high_gradient_fraction'])
        results['normal']['abs_pre'].append(stats_normal['mean_abs_pre_activation'])

        results['zero_bias']['orders'].append(order_zero)
        results['zero_bias']['gradients'].append(stats_zero['mean_gradient'])
        results['zero_bias']['high_grad_frac'].append(stats_zero['high_gradient_fraction'])
        results['zero_bias']['abs_pre'].append(stats_zero['mean_abs_pre_activation'])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # Convert to arrays
    for key in results:
        for metric in results[key]:
            results[key][metric] = np.array(results[key][metric])

    # Analysis 1: Order comparison (replicating RES-126 finding)
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Order Comparison (Normal vs Zero-Bias)")
    print("=" * 70)

    order_diff = results['zero_bias']['orders'] - results['normal']['orders']
    effect_size = np.mean(order_diff) / np.std(order_diff) if np.std(order_diff) > 0 else 0
    t_stat, p_paired = stats.ttest_rel(results['zero_bias']['orders'], results['normal']['orders'])

    print(f"Normal CPPN order:    {np.mean(results['normal']['orders']):.4f} +/- {np.std(results['normal']['orders']):.4f}")
    print(f"Zero-bias order:      {np.mean(results['zero_bias']['orders']):.4f} +/- {np.std(results['zero_bias']['orders']):.4f}")
    print(f"Mean difference:      {np.mean(order_diff):.4f}")
    print(f"Effect size (d):      {effect_size:.4f}")
    print(f"Paired t-test p:      {p_paired:.2e}")

    # Analysis 2: Gradient comparison
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Activation Gradient Comparison")
    print("=" * 70)

    print(f"Normal mean gradient:     {np.mean(results['normal']['gradients']):.4f}")
    print(f"Zero-bias mean gradient:  {np.mean(results['zero_bias']['gradients']):.4f}")

    grad_diff = results['zero_bias']['gradients'] - results['normal']['gradients']
    t_grad, p_grad = stats.ttest_rel(results['zero_bias']['gradients'], results['normal']['gradients'])
    effect_grad = np.mean(grad_diff) / np.std(grad_diff) if np.std(grad_diff) > 0 else 0

    print(f"Mean gradient difference: {np.mean(grad_diff):.4f}")
    print(f"Effect size (d):          {effect_grad:.4f}")
    print(f"Paired t-test p:          {p_grad:.2e}")

    # Analysis 3: High-gradient fraction comparison
    print("\n" + "=" * 70)
    print("ANALYSIS 3: High-Gradient Region Fraction")
    print("=" * 70)

    print(f"Normal high-grad fraction:     {np.mean(results['normal']['high_grad_frac']):.4f}")
    print(f"Zero-bias high-grad fraction:  {np.mean(results['zero_bias']['high_grad_frac']):.4f}")

    hgf_diff = results['zero_bias']['high_grad_frac'] - results['normal']['high_grad_frac']
    t_hgf, p_hgf = stats.ttest_rel(results['zero_bias']['high_grad_frac'], results['normal']['high_grad_frac'])
    effect_hgf = np.mean(hgf_diff) / np.std(hgf_diff) if np.std(hgf_diff) > 0 else 0

    print(f"Effect size (d):               {effect_hgf:.4f}")
    print(f"Paired t-test p:               {p_hgf:.2e}")

    # Analysis 4: Pre-activation magnitude comparison
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Pre-Activation Magnitude")
    print("=" * 70)

    print(f"Normal |pre-activation|:     {np.mean(results['normal']['abs_pre']):.4f}")
    print(f"Zero-bias |pre-activation|:  {np.mean(results['zero_bias']['abs_pre']):.4f}")

    pre_diff = results['zero_bias']['abs_pre'] - results['normal']['abs_pre']
    t_pre, p_pre = stats.ttest_rel(results['zero_bias']['abs_pre'], results['normal']['abs_pre'])
    effect_pre = np.mean(pre_diff) / np.std(pre_diff) if np.std(pre_diff) > 0 else 0

    print(f"Effect size (d):             {effect_pre:.4f}")
    print(f"Paired t-test p:             {p_pre:.2e}")

    # Analysis 5: Correlation between gradient and order
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Gradient-Order Correlation")
    print("=" * 70)

    # Pool all samples
    all_orders = np.concatenate([results['normal']['orders'], results['zero_bias']['orders']])
    all_gradients = np.concatenate([results['normal']['gradients'], results['zero_bias']['gradients']])
    all_high_grad = np.concatenate([results['normal']['high_grad_frac'], results['zero_bias']['high_grad_frac']])

    r_grad, p_r_grad = stats.pearsonr(all_gradients, all_orders)
    r_hgf, p_r_hgf = stats.pearsonr(all_high_grad, all_orders)

    print(f"Correlation (mean gradient vs order): r={r_grad:.4f}, p={p_r_grad:.2e}")
    print(f"Correlation (high-grad frac vs order): r={r_hgf:.4f}, p={p_r_hgf:.2e}")

    # Analysis 6: Mediation analysis - Does gradient mediate bias->order?
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Mediation Analysis")
    print("=" * 70)
    print("Testing if activation gradients mediate the bias->order relationship...")

    # Create binary indicator for condition
    condition = np.array([0] * n_samples + [1] * n_samples)  # 0=normal, 1=zero-bias

    # Path a: condition -> mediator (gradient)
    r_a, p_a = stats.pointbiserialr(condition, all_gradients)
    print(f"Path a (bias removal -> gradient): r={r_a:.4f}, p={p_a:.2e}")

    # Path b: mediator -> outcome (controlling for condition)
    # Use residuals
    from scipy.stats import spearmanr
    r_b, p_b = spearmanr(all_gradients, all_orders)
    print(f"Path b (gradient -> order): rho={r_b:.4f}, p={p_b:.2e}")

    # Path c: condition -> outcome (total effect)
    r_c, p_c = stats.pointbiserialr(condition, all_orders)
    print(f"Path c (bias removal -> order): r={r_c:.4f}, p={p_c:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Validate the MECHANISM, not just the effect
    # Requirements:
    # 1. Zero-bias has higher order (replicates RES-126)
    # 2. Zero-bias has higher gradients (supports mechanism)
    # 3. Gradients correlate with order (mechanism is relevant)
    # 4. Pre-activation magnitude is lower for zero-bias (confirms bias shifts away from origin)
    hypothesis_supported = (
        effect_size > 0 and p_paired < 0.01 and  # Zero-bias higher order
        effect_grad > 0.2 and p_grad < 0.01 and  # Zero-bias higher gradients
        r_grad > 0.4 and p_r_grad < 0.01 and    # Gradient-order correlation
        effect_pre < -0.2  # Zero-bias has lower |pre-activation| (closer to origin)
    )

    print(f"\nHypothesis: Bias removal increases order by keeping activations")
    print(f"in high-gradient regions of activation functions.")
    print()
    print(f"Key findings:")
    print(f"  1. Order difference: d={effect_size:.3f} (zero-bias {'higher' if effect_size > 0 else 'lower'})")
    print(f"  2. Gradient difference: d={effect_grad:.3f} (zero-bias {'higher' if effect_grad > 0 else 'lower'})")
    print(f"  3. Gradient-order correlation: r={r_grad:.3f}")
    print()

    if hypothesis_supported:
        print("RESULT: HYPOTHESIS SUPPORTED")
        print("Zero-bias CPPNs have higher activation gradients, and higher")
        print("gradients correlate with higher order. This supports the mechanism.")
        status = "validated"
    else:
        # Check alternative: maybe effect is too small to matter
        if abs(effect_size) < 0.1 and p_paired > 0.01:
            print("RESULT: INCONCLUSIVE - Effect size too small to reliably detect mechanism")
            status = "inconclusive"
        elif effect_size > 0 and effect_grad <= 0:
            print("RESULT: REFUTED - Zero-bias has higher order but NOT higher gradients")
            print("The mechanism is NOT gradient-based. Needs alternative explanation.")
            status = "refuted"
        elif r_grad < 0.1:
            print("RESULT: REFUTED - Gradients don't correlate with order")
            status = "refuted"
        else:
            print("RESULT: INCONCLUSIVE - Mixed evidence")
            status = "inconclusive"

    # Final metrics for logging
    print("\n" + "=" * 70)
    print("METRICS FOR RESEARCH LOG")
    print("=" * 70)
    print(f"order_effect_size: {effect_size:.4f}")
    print(f"order_p_value: {p_paired:.2e}")
    print(f"gradient_effect_size: {effect_grad:.4f}")
    print(f"gradient_order_correlation: {r_grad:.4f}")
    print(f"high_grad_effect_size: {effect_hgf:.4f}")
    print(f"status: {status}")

    return {
        'status': status,
        'order_effect_size': effect_size,
        'order_p_value': p_paired,
        'gradient_effect_size': effect_grad,
        'gradient_order_correlation': r_grad,
    }


if __name__ == "__main__":
    main()
