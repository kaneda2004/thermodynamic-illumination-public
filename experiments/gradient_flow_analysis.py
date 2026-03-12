"""
RES-095: Gradient Flow Analysis in CPPNs

Hypothesis: High-order CPPNs exhibit better gradient flow (lower vanishing/exploding
gradient metrics) than low-order CPPNs during forward propagation.

We measure gradient flow by tracking layer-wise activation statistics:
- Activation magnitude stability (std across layers)
- No saturation (activations not stuck at 0 or 1)
- Jacobian norm (if applicable)

For CPPNs we track pre-activation statistics at each node during forward pass.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, order_multiplicative,
    set_global_seed, PRIOR_SIGMA
)
from scipy import stats


def compute_activation_derivatives():
    """Get derivatives for each activation function at test points."""
    test_points = np.linspace(-5, 5, 1000)
    derivatives = {}
    eps = 1e-5

    for name, func in ACTIVATIONS.items():
        # Numerical derivative
        deriv = (func(test_points + eps) - func(test_points - eps)) / (2 * eps)
        derivatives[name] = {
            'mean_abs_deriv': np.mean(np.abs(deriv)),
            'max_deriv': np.max(np.abs(deriv)),
            'min_deriv': np.min(np.abs(deriv)),
            'deriv_at_zero': np.abs((func(eps) - func(-eps)) / (2 * eps))
        }
    return derivatives


def trace_forward_pass(cppn: CPPN, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Trace forward pass through CPPN, recording pre-activation values
    and activation outputs at each node.

    Returns statistics about gradient flow health.
    """
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)
    values = {0: x, 1: y, 2: r, 3: bias}

    # Track statistics
    pre_activations = []  # Before activation function
    post_activations = []  # After activation function
    node_info = []

    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)

        # Compute pre-activation (weighted sum + bias)
        total = np.zeros_like(x) + node.bias
        has_input = False
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
                has_input = True

        if not has_input and nid != cppn.output_id:
            # Skip disconnected nodes
            continue

        pre_activations.append(total.flatten())

        # Apply activation
        activated = ACTIVATIONS[node.activation](total)
        values[nid] = activated
        post_activations.append(activated.flatten())

        node_info.append({
            'id': nid,
            'activation': node.activation,
            'pre_mean': np.mean(total),
            'pre_std': np.std(total),
            'post_mean': np.mean(activated),
            'post_std': np.std(activated),
        })

    if len(pre_activations) == 0:
        return None

    # Gradient flow health metrics
    metrics = {}

    # 1. Pre-activation magnitude stability across layers
    pre_stds = [np.std(p) for p in pre_activations]
    metrics['pre_activation_std_ratio'] = max(pre_stds) / (min(pre_stds) + 1e-10)
    metrics['pre_activation_mean_std'] = np.mean(pre_stds)

    # 2. Check for vanishing: activations collapsing to narrow range
    post_ranges = [np.max(p) - np.min(p) for p in post_activations]
    metrics['output_range'] = post_ranges[-1]  # Final output range
    metrics['mean_layer_range'] = np.mean(post_ranges)

    # 3. Check for saturation (values near 0 or 1 for bounded activations)
    # Only for sigmoid/tanh outputs
    final_output = post_activations[-1]
    near_zero = np.mean(np.abs(final_output) < 0.01)
    near_one = np.mean(np.abs(final_output - 1) < 0.01)
    metrics['saturation_fraction'] = near_zero + near_one

    # 4. Layer-to-layer variance ratio (gradient flow indicator)
    if len(post_activations) >= 2:
        variance_ratios = []
        for i in range(1, len(post_activations)):
            var_curr = np.var(post_activations[i])
            var_prev = np.var(post_activations[i-1])
            if var_prev > 1e-10:
                variance_ratios.append(var_curr / var_prev)
        if variance_ratios:
            metrics['mean_variance_ratio'] = np.mean(variance_ratios)
            metrics['variance_stability'] = 1.0 / (1.0 + np.std(variance_ratios))
        else:
            metrics['mean_variance_ratio'] = 1.0
            metrics['variance_stability'] = 1.0
    else:
        metrics['mean_variance_ratio'] = 1.0
        metrics['variance_stability'] = 1.0

    # 5. Count depth (number of processing layers)
    metrics['depth'] = len(pre_activations)

    # 6. Combined gradient flow health score
    # Good gradient flow: stable variance ratios, no saturation, good output range
    health = (
        metrics['variance_stability'] *
        (1 - metrics['saturation_fraction']) *
        min(1.0, metrics['output_range'] / 0.5)  # Want range > 0.5
    )
    metrics['gradient_health'] = health

    return metrics


def generate_random_cppn(n_hidden: int = 3, complexity: float = 1.0) -> CPPN:
    """Generate a random CPPN with specified complexity."""
    cppn = CPPN()

    # Add hidden nodes
    activations = ['sin', 'cos', 'gauss', 'sigmoid', 'tanh', 'abs', 'relu', 'ring']
    next_id = 5
    hidden_ids = []

    for _ in range(n_hidden):
        act = np.random.choice(activations)
        bias = np.random.randn() * PRIOR_SIGMA * complexity
        cppn.nodes.append(Node(next_id, act, bias))
        hidden_ids.append(next_id)
        next_id += 1

    # Clear default connections
    cppn.connections = []

    # Add connections (inputs -> hidden -> output)
    # More complexity = more connections
    for hid in hidden_ids:
        # Connect random inputs to this hidden node
        for inp in cppn.input_ids:
            if np.random.random() < 0.6:
                w = np.random.randn() * PRIOR_SIGMA * complexity
                cppn.connections.append(Connection(inp, hid, w))

    # Connect hidden nodes to output
    for hid in hidden_ids:
        if np.random.random() < 0.7:
            w = np.random.randn() * PRIOR_SIGMA * complexity
            cppn.connections.append(Connection(hid, cppn.output_id, w))

    # Ensure at least one path to output
    if not any(c.to_id == cppn.output_id for c in cppn.connections):
        hid = np.random.choice(hidden_ids)
        w = np.random.randn() * PRIOR_SIGMA * complexity
        cppn.connections.append(Connection(hid, cppn.output_id, w))

    # Ensure hidden nodes have inputs
    for hid in hidden_ids:
        if not any(c.to_id == hid for c in cppn.connections):
            inp = np.random.choice(cppn.input_ids)
            w = np.random.randn() * PRIOR_SIGMA * complexity
            cppn.connections.append(Connection(inp, hid, w))

    return cppn


def run_experiment(n_samples: int = 500, n_bootstrap: int = 1000):
    """Main experiment: correlate gradient flow with order."""
    print("=" * 60)
    print("RES-095: Gradient Flow Analysis in CPPNs")
    print("=" * 60)

    set_global_seed(42)

    # Generate coordinate grid
    coords = np.linspace(-1, 1, 32)
    x, y = np.meshgrid(coords, coords)

    results = []

    print(f"\nGenerating {n_samples} CPPNs and measuring gradient flow...")

    for i in range(n_samples):
        # Vary complexity to get diverse CPPNs
        n_hidden = np.random.randint(1, 8)
        complexity = np.random.uniform(0.5, 2.0)

        cppn = generate_random_cppn(n_hidden, complexity)

        # Render and compute order
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Trace gradient flow
        gf_metrics = trace_forward_pass(cppn, x, y)

        if gf_metrics is not None:
            results.append({
                'order': order,
                'n_hidden': n_hidden,
                'complexity': complexity,
                **gf_metrics
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} CPPNs")

    print(f"\nAnalyzed {len(results)} valid CPPNs")

    # Extract arrays
    orders = np.array([r['order'] for r in results])
    gradient_health = np.array([r['gradient_health'] for r in results])
    variance_stability = np.array([r['variance_stability'] for r in results])
    saturation = np.array([r['saturation_fraction'] for r in results])
    output_range = np.array([r['output_range'] for r in results])
    variance_ratio = np.array([r['mean_variance_ratio'] for r in results])

    # Split into high/low order groups
    median_order = np.median(orders)
    high_order_mask = orders > median_order
    low_order_mask = orders <= median_order

    print(f"\nOrder statistics:")
    print(f"  Median order: {median_order:.4f}")
    print(f"  High-order group: {np.sum(high_order_mask)} samples, mean order: {np.mean(orders[high_order_mask]):.4f}")
    print(f"  Low-order group: {np.sum(low_order_mask)} samples, mean order: {np.mean(orders[low_order_mask]):.4f}")

    # Primary hypothesis: gradient health correlates with order
    print("\n" + "-" * 60)
    print("GRADIENT FLOW ANALYSIS")
    print("-" * 60)

    # Pearson correlation: order vs gradient health
    r_health, p_health = stats.pearsonr(orders, gradient_health)
    print(f"\n1. Order vs Gradient Health:")
    print(f"   Pearson r = {r_health:.4f}, p = {p_health:.2e}")

    # Spearman (rank) correlation for robustness
    rho_health, p_spearman = stats.spearmanr(orders, gradient_health)
    print(f"   Spearman rho = {rho_health:.4f}, p = {p_spearman:.2e}")

    # Two-sample t-test: high vs low order groups
    t_stat, p_ttest = stats.ttest_ind(
        gradient_health[high_order_mask],
        gradient_health[low_order_mask]
    )
    effect_size = (np.mean(gradient_health[high_order_mask]) - np.mean(gradient_health[low_order_mask])) / np.std(gradient_health)

    print(f"\n2. High vs Low Order Groups (Gradient Health):")
    print(f"   High-order mean: {np.mean(gradient_health[high_order_mask]):.4f}")
    print(f"   Low-order mean: {np.mean(gradient_health[low_order_mask]):.4f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_ttest:.2e}")
    print(f"   Cohen's d effect size: {effect_size:.4f}")

    # Individual gradient flow components
    print("\n3. Component Analysis:")

    # Variance stability
    r_var, p_var = stats.pearsonr(orders, variance_stability)
    print(f"   Order vs Variance Stability: r={r_var:.4f}, p={p_var:.2e}")

    # Saturation (expect negative correlation - less saturation = higher order)
    r_sat, p_sat = stats.pearsonr(orders, saturation)
    print(f"   Order vs Saturation: r={r_sat:.4f}, p={p_sat:.2e}")

    # Output range
    r_range, p_range = stats.pearsonr(orders, output_range)
    print(f"   Order vs Output Range: r={r_range:.4f}, p={p_range:.2e}")

    # Bootstrap confidence interval for main effect
    print("\n4. Bootstrap Confidence Interval (effect size):")
    bootstrap_effects = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(orders), size=len(orders), replace=True)
        boot_orders = orders[idx]
        boot_health = gradient_health[idx]
        boot_median = np.median(boot_orders)
        high_mask = boot_orders > boot_median
        low_mask = ~high_mask
        if np.sum(high_mask) > 0 and np.sum(low_mask) > 0:
            boot_effect = (np.mean(boot_health[high_mask]) - np.mean(boot_health[low_mask])) / (np.std(boot_health) + 1e-10)
            bootstrap_effects.append(boot_effect)

    ci_lower = np.percentile(bootstrap_effects, 2.5)
    ci_upper = np.percentile(bootstrap_effects, 97.5)
    print(f"   Effect size 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Determine status
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Criteria: significant correlation (p < 0.01) and effect size > 0.5
    significant = p_health < 0.01 and abs(r_health) > 0.1
    meaningful_effect = abs(effect_size) > 0.3
    correct_direction = r_health > 0  # Higher order should correlate with better gradient flow

    if significant and meaningful_effect and correct_direction:
        status = "VALIDATED"
        print(f"STATUS: {status}")
        print(f"High-order CPPNs show significantly better gradient flow.")
        print(f"Effect size d={effect_size:.3f} indicates a {'strong' if abs(effect_size) > 0.8 else 'moderate' if abs(effect_size) > 0.5 else 'small'} effect.")
    elif significant and not correct_direction:
        status = "REFUTED"
        print(f"STATUS: {status}")
        print(f"Significant relationship found, but in OPPOSITE direction.")
        print(f"Low-order CPPNs actually show better gradient flow metrics.")
    elif not significant:
        status = "REFUTED"
        print(f"STATUS: {status}")
        print(f"No significant relationship between order and gradient flow.")
        print(f"r={r_health:.3f} with p={p_health:.2e} fails significance threshold.")
    else:
        status = "INCONCLUSIVE"
        print(f"STATUS: {status}")
        print(f"Mixed evidence - significant but weak effect size.")

    print(f"\nKey Metrics:")
    print(f"  Correlation (r): {r_health:.4f}")
    print(f"  P-value: {p_health:.2e}")
    print(f"  Effect size (d): {effect_size:.4f}")

    return {
        'status': status,
        'correlation': r_health,
        'p_value': p_health,
        'effect_size': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': len(results),
        'high_order_health_mean': float(np.mean(gradient_health[high_order_mask])),
        'low_order_health_mean': float(np.mean(gradient_health[low_order_mask])),
    }


if __name__ == '__main__':
    results = run_experiment(n_samples=500, n_bootstrap=1000)

    print("\n" + "-" * 60)
    print("SUMMARY FOR LOG")
    print("-" * 60)
    print(f"Status: {results['status']}")
    print(f"Effect size: {results['effect_size']:.3f}")
    print(f"P-value: {results['p_value']:.2e}")
