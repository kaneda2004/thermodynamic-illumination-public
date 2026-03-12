"""
RES-116: CPPN network depth affects convergence speed during nested sampling.

Hypothesis: Deeper networks (more hidden layers) require fewer iterations to reach
high order thresholds.

Methodology:
- Create CPPNs with 0, 1, 2, and 3 hidden layers
- Run nested sampling and measure iterations to reach various order thresholds
- Compare convergence curves across architectures
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative, elliptical_slice_sample, log_prior
)
from scipy import stats


def create_cppn_with_depth(n_hidden_layers: int, hidden_size: int = 4, seed: int = None):
    """
    Create a CPPN with specified number of hidden layers.

    Architecture:
    - 4 inputs (x, y, r, bias)
    - n_hidden_layers layers with hidden_size nodes each
    - 1 output
    """
    if seed is not None:
        np.random.seed(seed)

    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
    ]
    input_ids = [0, 1, 2, 3]

    connections = []
    next_id = 4

    activations = ['sin', 'tanh', 'gauss', 'sigmoid']

    if n_hidden_layers == 0:
        # Direct input-to-output
        output_id = next_id
        nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))
        for inp in input_ids:
            connections.append(Connection(inp, output_id, np.random.randn() * PRIOR_SIGMA))
    else:
        # First hidden layer connects to inputs
        prev_layer_ids = input_ids

        for layer in range(n_hidden_layers):
            layer_ids = []
            for h in range(hidden_size):
                node_id = next_id
                next_id += 1
                act = activations[h % len(activations)]
                nodes.append(Node(node_id, act, np.random.randn() * PRIOR_SIGMA))
                layer_ids.append(node_id)

                # Connect from all previous layer nodes
                for prev_id in prev_layer_ids:
                    connections.append(Connection(prev_id, node_id, np.random.randn() * PRIOR_SIGMA))

            prev_layer_ids = layer_ids

        # Output layer
        output_id = next_id
        nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))
        for prev_id in prev_layer_ids:
            connections.append(Connection(prev_id, output_id, np.random.randn() * PRIOR_SIGMA))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=output_id)
    return cppn


def measure_convergence(cppn: CPPN, target_order: float, max_iterations: int = 500,
                        image_size: int = 32) -> dict:
    """
    Measure how many ESS iterations to reach target order.

    Returns dict with:
    - iterations_to_target: number of iterations (or max_iterations if not reached)
    - reached_target: bool
    - final_order: order value at end
    - orders_history: list of order values
    """
    orders = []
    current_cppn = cppn.copy()
    img = current_cppn.render(image_size)
    current_order = order_multiplicative(img)
    orders.append(current_order)

    iterations_to_target = max_iterations
    reached_target = False

    for i in range(max_iterations):
        # Use ESS to sample - threshold is current minus small epsilon to allow exploration
        threshold = current_order * 0.95  # Allow some downward movement

        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            current_cppn, threshold, image_size, order_multiplicative,
            max_contractions=50, max_restarts=3
        )

        if success and new_order > current_order:
            current_cppn = new_cppn
            current_order = new_order

        orders.append(current_order)

        if current_order >= target_order and not reached_target:
            iterations_to_target = i + 1
            reached_target = True
            break

    return {
        'iterations_to_target': iterations_to_target,
        'reached_target': reached_target,
        'final_order': current_order,
        'orders_history': orders
    }


def run_experiment(n_trials: int = 30, target_orders: list = None):
    """
    Run convergence experiment comparing different depths.
    """
    if target_orders is None:
        target_orders = [0.3, 0.4, 0.5]

    depths = [0, 1, 2, 3]
    results = {d: {t: [] for t in target_orders} for d in depths}

    print("=" * 60)
    print("RES-116: Convergence Speed vs Network Depth")
    print("=" * 60)
    print(f"Depths: {depths}")
    print(f"Target orders: {target_orders}")
    print(f"Trials per condition: {n_trials}")
    print()

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")
        base_seed = trial * 100

        for depth in depths:
            cppn = create_cppn_with_depth(depth, hidden_size=4, seed=base_seed + depth)

            for target in target_orders:
                result = measure_convergence(cppn.copy(), target, max_iterations=300)
                results[depth][target].append(result['iterations_to_target'])

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Analyze results for each target order
    statistical_results = []

    for target in target_orders:
        print(f"\nTarget order = {target}:")
        print("-" * 40)

        for depth in depths:
            iters = results[depth][target]
            reached = [i for i in iters if i < 300]
            pct_reached = len(reached) / len(iters) * 100
            mean_iters = np.mean(iters)
            std_iters = np.std(iters)
            print(f"  Depth {depth}: {mean_iters:.1f} +/- {std_iters:.1f} iterations "
                  f"({pct_reached:.0f}% reached)")

        # Compare depth=0 vs depth=2 (main comparison)
        iters_shallow = results[0][target]
        iters_deep = results[2][target]

        t_stat, p_value = stats.ttest_ind(iters_shallow, iters_deep)
        effect_size = (np.mean(iters_shallow) - np.mean(iters_deep)) / np.sqrt(
            (np.var(iters_shallow) + np.var(iters_deep)) / 2)

        statistical_results.append({
            'target': target,
            't_stat': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'shallow_mean': np.mean(iters_shallow),
            'deep_mean': np.mean(iters_deep)
        })

        print(f"\n  Depth 0 vs Depth 2:")
        print(f"    t-statistic: {t_stat:.3f}")
        print(f"    p-value: {p_value:.6f}")
        print(f"    Effect size (Cohen's d): {effect_size:.3f}")

    # Overall analysis - compare all depths with ANOVA
    print("\n" + "=" * 60)
    print("OVERALL ANALYSIS (ANOVA across depths for target=0.4)")
    print("=" * 60)

    target_for_anova = 0.4
    groups = [results[d][target_for_anova] for d in depths]
    f_stat, p_anova = stats.f_oneway(*groups)

    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_anova:.6f}")

    # Correlation between depth and convergence speed
    all_depths = []
    all_iters = []
    for d in depths:
        for i in results[d][target_for_anova]:
            all_depths.append(d)
            all_iters.append(i)

    r, p_corr = stats.pearsonr(all_depths, all_iters)
    print(f"\nCorrelation (depth vs iterations): r={r:.3f}, p={p_corr:.6f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Use the middle target for primary result
    primary = statistical_results[1]  # target=0.4

    validated = primary['p_value'] < 0.01 and abs(primary['effect_size']) > 0.5

    if validated:
        if primary['effect_size'] > 0:
            direction = "SLOWER"
            verdict = "REFUTED (deeper networks converge SLOWER)"
        else:
            direction = "FASTER"
            verdict = "VALIDATED (deeper networks converge faster)"
    else:
        verdict = "INCONCLUSIVE"

    print(f"Status: {verdict}")
    print(f"Primary comparison (target=0.4): p={primary['p_value']:.6f}, d={primary['effect_size']:.3f}")
    print(f"Shallow (depth=0): {primary['shallow_mean']:.1f} iterations")
    print(f"Deep (depth=2): {primary['deep_mean']:.1f} iterations")

    return {
        'results': results,
        'statistical': statistical_results,
        'anova_p': p_anova,
        'correlation_r': r,
        'correlation_p': p_corr,
        'validated': validated,
        'effect_size': primary['effect_size'],
        'p_value': primary['p_value']
    }


if __name__ == "__main__":
    np.random.seed(42)
    output = run_experiment(n_trials=30)

    print("\n" + "=" * 60)
    print("SUMMARY FOR LOG")
    print("=" * 60)
    print(f"Effect size: {output['effect_size']:.3f}")
    print(f"P-value: {output['p_value']:.6f}")
    print(f"ANOVA p: {output['anova_p']:.6f}")
    print(f"Validated: {output['validated']}")
