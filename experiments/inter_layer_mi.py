"""
RES-087: Inter-Layer Mutual Information in CPPNs

Hypothesis: High-order CPPN images exhibit higher inter-layer mutual information
between hidden layer activations than low-order CPPN images, indicating more
coordinated information processing.

Approach:
1. Generate CPPNs with multiple hidden nodes
2. Extract activations from each hidden node
3. Compute pairwise mutual information between hidden node activations
4. Correlate average inter-layer MI with order score
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative, set_global_seed
)


def create_cppn_with_hidden(n_hidden: int = 4, seed: int = None) -> CPPN:
    """Create a CPPN with specified number of hidden nodes."""
    if seed is not None:
        np.random.seed(seed)

    input_ids = [0, 1, 2, 3]  # x, y, r, bias
    output_id = 4
    hidden_start = 5

    # Create nodes
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
        Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA),
    ]

    # Add hidden nodes with random activations
    act_names = list(ACTIVATIONS.keys())
    hidden_ids = []
    for i in range(n_hidden):
        hid = hidden_start + i
        hidden_ids.append(hid)
        act = np.random.choice(act_names)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    # Create connections: input -> hidden, hidden -> hidden, hidden -> output
    connections = []

    # Input to hidden (random subset)
    for hid in hidden_ids:
        for inp in input_ids:
            if np.random.rand() < 0.7:  # 70% connection probability
                connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))

    # Hidden to hidden (sparse connections, avoid cycles by only connecting lower -> higher)
    for i, hid_from in enumerate(hidden_ids):
        for hid_to in hidden_ids[i+1:]:
            if np.random.rand() < 0.3:
                connections.append(Connection(hid_from, hid_to, np.random.randn() * PRIOR_SIGMA))

    # Hidden to output
    for hid in hidden_ids:
        if np.random.rand() < 0.8:  # 80% probability
            connections.append(Connection(hid, output_id, np.random.randn() * PRIOR_SIGMA))

    # Ensure at least one path exists
    if not any(c.to_id == output_id for c in connections):
        connections.append(Connection(hidden_ids[-1], output_id, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=output_id)


def get_layer_activations(cppn: CPPN, size: int = 32) -> dict:
    """Get activations of all nodes for given input size."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}

    eval_order = cppn._get_eval_order()
    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[nid] = ACTIVATIONS[node.activation](total)

    return values


def discretize_activation(arr: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Discretize continuous activations into bins for MI computation."""
    flat = arr.flatten()
    # Handle edge case of constant values
    if np.std(flat) < 1e-10:
        return np.zeros_like(flat, dtype=int)

    # Use quantile-based binning for robustness
    percentiles = np.percentile(flat, np.linspace(0, 100, n_bins + 1))
    # Remove duplicates
    percentiles = np.unique(percentiles)
    return np.digitize(flat, percentiles[1:-1])


def compute_pairwise_mi(activations: dict, hidden_ids: list, n_bins: int = 20) -> float:
    """Compute average pairwise MI between hidden layer activations."""
    if len(hidden_ids) < 2:
        return 0.0

    mi_values = []
    for i, hid1 in enumerate(hidden_ids):
        for hid2 in hidden_ids[i+1:]:
            if hid1 in activations and hid2 in activations:
                disc1 = discretize_activation(activations[hid1], n_bins)
                disc2 = discretize_activation(activations[hid2], n_bins)
                mi = mutual_info_score(disc1, disc2)
                mi_values.append(mi)

    return np.mean(mi_values) if mi_values else 0.0


def run_experiment(n_samples: int = 200, n_hidden: int = 4, seed: int = 42):
    """Run the inter-layer MI experiment."""
    set_global_seed(seed)

    results = []
    hidden_start = 5
    hidden_ids = list(range(hidden_start, hidden_start + n_hidden))

    print(f"Generating {n_samples} CPPNs with {n_hidden} hidden nodes...")

    for i in range(n_samples):
        cppn = create_cppn_with_hidden(n_hidden=n_hidden, seed=seed + i)

        # Get activations
        activations = get_layer_activations(cppn, size=32)

        # Render and compute order
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        # Compute inter-layer MI
        inter_mi = compute_pairwise_mi(activations, hidden_ids)

        # Also compute input-hidden MI as baseline
        input_hidden_mi = []
        for inp in [0, 1, 2]:  # x, y, r (not bias)
            for hid in hidden_ids:
                if hid in activations:
                    disc_inp = discretize_activation(activations[inp])
                    disc_hid = discretize_activation(activations[hid])
                    mi = mutual_info_score(disc_inp, disc_hid)
                    input_hidden_mi.append(mi)
        avg_input_hidden_mi = np.mean(input_hidden_mi) if input_hidden_mi else 0.0

        results.append({
            'order': order,
            'inter_layer_mi': inter_mi,
            'input_hidden_mi': avg_input_hidden_mi,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    return results


def analyze_results(results: list) -> dict:
    """Analyze correlation between order and inter-layer MI."""
    orders = np.array([r['order'] for r in results])
    inter_mi = np.array([r['inter_layer_mi'] for r in results])
    input_hidden_mi = np.array([r['input_hidden_mi'] for r in results])

    # Pearson correlation: order vs inter-layer MI
    r_inter, p_inter = stats.pearsonr(orders, inter_mi)

    # Pearson correlation: order vs input-hidden MI
    r_input, p_input = stats.pearsonr(orders, input_hidden_mi)

    # Split into high/low order groups for effect size
    median_order = np.median(orders)
    high_order_mi = inter_mi[orders > median_order]
    low_order_mi = inter_mi[orders <= median_order]

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(high_order_mi) + np.var(low_order_mi)) / 2)
    cohens_d = (np.mean(high_order_mi) - np.mean(low_order_mi)) / (pooled_std + 1e-10)

    # Mann-Whitney U test (non-parametric)
    u_stat, p_mannwhitney = stats.mannwhitneyu(high_order_mi, low_order_mi, alternative='greater')

    return {
        'n_samples': len(results),
        'correlation_inter_layer': r_inter,
        'p_value_inter_layer': p_inter,
        'correlation_input_hidden': r_input,
        'p_value_input_hidden': p_input,
        'mean_high_order_mi': float(np.mean(high_order_mi)),
        'mean_low_order_mi': float(np.mean(low_order_mi)),
        'cohens_d': cohens_d,
        'mannwhitney_p': p_mannwhitney,
        'order_range': (float(np.min(orders)), float(np.max(orders))),
        'inter_mi_range': (float(np.min(inter_mi)), float(np.max(inter_mi))),
    }


def run_robustness_check():
    """Run experiment with multiple parameter configurations."""
    configs = [
        {'n_samples': 300, 'n_hidden': 5, 'seed': 42},
        {'n_samples': 300, 'n_hidden': 8, 'seed': 123},
        {'n_samples': 300, 'n_hidden': 3, 'seed': 789},
    ]

    all_results = []
    for i, cfg in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)}: {cfg} ---")
        results = run_experiment(**cfg)
        analysis = analyze_results(results)
        all_results.append({
            'config': cfg,
            'correlation': analysis['correlation_inter_layer'],
            'p_value': analysis['p_value_inter_layer'],
            'cohens_d': analysis['cohens_d'],
        })
        print(f"  r = {analysis['correlation_inter_layer']:.4f}, p = {analysis['p_value_inter_layer']:.2e}, d = {analysis['cohens_d']:.3f}")

    return all_results


if __name__ == '__main__':
    print("=" * 60)
    print("RES-087: Inter-Layer Mutual Information in CPPNs")
    print("=" * 60)

    # Run robustness check across configurations
    all_results = run_robustness_check()

    # Aggregate statistics
    correlations = [r['correlation'] for r in all_results]
    p_values = [r['p_value'] for r in all_results]
    effect_sizes = [r['cohens_d'] for r in all_results]

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS ACROSS CONFIGURATIONS")
    print("=" * 60)
    print(f"Mean correlation: {np.mean(correlations):.4f} (+/- {np.std(correlations):.4f})")
    print(f"P-values range: {min(p_values):.2e} to {max(p_values):.2e}")
    print(f"Mean Cohen's d: {np.mean(effect_sizes):.4f} (+/- {np.std(effect_sizes):.4f})")

    # Determine result based on consistency
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    consistent_direction = all(c < 0 for c in correlations) or all(c > 0 for c in correlations)
    all_significant = all(p < 0.05 for p in p_values)
    strong_effect = np.mean(np.abs(effect_sizes)) > 0.5

    if consistent_direction and all_significant and strong_effect:
        if np.mean(correlations) > 0:
            print("VALIDATED: High-order CPPNs consistently show higher inter-layer MI")
            status = "validated"
        else:
            print("REFUTED: High-order CPPNs consistently show LOWER inter-layer MI")
            status = "refuted"
    elif consistent_direction and all_significant:
        direction = "positive" if np.mean(correlations) > 0 else "negative"
        print(f"REFUTED: Consistent {direction} but weak effect (|d| < 0.5)")
        status = "refuted"
    elif not consistent_direction:
        print("REFUTED: Inconsistent direction across configurations")
        status = "refuted"
    else:
        print("INCONCLUSIVE: Results not statistically robust")
        status = "inconclusive"

    print(f"\nFinal Status: {status.upper()}")
    print(f"Mean Effect size: {np.mean(effect_sizes):.3f}")
    print(f"Effect consistent negative: {all(c < 0 for c in correlations)}")
