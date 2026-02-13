"""
RES-203: Activation sign flip rate through layers inversely correlates with CPPN output order

Hypothesis: CPPNs where activations maintain consistent sign through layers produce higher
order outputs than those with frequent sign flips. Stable signal propagation (low flip rate)
should correlate with coherent structure.

Metric: For each spatial position, count how often the pre-activation sign changes between
consecutive hidden layers, averaged across all positions.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, Node, Connection, ACTIVATIONS, order_multiplicative
from scipy import stats

np.random.seed(42)


def create_cppn_with_hidden(n_hidden=4, depth=3):
    """Create CPPN with hidden layers."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    node_id = 4
    prev_layer = [0, 1, 2, 3]  # input ids
    layer_ids = []

    for layer in range(depth):
        current_layer = []
        for _ in range(n_hidden):
            act = np.random.choice(['tanh', 'sin', 'sigmoid', 'gauss'])
            bias = np.random.randn()
            nodes.append(Node(node_id, act, bias))
            for prev_id in prev_layer:
                connections.append(Connection(prev_id, node_id, np.random.randn()))
            current_layer.append(node_id)
            node_id += 1
        prev_layer = current_layer
        layer_ids.append(current_layer)

    # Output node
    nodes.append(Node(node_id, 'sigmoid', np.random.randn()))
    for prev_id in prev_layer:
        connections.append(Connection(prev_id, node_id, np.random.randn()))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0,1,2,3], output_id=node_id)
    return cppn, layer_ids


def compute_layer_preactivations(cppn, layer_ids, size=32):
    """
    Compute pre-activation values for each hidden layer.
    Returns dict mapping layer_index -> array of shape (n_nodes, size, size).
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    layer_preacts = {}

    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        # Store pre-activation for hidden nodes
        for layer_idx, layer in enumerate(layer_ids):
            if nid in layer:
                if layer_idx not in layer_preacts:
                    layer_preacts[layer_idx] = []
                layer_preacts[layer_idx].append(total.copy())

        values[nid] = ACTIVATIONS[node.activation](total)

    # Stack into arrays
    for layer_idx in layer_preacts:
        layer_preacts[layer_idx] = np.array(layer_preacts[layer_idx])

    return layer_preacts


def compute_sign_flip_rate(layer_preacts):
    """
    Compute rate of sign flips between consecutive layers.
    For each pixel position, count sign changes across layers.
    """
    layer_indices = sorted(layer_preacts.keys())
    if len(layer_indices) < 2:
        return 0.0

    total_flips = 0
    total_transitions = 0

    for i in range(len(layer_indices) - 1):
        curr_layer = layer_indices[i]
        next_layer = layer_indices[i + 1]

        # Average pre-activation across nodes in each layer
        curr_avg = layer_preacts[curr_layer].mean(axis=0)
        next_avg = layer_preacts[next_layer].mean(axis=0)

        # Count sign flips
        curr_signs = np.sign(curr_avg)
        next_signs = np.sign(next_avg)

        # Only count where both are non-zero
        valid_mask = (curr_signs != 0) & (next_signs != 0)
        flips = np.sum((curr_signs != next_signs) & valid_mask)
        total_flips += flips
        total_transitions += np.sum(valid_mask)

    if total_transitions == 0:
        return 0.0

    return total_flips / total_transitions


def main():
    n_samples = 500
    orders = []
    flip_rates = []

    print(f"Generating {n_samples} CPPNs with 3 hidden layers (4 nodes each)...")

    for i in range(n_samples):
        cppn, layer_ids = create_cppn_with_hidden(n_hidden=4, depth=3)
        img = cppn.render(32)
        order = order_multiplicative(img)
        layer_preacts = compute_layer_preactivations(cppn, layer_ids, 32)
        flip_rate = compute_sign_flip_rate(layer_preacts)

        orders.append(order)
        flip_rates.append(flip_rate)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples}")

    orders = np.array(orders)
    flip_rates = np.array(flip_rates)

    # Check for constant array
    if np.std(flip_rates) < 1e-10:
        print("\nERROR: Flip rates are constant - cannot compute correlation")
        print(f"  Flip rate mean: {flip_rates.mean():.6f}")
        print(f"  Flip rate std: {flip_rates.std():.6f}")
        return

    # Correlation analysis
    r, p = stats.pearsonr(flip_rates, orders)
    rho, p_spearman = stats.spearmanr(flip_rates, orders)

    # Split into high/low order quartiles
    low_thresh = np.percentile(orders, 25)
    high_thresh = np.percentile(orders, 75)

    low_mask = orders <= low_thresh
    high_mask = orders >= high_thresh

    low_flip = flip_rates[low_mask]
    high_flip = flip_rates[high_mask]

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(low_flip)-1)*np.std(low_flip)**2 +
                          (len(high_flip)-1)*np.std(high_flip)**2) /
                         (len(low_flip) + len(high_flip) - 2))
    cohens_d = (low_flip.mean() - high_flip.mean()) / (pooled_std + 1e-10)

    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(low_flip, high_flip, alternative='two-sided')

    print("\n" + "="*60)
    print("RESULTS: Activation Sign Flip Rate vs Order")
    print("="*60)
    print(f"\nCorrelations:")
    print(f"  Pearson r = {r:.4f} (p = {p:.2e})")
    print(f"  Spearman rho = {rho:.4f} (p = {p_spearman:.2e})")

    print(f"\nQuartile comparison:")
    print(f"  Low-order flip rate:  mean = {low_flip.mean():.4f} (std = {low_flip.std():.4f})")
    print(f"  High-order flip rate: mean = {high_flip.mean():.4f} (std = {high_flip.std():.4f})")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  Mann-Whitney p = {mw_p:.2e}")

    print(f"\nDescriptive stats:")
    print(f"  Flip rate range: [{flip_rates.min():.4f}, {flip_rates.max():.4f}]")
    print(f"  Flip rate mean: {flip_rates.mean():.4f}")
    print(f"  Order range: [{orders.min():.4f}, {orders.max():.4f}]")

    # Hypothesis evaluation
    print("\n" + "="*60)
    print("HYPOTHESIS EVALUATION")
    print("="*60)

    sig_threshold = 0.01
    effect_threshold = 0.5

    is_significant = p < sig_threshold
    has_effect = abs(cohens_d) >= effect_threshold
    correct_direction = r < 0  # Expect negative correlation (inverse)

    print(f"  Significant (p < {sig_threshold}): {is_significant}")
    print(f"  Effect size (|d| >= {effect_threshold}): {has_effect}")
    print(f"  Correct direction (r < 0): {correct_direction}")

    if is_significant and has_effect and correct_direction:
        status = "validated"
        print(f"\nSTATUS: VALIDATED - Strong inverse correlation")
    elif is_significant and has_effect:
        status = "refuted"
        print(f"\nSTATUS: REFUTED - Significant effect but wrong direction")
    else:
        status = "refuted"
        print(f"\nSTATUS: REFUTED - No significant effect or weak effect")

    # Output for log
    print("\n" + "="*60)
    print("LOG ENTRY SUMMARY")
    print("="*60)
    print(f"Status: {status}")
    print(f"r = {r:.3f}, rho = {rho:.3f}, p = {p:.2e}, d = {cohens_d:.2f}")

    return {
        'r': r,
        'rho': rho,
        'p': p,
        'cohens_d': cohens_d,
        'status': status,
        'low_mean': low_flip.mean(),
        'high_mean': high_flip.mean()
    }


if __name__ == '__main__':
    main()
