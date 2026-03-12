"""
RES-159: Input-layer weight variance predicts output-layer weight variance in high-order CPPNs

Hypothesis: In high-order CPPNs, there's a correlation between input-layer weight
variance and output-layer weight variance that doesn't exist in low-order CPPNs.

This explores whether different weight groups are functionally coupled.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, PRIOR_SIGMA
from scipy import stats

def create_deep_cppn(depth=2):
    """Create a CPPN with hidden layers to have distinct weight groups."""
    cppn = CPPN()
    cppn.nodes = []
    cppn.connections = []

    # Input nodes (0-3)
    from core.thermo_sampler_v3 import Node, Connection
    for i in range(4):
        cppn.nodes.append(Node(i, 'identity', 0.0))

    # Hidden layers
    node_id = 5
    prev_layer = [0, 1, 2, 3]
    hidden_layers = []

    for d in range(depth):
        layer_nodes = []
        for _ in range(4):  # 4 hidden nodes per layer
            activation = np.random.choice(['tanh', 'sin', 'gauss'])
            cppn.nodes.append(Node(node_id, activation, np.random.randn() * PRIOR_SIGMA))
            layer_nodes.append(node_id)
            # Connect from previous layer
            for prev_id in prev_layer:
                cppn.connections.append(Connection(prev_id, node_id, np.random.randn() * PRIOR_SIGMA))
            node_id += 1
        hidden_layers.append(layer_nodes)
        prev_layer = layer_nodes

    # Output node (id=4)
    cppn.nodes.append(Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA))
    for prev_id in prev_layer:
        cppn.connections.append(Connection(prev_id, 4, np.random.randn() * PRIOR_SIGMA))

    cppn.output_id = 4
    return cppn, hidden_layers

def get_layer_weights(cppn, hidden_layers):
    """Extract weights by layer group."""
    # Input layer connections (from inputs 0-3)
    input_weights = [c.weight for c in cppn.connections
                     if c.from_id in [0, 1, 2, 3] and c.enabled]

    # Output layer connections (to output 4)
    output_weights = [c.weight for c in cppn.connections
                      if c.to_id == 4 and c.enabled]

    # Hidden layer internal connections
    hidden_weights = [c.weight for c in cppn.connections
                      if c.from_id not in [0, 1, 2, 3] and c.to_id != 4 and c.enabled]

    return input_weights, hidden_weights, output_weights

def main():
    np.random.seed(42)
    n_samples = 500

    # Generate CPPNs with various orders
    input_vars = []
    output_vars = []
    hidden_vars = []
    orders = []

    print("Generating CPPNs and computing weight variances...")
    for i in range(n_samples):
        cppn, hidden_layers = create_deep_cppn(depth=2)
        img = cppn.render(32)
        order = order_multiplicative(img)

        input_w, hidden_w, output_w = get_layer_weights(cppn, hidden_layers)

        input_vars.append(np.var(input_w))
        output_vars.append(np.var(output_w))
        hidden_vars.append(np.var(hidden_w) if hidden_w else 0)
        orders.append(order)

    input_vars = np.array(input_vars)
    output_vars = np.array(output_vars)
    hidden_vars = np.array(hidden_vars)
    orders = np.array(orders)

    # Split by order
    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = ~high_mask

    print(f"\nOrder distribution: median={median_order:.4f}, max={orders.max():.4f}")
    print(f"High-order: n={high_mask.sum()}, Low-order: n={low_mask.sum()}")

    # Correlation between input and output weight variances
    r_all, p_all = stats.spearmanr(input_vars, output_vars)
    r_high, p_high = stats.spearmanr(input_vars[high_mask], output_vars[high_mask])
    r_low, p_low = stats.spearmanr(input_vars[low_mask], output_vars[low_mask])

    print(f"\n=== Input-Output Weight Variance Correlation ===")
    print(f"All CPPNs: rho={r_all:.3f}, p={p_all:.4f}")
    print(f"High-order: rho={r_high:.3f}, p={p_high:.4f}")
    print(f"Low-order: rho={r_low:.3f}, p={p_low:.4f}")

    # Test if correlations differ
    from scipy.stats import fisher_exact

    # Fisher z-transform to compare correlations
    z_high = np.arctanh(r_high)
    z_low = np.arctanh(r_low)
    n_high = high_mask.sum()
    n_low = low_mask.sum()
    se_diff = np.sqrt(1/(n_high-3) + 1/(n_low-3))
    z_diff = (z_high - z_low) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    print(f"\nCorrelation difference test: z={z_diff:.3f}, p={p_diff:.4f}")

    # Effect size (Cohen's q for correlation difference)
    cohens_q = abs(z_high - z_low)
    print(f"Effect size (Cohen's q): {cohens_q:.3f}")

    # Does input variance predict order?
    r_input_order, p_input_order = stats.spearmanr(input_vars, orders)
    r_output_order, p_output_order = stats.spearmanr(output_vars, orders)

    print(f"\n=== Weight Variance vs Order ===")
    print(f"Input variance vs order: rho={r_input_order:.3f}, p={p_input_order:.4f}")
    print(f"Output variance vs order: rho={r_output_order:.3f}, p={p_output_order:.4f}")

    # Does hidden-to-output correlation differ?
    r_hidden_output_high, _ = stats.spearmanr(hidden_vars[high_mask], output_vars[high_mask])
    r_hidden_output_low, _ = stats.spearmanr(hidden_vars[low_mask], output_vars[low_mask])

    print(f"\n=== Hidden-Output Variance Correlation ===")
    print(f"High-order: rho={r_hidden_output_high:.3f}")
    print(f"Low-order: rho={r_hidden_output_low:.3f}")

    # Mean variances
    print(f"\n=== Mean Weight Variances ===")
    print(f"Input - High: {input_vars[high_mask].mean():.4f}, Low: {input_vars[low_mask].mean():.4f}")
    print(f"Output - High: {output_vars[high_mask].mean():.4f}, Low: {output_vars[low_mask].mean():.4f}")

    # Cohen's d for variance differences
    d_input = (input_vars[high_mask].mean() - input_vars[low_mask].mean()) / np.sqrt(
        (input_vars[high_mask].var() + input_vars[low_mask].var()) / 2)
    d_output = (output_vars[high_mask].mean() - output_vars[low_mask].mean()) / np.sqrt(
        (output_vars[high_mask].var() + output_vars[low_mask].var()) / 2)

    print(f"\n=== Cohen's d (High vs Low Order) ===")
    print(f"Input variance: d={d_input:.3f}")
    print(f"Output variance: d={d_output:.3f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if abs(cohens_q) > 0.1 and p_diff < 0.05:
        print("VALIDATED: Input-output weight variance correlation differs between high/low order")
    elif p_all < 0.01 and abs(r_all) > 0.3:
        print("PARTIAL: General correlation exists but no high/low order difference")
    else:
        print("REFUTED: No meaningful correlation between weight group variances")

    print(f"\nKey metrics:")
    print(f"  - Overall input-output correlation: r={r_all:.3f}")
    print(f"  - High-order correlation: r={r_high:.3f}")
    print(f"  - Low-order correlation: r={r_low:.3f}")
    print(f"  - Correlation difference: q={cohens_q:.3f}, p={p_diff:.4f}")

if __name__ == "__main__":
    main()
