"""
RES-190: Higher within-layer activation decorrelation predicts CPPN order

Hypothesis: High-order CPPNs produce more decorrelated (independent) representations
within hidden layers compared to low-order CPPNs. This relates to representational
disentanglement - diverse independent features may enable more structured outputs.

Metric: Mean absolute pairwise correlation between hidden neuron activations
across spatial locations (lower = more decorrelated).

Builds on RES-138 (spatial variance correlates with order) and RES-087 (inter-layer MI).
"""

import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

import numpy as np
from scipy import stats
from core.thermo_sampler_v3 import CPPN, order_multiplicative, ACTIVATIONS

def create_cppn_with_hidden(n_hidden=4):
    """Create CPPN with hidden layer for activation analysis."""
    cppn = CPPN()
    # Add hidden nodes
    hidden_ids = []
    for i in range(n_hidden):
        node_id = 5 + i
        hidden_ids.append(node_id)
        from core.thermo_sampler_v3 import Node, Connection
        cppn.nodes.append(Node(node_id, 'tanh', np.random.randn()))
        # Connect all inputs to hidden
        for inp in cppn.input_ids:
            cppn.connections.append(Connection(inp, node_id, np.random.randn()))
        # Connect hidden to output
        cppn.connections.append(Connection(node_id, cppn.output_id, np.random.randn()))
    # Remove direct input->output connections
    cppn.connections = [c for c in cppn.connections if c.to_id != cppn.output_id or c.from_id >= 5]
    return cppn, hidden_ids

def get_hidden_activations(cppn, hidden_ids, size=32):
    """Get activations for each hidden neuron across spatial locations."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}

    activations = []
    for node_id in hidden_ids:
        node = next(n for n in cppn.nodes if n.id == node_id)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == node_id and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        act = ACTIVATIONS[node.activation](total)
        activations.append(act.flatten())

    return np.array(activations)  # Shape: (n_hidden, size*size)

def compute_mean_abs_correlation(activations):
    """Compute mean absolute pairwise correlation between hidden neurons."""
    n_hidden = activations.shape[0]
    if n_hidden < 2:
        return 0.0

    corr_matrix = np.corrcoef(activations)
    # Get upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n_hidden, k=1)]
    return np.mean(np.abs(upper_tri))

def main():
    np.random.seed(42)
    n_samples = 1000
    n_hidden = 5

    orders = []
    decorrelations = []  # Lower correlation = more decorrelated
    mean_correlations = []

    print(f"Sampling {n_samples} CPPNs with {n_hidden} hidden neurons...")

    for i in range(n_samples):
        cppn, hidden_ids = create_cppn_with_hidden(n_hidden)
        img = cppn.render(32)
        order = order_multiplicative(img)

        activations = get_hidden_activations(cppn, hidden_ids)
        mean_corr = compute_mean_abs_correlation(activations)

        orders.append(order)
        mean_correlations.append(mean_corr)
        decorrelations.append(1 - mean_corr)  # Invert so higher = more decorrelated

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples}")

    orders = np.array(orders)
    mean_correlations = np.array(mean_correlations)
    decorrelations = np.array(decorrelations)

    # Core analysis: correlation between decorrelation and order
    r_decorr, p_decorr = stats.pearsonr(decorrelations, orders)
    rho_decorr, p_rho = stats.spearmanr(decorrelations, orders)

    # Cohen's d: compare high vs low order quartiles
    high_order_mask = orders > np.percentile(orders, 75)
    low_order_mask = orders < np.percentile(orders, 25)

    high_decorr = decorrelations[high_order_mask]
    low_decorr = decorrelations[low_order_mask]

    pooled_std = np.sqrt((np.var(high_decorr) + np.var(low_decorr)) / 2)
    cohen_d = (np.mean(high_decorr) - np.mean(low_decorr)) / (pooled_std + 1e-10)

    # t-test
    t_stat, p_ttest = stats.ttest_ind(high_decorr, low_decorr)

    print("\n" + "="*60)
    print("RESULTS: Within-layer Activation Decorrelation vs Order")
    print("="*60)

    print(f"\nCorrelation with Order:")
    print(f"  Pearson r (decorrelation, order):  {r_decorr:.4f} (p={p_decorr:.2e})")
    print(f"  Spearman rho:                      {rho_decorr:.4f} (p={p_rho:.2e})")

    print(f"\nQuartile Comparison:")
    print(f"  High-order mean decorrelation: {np.mean(high_decorr):.4f} (std={np.std(high_decorr):.4f})")
    print(f"  Low-order mean decorrelation:  {np.mean(low_decorr):.4f} (std={np.std(low_decorr):.4f})")
    print(f"  Cohen's d: {cohen_d:.2f}")
    print(f"  t-test p-value: {p_ttest:.2e}")

    print(f"\nDescriptives:")
    print(f"  Order range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"  Mean correlation range: [{mean_correlations.min():.4f}, {mean_correlations.max():.4f}]")
    print(f"  High-order count: {high_order_mask.sum()}, Low-order count: {low_order_mask.sum()}")

    # Validation criteria
    validated = abs(cohen_d) > 0.5 and p_ttest < 0.01

    print("\n" + "="*60)
    print("VALIDATION:")
    print(f"  Effect size |d| > 0.5: {'PASS' if abs(cohen_d) > 0.5 else 'FAIL'} (d={cohen_d:.2f})")
    print(f"  p < 0.01:              {'PASS' if p_ttest < 0.01 else 'FAIL'} (p={p_ttest:.2e})")
    print(f"  STATUS: {'VALIDATED' if validated else 'REFUTED'}")
    print("="*60)

    # Additional analysis: partial correlation controlling for spatial variance
    # Get spatial variance for each CPPN
    spatial_vars = []
    np.random.seed(42)
    for i in range(n_samples):
        cppn, hidden_ids = create_cppn_with_hidden(n_hidden)
        activations = get_hidden_activations(cppn, hidden_ids)
        spatial_var = np.mean(np.var(activations, axis=1))
        spatial_vars.append(spatial_var)

    spatial_vars = np.array(spatial_vars)

    # Partial correlation: decorrelation with order, controlling for spatial variance
    from scipy.stats import pearsonr
    # Residualize decorrelation on spatial variance
    slope, intercept = np.polyfit(spatial_vars, decorrelations, 1)
    decorr_resid = decorrelations - (slope * spatial_vars + intercept)
    r_partial, p_partial = pearsonr(decorr_resid, orders)

    print(f"\nPartial correlation (controlling for spatial variance):")
    print(f"  r = {r_partial:.4f}, p = {p_partial:.2e}")

    return {
        'pearson_r': r_decorr,
        'p_value': p_decorr,
        'cohen_d': cohen_d,
        'p_ttest': p_ttest,
        'validated': validated
    }

if __name__ == "__main__":
    main()
