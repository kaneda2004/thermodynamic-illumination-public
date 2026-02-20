"""
RES-124: Gradient Flow and CPPN Image Order

Hypothesis: Gradient magnitude through CPPN layers correlates with image order -
networks with more balanced gradient flow (avoiding vanishing/exploding) produce
higher-order images.

Method:
1. Generate many random CPPN architectures (varying depths)
2. Compute gradient flow statistics using finite differences
3. Measure order of generated images
4. Analyze correlation between gradient flow balance and order
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, Node, Connection, ACTIVATIONS, order_multiplicative

np.random.seed(42)


def create_cppn_with_hidden_layers(n_hidden_layers: int, nodes_per_layer: int = 3) -> CPPN:
    """Create a CPPN with specified number of hidden layers."""
    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
    ]
    connections = []

    input_ids = [0, 1, 2, 3]
    next_id = 5  # Output will be 4

    # Activation functions for hidden nodes
    hidden_activations = ['sin', 'tanh', 'gauss', 'sigmoid', 'cos']

    # Build hidden layers
    prev_layer_ids = input_ids
    all_hidden_ids = []

    for layer_idx in range(n_hidden_layers):
        layer_ids = []
        for node_idx in range(nodes_per_layer):
            node_id = next_id
            next_id += 1
            activation = np.random.choice(hidden_activations)
            bias = np.random.randn()
            nodes.append(Node(node_id, activation, bias))
            layer_ids.append(node_id)

            # Connect from previous layer (or inputs)
            for from_id in prev_layer_ids:
                weight = np.random.randn()
                connections.append(Connection(from_id, node_id, weight))

        all_hidden_ids.extend(layer_ids)
        prev_layer_ids = layer_ids

    # Output node
    output_activation = 'sigmoid'
    nodes.append(Node(4, output_activation, np.random.randn()))

    # Connect final hidden layer (or inputs if no hidden) to output
    for from_id in prev_layer_ids:
        weight = np.random.randn()
        connections.append(Connection(from_id, 4, weight))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)
    return cppn


def compute_gradient_flow_stats(cppn: CPPN, size: int = 32, eps: float = 1e-5) -> dict:
    """
    Compute gradient flow statistics using finite differences.

    Returns:
    - mean_grad: Mean absolute gradient magnitude
    - std_grad: Std of gradient magnitudes
    - grad_ratio: Ratio of max to min gradient (indicator of exploding/vanishing)
    - balanced_flow: Measure of gradient balance (lower is better)
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    weights = cppn.get_weights()
    n_weights = len(weights)

    gradients = []

    # Compute gradient for each weight using finite differences
    base_output = cppn.activate(x, y)

    for i in range(n_weights):
        perturbed_weights = weights.copy()
        perturbed_weights[i] += eps
        cppn.set_weights(perturbed_weights)
        perturbed_output = cppn.activate(x, y)

        # Reset weights
        cppn.set_weights(weights)

        # Gradient magnitude (output change per weight change)
        grad = np.abs(perturbed_output - base_output) / eps
        grad_magnitude = np.mean(grad)
        gradients.append(grad_magnitude)

    gradients = np.array(gradients)
    gradients = gradients[gradients > 1e-10]  # Filter near-zero gradients

    if len(gradients) < 2:
        return {
            'mean_grad': 0,
            'std_grad': 0,
            'grad_ratio': float('inf'),
            'balanced_flow': 0,
            'n_active_weights': 0
        }

    mean_grad = np.mean(gradients)
    std_grad = np.std(gradients)

    # Ratio of max to min gradient (exploding/vanishing indicator)
    grad_ratio = np.max(gradients) / (np.min(gradients) + 1e-10)

    # Balanced flow metric: lower gradient ratio = more balanced
    # Use log to handle large ratios
    balanced_flow = 1 / (1 + np.log1p(grad_ratio))

    return {
        'mean_grad': mean_grad,
        'std_grad': std_grad,
        'grad_ratio': grad_ratio,
        'balanced_flow': balanced_flow,
        'n_active_weights': len(gradients)
    }


def run_experiment(n_samples: int = 200, max_depth: int = 5):
    """Run the gradient flow vs order experiment."""

    results = []

    for _ in range(n_samples):
        # Vary depth
        n_hidden = np.random.randint(0, max_depth + 1)

        # Create CPPN
        cppn = create_cppn_with_hidden_layers(n_hidden)

        # Render image and compute order
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute gradient flow statistics
        grad_stats = compute_gradient_flow_stats(cppn)

        results.append({
            'n_hidden_layers': n_hidden,
            'order': order,
            **grad_stats
        })

    return results


def analyze_results(results: list):
    """Analyze correlation between gradient flow and order."""

    # Filter valid results
    valid = [r for r in results if r['n_active_weights'] > 0 and r['order'] > 0]

    if len(valid) < 10:
        print("ERROR: Not enough valid samples")
        return None

    orders = np.array([r['order'] for r in valid])
    balanced_flow = np.array([r['balanced_flow'] for r in valid])
    mean_grad = np.array([r['mean_grad'] for r in valid])
    grad_ratio = np.array([r['grad_ratio'] for r in valid])

    # Primary analysis: balanced_flow vs order
    corr_balanced, p_balanced = stats.spearmanr(balanced_flow, orders)

    # Secondary: mean_grad vs order
    corr_mean, p_mean = stats.spearmanr(mean_grad, orders)

    # Tertiary: grad_ratio vs order (expect negative correlation)
    corr_ratio, p_ratio = stats.spearmanr(grad_ratio, orders)

    # Effect size (Cohen's d equivalent for correlation)
    # Using Fisher's z transformation
    effect_size = np.abs(corr_balanced) * np.sqrt(len(valid) - 3)

    # Bonferroni correction (3 tests)
    p_corrected = min(1.0, p_balanced * 3)

    # Bin analysis: compare high vs low balanced_flow
    median_bf = np.median(balanced_flow)
    high_bf_order = orders[balanced_flow > median_bf].mean()
    low_bf_order = orders[balanced_flow <= median_bf].mean()

    print("=" * 60)
    print("RES-124: Gradient Flow vs Image Order")
    print("=" * 60)
    print(f"\nSamples: {len(valid)}")
    print(f"\nOrder statistics:")
    print(f"  Mean: {orders.mean():.4f}")
    print(f"  Std:  {orders.std():.4f}")
    print(f"\nCorrelation Analysis (Spearman):")
    print(f"  balanced_flow vs order: r={corr_balanced:.4f}, p={p_balanced:.4e}")
    print(f"  mean_grad vs order:     r={corr_mean:.4f}, p={p_mean:.4e}")
    print(f"  grad_ratio vs order:    r={corr_ratio:.4f}, p={p_ratio:.4e}")
    print(f"\nBonferroni-corrected p-value (primary): {p_corrected:.4e}")
    print(f"Effect size (z-score): {effect_size:.4f}")
    print(f"\nBin Analysis (high vs low balanced_flow):")
    print(f"  High balanced_flow mean order: {high_bf_order:.4f}")
    print(f"  Low balanced_flow mean order:  {low_bf_order:.4f}")
    print(f"  Difference: {high_bf_order - low_bf_order:.4f}")

    # Determine validation status
    significant = p_corrected < 0.01
    effect_large = np.abs(corr_balanced) > 0.2

    if significant and effect_large:
        status = "VALIDATED"
    elif significant or effect_large:
        status = "INCONCLUSIVE"
    else:
        status = "REFUTED"

    print(f"\n{'='*60}")
    print(f"STATUS: {status}")
    print(f"{'='*60}")

    return {
        'status': status,
        'n_samples': len(valid),
        'correlation': corr_balanced,
        'p_value': p_corrected,
        'effect_size': effect_size,
        'high_bf_order': high_bf_order,
        'low_bf_order': low_bf_order
    }


def deeper_analysis(results: list):
    """Further investigate the mean_grad vs order relationship."""
    valid = [r for r in results if r['n_active_weights'] > 0 and r['order'] > 0]

    orders = np.array([r['order'] for r in valid])
    mean_grad = np.array([r['mean_grad'] for r in valid])
    depths = np.array([r['n_hidden_layers'] for r in valid])

    # Control for depth
    print("\n" + "="*60)
    print("DEEPER ANALYSIS: Mean Gradient vs Order")
    print("="*60)

    # Partial correlation controlling for depth
    from scipy.stats import pearsonr, spearmanr

    # Stratify by depth
    print("\nStratified by depth:")
    for d in range(6):
        mask = depths == d
        if mask.sum() > 10:
            corr, p = spearmanr(mean_grad[mask], orders[mask])
            print(f"  Depth {d}: n={mask.sum()}, r={corr:.3f}, p={p:.4f}")

    # High-order images analysis
    high_order_mask = orders > 0.3
    low_order_mask = orders < 0.1

    print(f"\nHigh order images (order > 0.3): n={high_order_mask.sum()}")
    if high_order_mask.sum() > 0:
        print(f"  Mean gradient: {mean_grad[high_order_mask].mean():.4f}")
        print(f"  Mean depth: {depths[high_order_mask].mean():.2f}")

    print(f"\nLow order images (order < 0.1): n={low_order_mask.sum()}")
    if low_order_mask.sum() > 0:
        print(f"  Mean gradient: {mean_grad[low_order_mask].mean():.4f}")
        print(f"  Mean depth: {depths[low_order_mask].mean():.2f}")

    # Effect size for mean_grad
    if high_order_mask.sum() > 5 and low_order_mask.sum() > 5:
        high_grad = mean_grad[high_order_mask]
        low_grad = mean_grad[low_order_mask]
        pooled_std = np.sqrt((high_grad.std()**2 + low_grad.std()**2) / 2)
        cohens_d = (high_grad.mean() - low_grad.mean()) / pooled_std
        print(f"\nCohen's d (high vs low order): {cohens_d:.3f}")

    # Retest mean_grad correlation with more samples
    corr, p = spearmanr(mean_grad, orders)
    print(f"\nFinal mean_grad vs order correlation:")
    print(f"  Spearman r = {corr:.4f}")
    print(f"  p-value = {p:.2e}")
    print(f"  Effect size (|r|) = {abs(corr):.4f}")

    return corr, p


if __name__ == "__main__":
    print("Running gradient flow experiment...")
    results = run_experiment(n_samples=300, max_depth=5)
    analysis = analyze_results(results)
    corr, p = deeper_analysis(results)
