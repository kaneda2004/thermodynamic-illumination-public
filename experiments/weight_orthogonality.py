"""
RES-086: Weight Orthogonality Analysis

Hypothesis: High-order CPPN weight matrices exhibit greater orthogonality
(lower condition number, more uniform singular values) than low-order CPPNs,
enabling richer representational capacity.

Approach:
1. Sample many CPPNs and compute their order scores
2. For each CPPN, reshape weights into a matrix form and compute:
   - Condition number (ratio of max/min singular values) - lower = more orthogonal
   - Singular value uniformity (std of normalized singular values) - lower = more uniform
3. Compare these metrics between high-order and low-order groups
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed, PRIOR_SIGMA

def create_cppn_with_hidden(n_hidden: int = 2) -> CPPN:
    """Create a CPPN with specified number of hidden nodes for more weight structure."""
    from core.thermo_sampler_v3 import Node, Connection

    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
    ]

    # Hidden nodes with random activations
    activations = ['sin', 'cos', 'gauss', 'tanh', 'sigmoid', 'abs']
    hidden_ids = []
    for i in range(n_hidden):
        node_id = 4 + i
        act = np.random.choice(activations)
        nodes.append(Node(node_id, act, np.random.randn() * PRIOR_SIGMA))
        hidden_ids.append(node_id)

    # Output node
    output_id = 4 + n_hidden
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))

    # Connections: input -> hidden -> output
    connections = []
    input_ids = [0, 1, 2, 3]

    # Input to hidden
    for inp_id in input_ids:
        for hid_id in hidden_ids:
            connections.append(Connection(inp_id, hid_id, np.random.randn() * PRIOR_SIGMA))

    # Hidden to output
    for hid_id in hidden_ids:
        connections.append(Connection(hid_id, output_id, np.random.randn() * PRIOR_SIGMA))

    # Some direct input to output connections
    for inp_id in input_ids:
        if np.random.random() < 0.5:
            connections.append(Connection(inp_id, output_id, np.random.randn() * PRIOR_SIGMA))

    return CPPN(
        nodes=nodes,
        connections=connections,
        input_ids=input_ids,
        output_id=output_id
    )


def compute_weight_matrix_metrics(cppn: CPPN) -> dict:
    """
    Compute orthogonality metrics for CPPN weights.

    Since CPPN weights are in a graph structure, we extract:
    1. Input-to-hidden weight matrix (if hidden nodes exist)
    2. Hidden-to-output weight matrix (if hidden nodes exist)
    3. All weights as a pseudo-matrix (reshape to most square form)
    """
    # Get all weight parameters
    weights = cppn.get_weights()
    n_weights = len(weights)

    if n_weights < 2:
        return None

    # Reshape weights into most-square matrix possible
    # Find factors closest to sqrt(n)
    sqrt_n = int(np.sqrt(n_weights))
    for rows in range(sqrt_n, 0, -1):
        if n_weights % rows == 0:
            cols = n_weights // rows
            break
    else:
        # Pad to make it work
        rows = sqrt_n
        cols = (n_weights + rows - 1) // rows
        padded_weights = np.zeros(rows * cols)
        padded_weights[:n_weights] = weights
        weights = padded_weights

    # Reshape into matrix
    W = weights.reshape(rows, cols)

    # Compute SVD
    try:
        U, s, Vh = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    # Metrics
    s_normalized = s / (np.sum(s) + 1e-10)

    metrics = {
        'condition_number': s[0] / (s[-1] + 1e-10),  # Lower = more orthogonal
        'sv_uniformity': np.std(s_normalized),  # Lower = more uniform singular values
        'sv_ratio_max_mean': s[0] / (np.mean(s) + 1e-10),  # Lower = more balanced
        'effective_rank': np.sum(s > 0.01 * s[0]),  # Higher = more effective dimensions
        'nuclear_norm': np.sum(s),  # Total "magnitude"
        'frobenius_norm': np.sqrt(np.sum(s**2)),
        'spectral_entropy': -np.sum(s_normalized * np.log(s_normalized + 1e-10)) / np.log(len(s) + 1e-10),  # Higher = more uniform
    }

    return metrics


def run_experiment(n_samples: int = 500, n_hidden: int = 3, seed: int = 42):
    """
    Generate many CPPNs, compute order and orthogonality metrics,
    and test correlation.
    """
    set_global_seed(seed)

    orders = []
    all_metrics = []

    print(f"Generating {n_samples} CPPNs with {n_hidden} hidden nodes...")

    for i in range(n_samples):
        # Create CPPN with hidden layers for more weight structure
        cppn = create_cppn_with_hidden(n_hidden)

        # Compute order
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute weight metrics
        metrics = compute_weight_matrix_metrics(cppn)

        if metrics is not None:
            orders.append(order)
            all_metrics.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    orders = np.array(orders)
    n_valid = len(orders)
    print(f"Valid samples: {n_valid}")

    # Split into high/low order groups
    median_order = np.median(orders)
    high_mask = orders > np.percentile(orders, 75)
    low_mask = orders < np.percentile(orders, 25)

    print(f"\nOrder statistics:")
    print(f"  Mean: {np.mean(orders):.4f}")
    print(f"  Std: {np.std(orders):.4f}")
    print(f"  Range: [{np.min(orders):.4f}, {np.max(orders):.4f}]")
    print(f"  High group (>75%): {np.sum(high_mask)} samples")
    print(f"  Low group (<25%): {np.sum(low_mask)} samples")

    # Analyze each metric
    results = {}
    metric_names = list(all_metrics[0].keys())

    print("\n" + "="*70)
    print("ORTHOGONALITY METRICS vs ORDER")
    print("="*70)

    for metric_name in metric_names:
        metric_values = np.array([m[metric_name] for m in all_metrics])

        # Correlation with order
        r, p = stats.pearsonr(orders, metric_values)

        # Group comparison
        high_values = metric_values[high_mask]
        low_values = metric_values[low_mask]

        t_stat, t_p = stats.ttest_ind(high_values, low_values)
        effect_size = (np.mean(high_values) - np.mean(low_values)) / np.sqrt(
            (np.std(high_values)**2 + np.std(low_values)**2) / 2 + 1e-10
        )

        results[metric_name] = {
            'correlation': r,
            'corr_p': p,
            'high_mean': np.mean(high_values),
            'low_mean': np.mean(low_values),
            't_stat': t_stat,
            't_p': t_p,
            'effect_size': effect_size,
        }

        print(f"\n{metric_name}:")
        print(f"  Correlation with order: r={r:.4f}, p={p:.2e}")
        print(f"  High-order mean: {np.mean(high_values):.4f} +/- {np.std(high_values):.4f}")
        print(f"  Low-order mean:  {np.mean(low_values):.4f} +/- {np.std(low_values):.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        print(f"  T-test p-value: {t_p:.2e}")

    # Key hypothesis tests
    print("\n" + "="*70)
    print("HYPOTHESIS TEST SUMMARY")
    print("="*70)

    # H1: Lower condition number in high-order CPPNs (more orthogonal)
    cn_result = results['condition_number']
    print(f"\nH1: High-order CPPNs have LOWER condition number (more orthogonal)")
    print(f"    High-order: {cn_result['high_mean']:.2f}, Low-order: {cn_result['low_mean']:.2f}")
    print(f"    Effect size: {cn_result['effect_size']:.4f}")
    print(f"    p-value: {cn_result['t_p']:.2e}")
    h1_supported = cn_result['high_mean'] < cn_result['low_mean'] and cn_result['t_p'] < 0.01
    print(f"    Supported: {h1_supported}")

    # H2: More uniform singular values in high-order CPPNs
    sv_result = results['sv_uniformity']
    print(f"\nH2: High-order CPPNs have LOWER sv_uniformity (more uniform singular values)")
    print(f"    High-order: {sv_result['high_mean']:.4f}, Low-order: {sv_result['low_mean']:.4f}")
    print(f"    Effect size: {sv_result['effect_size']:.4f}")
    print(f"    p-value: {sv_result['t_p']:.2e}")
    h2_supported = sv_result['high_mean'] < sv_result['low_mean'] and sv_result['t_p'] < 0.01
    print(f"    Supported: {h2_supported}")

    # H3: Higher spectral entropy in high-order CPPNs (more balanced)
    se_result = results['spectral_entropy']
    print(f"\nH3: High-order CPPNs have HIGHER spectral entropy (more balanced)")
    print(f"    High-order: {se_result['high_mean']:.4f}, Low-order: {se_result['low_mean']:.4f}")
    print(f"    Effect size: {se_result['effect_size']:.4f}")
    print(f"    p-value: {se_result['t_p']:.2e}")
    h3_supported = se_result['high_mean'] > se_result['low_mean'] and se_result['t_p'] < 0.01
    print(f"    Supported: {h3_supported}")

    # Overall verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)

    # Find the strongest effect
    strongest_metric = max(results.keys(), key=lambda k: abs(results[k]['effect_size']))
    strongest = results[strongest_metric]

    any_significant = any(r['t_p'] < 0.01 and abs(r['effect_size']) > 0.3 for r in results.values())

    if any_significant:
        if h1_supported or h2_supported or h3_supported:
            status = "VALIDATED"
            summary = f"Orthogonality metrics correlate with order. Strongest: {strongest_metric} (d={strongest['effect_size']:.3f}, p={strongest['t_p']:.2e})"
        else:
            # Effect is opposite to hypothesis
            status = "REFUTED"
            summary = f"Significant effect but OPPOSITE to hypothesis. {strongest_metric}: high-order has {'higher' if strongest['effect_size'] > 0 else 'lower'} values (d={strongest['effect_size']:.3f})"
    else:
        status = "REFUTED"
        summary = f"No significant relationship between weight orthogonality and order. Strongest effect: {strongest_metric} (d={strongest['effect_size']:.3f}, p={strongest['t_p']:.2e})"

    print(f"\nStatus: {status}")
    print(f"Summary: {summary}")

    return {
        'status': status,
        'summary': summary,
        'n_samples': n_valid,
        'results': results,
        'strongest_metric': strongest_metric,
        'strongest_effect': strongest['effect_size'],
        'strongest_p': strongest['t_p'],
    }


if __name__ == '__main__':
    result = run_experiment(n_samples=500, n_hidden=3, seed=42)

    print("\n" + "="*70)
    print("FINAL RESULT FOR LOG")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Effect size: {result['strongest_effect']:.4f}")
    print(f"p-value: {result['strongest_p']:.2e}")
    print(f"Summary: {result['summary']}")
