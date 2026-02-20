"""
RES-138: Spatial variance of hidden activations correlates with CPPN output order

Hypothesis: High-order CPPNs have higher spatial variance in hidden layer activations
across pixels, indicating more heterogeneous intermediate representations that enable
complex output patterns.

Approach: Generate N=500 CPPNs with hidden layers, compute:
1. Spatial variance of activations at each pixel for each hidden node
2. Aggregate variance across layers
3. Correlate with output order
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, Node, Connection, ACTIVATIONS, order_multiplicative
from scipy import stats
import json
from pathlib import Path

np.random.seed(42)

def create_cppn_with_hidden(n_hidden=4, depth=2):
    """Create CPPN with hidden layers."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    node_id = 4
    prev_layer = [0, 1, 2, 3]  # input ids

    for layer in range(depth):
        current_layer = []
        for _ in range(n_hidden):
            act = np.random.choice(['tanh', 'sin', 'sigmoid', 'gauss'])
            bias = np.random.randn()
            nodes.append(Node(node_id, act, bias))
            # Connect from all previous layer nodes
            for prev_id in prev_layer:
                connections.append(Connection(prev_id, node_id, np.random.randn()))
            current_layer.append(node_id)
            node_id += 1
        prev_layer = current_layer

    # Output node
    nodes.append(Node(node_id, 'sigmoid', np.random.randn()))
    for prev_id in prev_layer:
        connections.append(Connection(prev_id, node_id, np.random.randn()))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0,1,2,3], output_id=node_id)
    return cppn

def compute_hidden_activations(cppn, size=32):
    """Get activations at each hidden node for all pixels."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    hidden_activations = {}

    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[nid] = ACTIVATIONS[node.activation](total)

        if nid != cppn.output_id and nid not in cppn.input_ids:
            hidden_activations[nid] = values[nid]

    return hidden_activations

def compute_spatial_variance_metrics(hidden_activations):
    """Compute various spatial variance metrics."""
    variances = []
    ranges = []

    for nid, acts in hidden_activations.items():
        # Spatial variance across pixels
        variances.append(np.var(acts))
        # Range of activations
        ranges.append(np.ptp(acts))

    return {
        'mean_spatial_variance': np.mean(variances) if variances else 0,
        'max_spatial_variance': np.max(variances) if variances else 0,
        'mean_range': np.mean(ranges) if ranges else 0,
        'total_variance': np.sum(variances) if variances else 0,
    }

def main():
    N = 500
    results = []

    print(f"Generating {N} CPPNs with hidden layers...")

    for i in range(N):
        cppn = create_cppn_with_hidden(n_hidden=4, depth=2)
        img = cppn.render(32)
        order = order_multiplicative(img)

        hidden_acts = compute_hidden_activations(cppn, 32)
        variance_metrics = compute_spatial_variance_metrics(hidden_acts)

        results.append({
            'order': order,
            **variance_metrics
        })

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N} done")

    # Extract arrays
    orders = np.array([r['order'] for r in results])
    mean_vars = np.array([r['mean_spatial_variance'] for r in results])
    max_vars = np.array([r['max_spatial_variance'] for r in results])
    mean_ranges = np.array([r['mean_range'] for r in results])
    total_vars = np.array([r['total_variance'] for r in results])

    # Statistical tests
    r_mean, p_mean = stats.spearmanr(orders, mean_vars)
    r_max, p_max = stats.spearmanr(orders, max_vars)
    r_range, p_range = stats.spearmanr(orders, mean_ranges)
    r_total, p_total = stats.spearmanr(orders, total_vars)

    # Cohen's d for high vs low order
    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = orders <= median_order

    high_var = mean_vars[high_mask]
    low_var = mean_vars[low_mask]
    pooled_std = np.sqrt((np.var(high_var) + np.var(low_var)) / 2)
    cohens_d = (np.mean(high_var) - np.mean(low_var)) / (pooled_std + 1e-10)

    print("\n" + "="*60)
    print("RESULTS: Spatial Variance of Hidden Activations")
    print("="*60)

    print(f"\nCorrelations with order:")
    print(f"  Mean spatial variance: r={r_mean:.3f}, p={p_mean:.2e}")
    print(f"  Max spatial variance:  r={r_max:.3f}, p={p_max:.2e}")
    print(f"  Mean activation range: r={r_range:.3f}, p={p_range:.2e}")
    print(f"  Total variance:        r={r_total:.3f}, p={p_total:.2e}")

    print(f"\nHigh vs Low order comparison (mean spatial variance):")
    print(f"  High-order mean: {np.mean(high_var):.4f}")
    print(f"  Low-order mean:  {np.mean(low_var):.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Determine validation status
    if abs(r_mean) > 0.3 and p_mean < 0.01 and abs(cohens_d) > 0.5:
        if r_mean > 0:
            status = "VALIDATED"
        else:
            status = "REFUTED (negative correlation)"
    elif p_mean < 0.01:
        status = "INCONCLUSIVE (significant but small effect)"
    else:
        status = "REFUTED"

    print(f"\n{'='*60}")
    print(f"STATUS: {status}")
    print(f"Primary effect: r={r_mean:.3f}, d={cohens_d:.3f}, p={p_mean:.2e}")
    print(f"{'='*60}")

    # Save results
    output = {
        'n_samples': N,
        'correlations': {
            'mean_spatial_variance': {'r': float(r_mean), 'p': float(p_mean)},
            'max_spatial_variance': {'r': float(r_max), 'p': float(p_max)},
            'mean_range': {'r': float(r_range), 'p': float(p_range)},
            'total_variance': {'r': float(r_total), 'p': float(p_total)},
        },
        'cohens_d': float(cohens_d),
        'high_order_mean_var': float(np.mean(high_var)),
        'low_order_mean_var': float(np.mean(low_var)),
        'status': status
    }

    Path('/Users/matt/Development/monochrome_noise_converger/results').mkdir(exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/res138_spatial_activation_variance.json', 'w') as f:
        json.dump(output, f, indent=2)

    return output

if __name__ == '__main__':
    main()
