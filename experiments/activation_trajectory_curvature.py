"""
RES-163: Activation trajectory curvature through CPPN layers correlates with output order

Hypothesis: The curvature (second derivative) of activation values as they propagate
through CPPN layers correlates with output order. High curvature indicates strong
nonlinear transformation, which may enable more structured outputs.

This is distinct from:
- RES-158: Pre-activation velocity (first derivative, no correlation found)
- RES-138: Spatial variance of activations (found correlation)

Curvature = d²a/dl² where a is activation magnitude and l is layer depth.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, ACTIVATIONS, order_multiplicative, PRIOR_SIGMA
from scipy import stats

def create_deep_cppn(n_hidden=3, activation='tanh'):
    """Create a CPPN with hidden layers to measure trajectory through layers."""
    cppn = CPPN.__new__(CPPN)
    cppn.input_ids = [0, 1, 2, 3]
    cppn.output_id = 4 + n_hidden

    # Input nodes (identity activation, no bias)
    cppn.nodes = [
        type('Node', (), {'id': 0, 'activation': 'identity', 'bias': 0.0})(),
        type('Node', (), {'id': 1, 'activation': 'identity', 'bias': 0.0})(),
        type('Node', (), {'id': 2, 'activation': 'identity', 'bias': 0.0})(),
        type('Node', (), {'id': 3, 'activation': 'identity', 'bias': 0.0})(),
    ]

    # Hidden nodes
    for i in range(n_hidden):
        node_id = 4 + i
        cppn.nodes.append(type('Node', (), {
            'id': node_id,
            'activation': activation,
            'bias': np.random.randn() * PRIOR_SIGMA
        })())

    # Output node (sigmoid for binary output)
    cppn.nodes.append(type('Node', (), {
        'id': cppn.output_id,
        'activation': 'sigmoid',
        'bias': np.random.randn() * PRIOR_SIGMA
    })())

    # Connections: input -> hidden1, hidden1 -> hidden2, ..., hiddenN -> output
    cppn.connections = []

    # Input to first hidden
    for inp in cppn.input_ids:
        cppn.connections.append(type('Connection', (), {
            'from_id': inp,
            'to_id': 4,
            'weight': np.random.randn() * PRIOR_SIGMA,
            'enabled': True
        })())

    # Hidden to hidden connections (chain)
    for i in range(n_hidden - 1):
        cppn.connections.append(type('Connection', (), {
            'from_id': 4 + i,
            'to_id': 4 + i + 1,
            'weight': np.random.randn() * PRIOR_SIGMA,
            'enabled': True
        })())

    # Last hidden to output
    cppn.connections.append(type('Connection', (), {
        'from_id': 4 + n_hidden - 1,
        'to_id': cppn.output_id,
        'weight': np.random.randn() * PRIOR_SIGMA,
        'enabled': True
    })())

    return cppn


def get_layer_activations(cppn, size=32):
    """Get activations at each layer for all spatial positions."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    # Get evaluation order
    hidden_ids = sorted([n.id for n in cppn.nodes
                        if n.id not in cppn.input_ids and n.id != cppn.output_id])
    eval_order = hidden_ids + [cppn.output_id]

    values = {0: x, 1: y, 2: r, 3: bias}
    layer_activations = []

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        activated = ACTIVATIONS[node.activation](total)
        values[nid] = activated
        layer_activations.append(activated)

    return layer_activations


def compute_trajectory_curvature(layer_activations):
    """
    Compute curvature of activation trajectory through layers.

    For each spatial position, we have a sequence of activations [a0, a1, a2, ...].
    Curvature is the second derivative: d²a/dl².
    We compute the mean absolute curvature across all positions.
    """
    if len(layer_activations) < 3:
        return 0.0

    # Stack activations: shape (n_layers, height, width)
    stacked = np.array(layer_activations)

    # First derivative: velocity between layers
    velocity = np.diff(stacked, axis=0)  # (n_layers-1, h, w)

    # Second derivative: acceleration/curvature
    acceleration = np.diff(velocity, axis=0)  # (n_layers-2, h, w)

    # Mean absolute curvature across all layers and positions
    mean_abs_curvature = np.mean(np.abs(acceleration))

    # Also compute RMS curvature
    rms_curvature = np.sqrt(np.mean(acceleration ** 2))

    # Variance of curvature (how variable the curvature is)
    curvature_variance = np.var(acceleration)

    return {
        'mean_abs_curvature': float(mean_abs_curvature),
        'rms_curvature': float(rms_curvature),
        'curvature_variance': float(curvature_variance)
    }


def run_experiment(n_samples=500, n_hidden=4):
    """Run the activation trajectory curvature experiment."""
    np.random.seed(42)

    results = []

    for i in range(n_samples):
        cppn = create_deep_cppn(n_hidden=n_hidden, activation='tanh')

        # Get layer activations
        layer_activations = get_layer_activations(cppn, size=32)

        # Compute curvature metrics
        curvature_metrics = compute_trajectory_curvature(layer_activations)

        # Render and compute order
        img = (layer_activations[-1] > 0.5).astype(np.uint8)
        order = order_multiplicative(img)

        results.append({
            'order': order,
            **curvature_metrics
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{n_samples} samples")

    return results


def analyze_results(results):
    """Analyze correlation between curvature and order."""
    orders = np.array([r['order'] for r in results])
    mean_abs_curvatures = np.array([r['mean_abs_curvature'] for r in results])
    rms_curvatures = np.array([r['rms_curvature'] for r in results])
    curvature_variances = np.array([r['curvature_variance'] for r in results])

    # Correlations
    r_abs, p_abs = stats.pearsonr(mean_abs_curvatures, orders)
    r_rms, p_rms = stats.pearsonr(rms_curvatures, orders)
    r_var, p_var = stats.pearsonr(curvature_variances, orders)

    # Split into high/low order for Cohen's d
    median_order = np.median(orders)
    high_order_mask = orders >= median_order
    low_order_mask = orders < median_order

    high_order_curvatures = mean_abs_curvatures[high_order_mask]
    low_order_curvatures = mean_abs_curvatures[low_order_mask]

    # Cohen's d
    pooled_std = np.sqrt((np.var(high_order_curvatures) + np.var(low_order_curvatures)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(high_order_curvatures) - np.mean(low_order_curvatures)) / pooled_std
    else:
        cohens_d = 0.0

    # T-test
    t_stat, t_pvalue = stats.ttest_ind(high_order_curvatures, low_order_curvatures)

    return {
        'correlation_abs': {'r': float(r_abs), 'p': float(p_abs)},
        'correlation_rms': {'r': float(r_rms), 'p': float(p_rms)},
        'correlation_var': {'r': float(r_var), 'p': float(p_var)},
        'cohens_d': float(cohens_d),
        't_test_pvalue': float(t_pvalue),
        'high_order_mean_curvature': float(np.mean(high_order_curvatures)),
        'low_order_mean_curvature': float(np.mean(low_order_curvatures)),
        'order_stats': {
            'mean': float(np.mean(orders)),
            'std': float(np.std(orders)),
            'median': float(median_order)
        },
        'curvature_stats': {
            'mean': float(np.mean(mean_abs_curvatures)),
            'std': float(np.std(mean_abs_curvatures))
        }
    }


if __name__ == '__main__':
    print("RES-163: Activation trajectory curvature experiment")
    print("=" * 60)

    # Run experiment
    results = run_experiment(n_samples=500, n_hidden=4)

    # Analyze
    analysis = analyze_results(results)

    print("\nResults:")
    print(f"Correlation (mean abs curvature vs order): r={analysis['correlation_abs']['r']:.4f}, p={analysis['correlation_abs']['p']:.2e}")
    print(f"Correlation (RMS curvature vs order): r={analysis['correlation_rms']['r']:.4f}, p={analysis['correlation_rms']['p']:.2e}")
    print(f"Correlation (curvature variance vs order): r={analysis['correlation_var']['r']:.4f}, p={analysis['correlation_var']['p']:.2e}")
    print(f"Cohen's d: {analysis['cohens_d']:.4f}")
    print(f"T-test p-value: {analysis['t_test_pvalue']:.2e}")
    print(f"High-order mean curvature: {analysis['high_order_mean_curvature']:.4f}")
    print(f"Low-order mean curvature: {analysis['low_order_mean_curvature']:.4f}")

    # Save results
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/activation_trajectory_curvature', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/activation_trajectory_curvature/results.json', 'w') as f:
        json.dump({
            'experiment_id': 'RES-163',
            'hypothesis': 'Activation trajectory curvature through CPPN layers correlates with output order',
            'n_samples': 500,
            'n_hidden_layers': 4,
            'analysis': analysis
        }, f, indent=2)

    print("\nResults saved to results/activation_trajectory_curvature/results.json")

    # Determine verdict
    r = analysis['correlation_abs']['r']
    p = analysis['correlation_abs']['p']
    d = analysis['cohens_d']

    print("\n" + "=" * 60)
    if p < 0.01 and abs(d) > 0.5:
        print(f"VERDICT: VALIDATED - significant correlation with sufficient effect size")
    elif p < 0.01:
        print(f"VERDICT: REFUTED - significant correlation but effect size d={d:.2f} below 0.5 threshold")
    else:
        print(f"VERDICT: REFUTED - no significant correlation (p={p:.2e})")
