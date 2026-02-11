"""
RES-158: Pre-activation velocity (layer-to-layer delta) predicts CPPN output order

Hypothesis: The rate of change of pre-activation values through CPPN layers
(velocity = |pre_act[l] - pre_act[l-1]|) correlates with output order.

Rationale: High-order images may require rapid transformation of signals, or
conversely stable propagation. This metric captures dynamics not tested by
saturation (RES-130), variance (RES-138), or gradient flow (RES-131).
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, ACTIVATIONS
from scipy import stats
import json

def make_deep_cppn(depth=4, hidden_per_layer=3):
    """Create CPPN with specified depth for layer-wise analysis."""
    from core.thermo_sampler_v3 import Node, Connection, PRIOR_SIGMA

    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
    ]
    connections = []

    next_id = 5
    prev_layer_ids = [0, 1, 2, 3]

    activations = ['tanh', 'sin', 'sigmoid', 'gauss']

    for layer in range(depth):
        layer_ids = []
        for h in range(hidden_per_layer):
            act = activations[(layer + h) % len(activations)]
            nodes.append(Node(next_id, act, np.random.randn() * PRIOR_SIGMA))
            for prev_id in prev_layer_ids:
                connections.append(Connection(prev_id, next_id, np.random.randn() * PRIOR_SIGMA))
            layer_ids.append(next_id)
            next_id += 1
        prev_layer_ids = layer_ids

    # Output node
    output_id = next_id
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))
    for prev_id in prev_layer_ids:
        connections.append(Connection(prev_id, output_id, np.random.randn() * PRIOR_SIGMA))

    cppn = CPPN(
        nodes=nodes,
        connections=connections,
        input_ids=[0, 1, 2, 3],
        output_id=output_id
    )
    return cppn

def get_layer_preactivations(cppn, size=32):
    """Extract pre-activation values at each layer."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    preactivations = []

    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        preactivations.append(total.flatten())
        values[nid] = ACTIVATIONS[node.activation](total)

    return preactivations

def compute_velocity(preactivations):
    """Compute layer-to-layer velocity (mean abs difference)."""
    velocities = []
    for i in range(1, len(preactivations)):
        delta = np.abs(preactivations[i] - preactivations[i-1])
        velocities.append(np.mean(delta))
    return velocities

def run_experiment(n_samples=500, depth=4):
    """Main experiment."""
    np.random.seed(42)

    orders = []
    mean_velocities = []
    max_velocities = []
    velocity_trends = []  # Is velocity increasing or decreasing through layers?

    for i in range(n_samples):
        cppn = make_deep_cppn(depth=depth)
        img = cppn.render(32)
        order = order_multiplicative(img)

        preacts = get_layer_preactivations(cppn, 32)
        velocities = compute_velocity(preacts)

        if len(velocities) > 0:
            orders.append(order)
            mean_velocities.append(np.mean(velocities))
            max_velocities.append(np.max(velocities))
            # Trend: positive = accelerating, negative = decelerating
            if len(velocities) > 1:
                trend = np.polyfit(range(len(velocities)), velocities, 1)[0]
                velocity_trends.append(trend)
            else:
                velocity_trends.append(0)

    orders = np.array(orders)
    mean_velocities = np.array(mean_velocities)
    max_velocities = np.array(max_velocities)
    velocity_trends = np.array(velocity_trends)

    # Correlations
    r_mean, p_mean = stats.spearmanr(orders, mean_velocities)
    r_max, p_max = stats.spearmanr(orders, max_velocities)
    r_trend, p_trend = stats.spearmanr(orders, velocity_trends)

    # Cohen's d for high vs low order groups
    median_order = np.median(orders)
    high_idx = orders > median_order
    low_idx = orders <= median_order

    high_vel = mean_velocities[high_idx]
    low_vel = mean_velocities[low_idx]

    pooled_std = np.sqrt((np.var(high_vel) + np.var(low_vel)) / 2)
    cohens_d = (np.mean(high_vel) - np.mean(low_vel)) / pooled_std if pooled_std > 0 else 0

    results = {
        'n_samples': n_samples,
        'depth': depth,
        'mean_velocity_corr': float(r_mean),
        'mean_velocity_p': float(p_mean),
        'max_velocity_corr': float(r_max),
        'max_velocity_p': float(p_max),
        'trend_corr': float(r_trend),
        'trend_p': float(p_trend),
        'cohens_d': float(cohens_d),
        'high_order_mean_vel': float(np.mean(high_vel)),
        'low_order_mean_vel': float(np.mean(low_vel)),
        'order_range': [float(orders.min()), float(orders.max())],
        'velocity_range': [float(mean_velocities.min()), float(mean_velocities.max())]
    }

    print("=== RES-158: Pre-activation Velocity Experiment ===")
    print(f"Samples: {n_samples}, Depth: {depth}")
    print(f"\nMean velocity correlation: r={r_mean:.3f}, p={p_mean:.2e}")
    print(f"Max velocity correlation: r={r_max:.3f}, p={p_max:.2e}")
    print(f"Trend correlation: r={r_trend:.3f}, p={p_trend:.2e}")
    print(f"\nCohen's d: {cohens_d:.3f}")
    print(f"High-order mean velocity: {np.mean(high_vel):.3f}")
    print(f"Low-order mean velocity: {np.mean(low_vel):.3f}")

    # Determine outcome
    significant = p_mean < 0.01 and abs(cohens_d) > 0.5
    print(f"\nSignificant (p<0.01, |d|>0.5): {significant}")

    return results

if __name__ == '__main__':
    results = run_experiment(n_samples=500, depth=4)

    import os
    os.makedirs('results/activation_velocity', exist_ok=True)
    with open('results/activation_velocity/res158_results.json', 'w') as f:
        json.dump(results, f, indent=2)
