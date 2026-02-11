#!/usr/bin/env python3
"""
RES-120: Noise-Injected Sampling vs Standard Search

HYPOTHESIS: Noise-injected CPPN sampling finds higher-order images faster
than standard mutation-based exploration when searching for images above
a given order threshold.

DOMAIN: noise_injection

Approach: Compare two search strategies:
1. Standard: Gaussian mutation of CPPN weights
2. Noise-injected: Add noise to hidden activations during evaluation

Both start from same random CPPNs and search for images with order > threshold.
Measure: number of function evaluations to find K high-order images.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, PRIOR_SIGMA
from scipy import stats
from dataclasses import dataclass


ACTIVATIONS = {
    'sigmoid': lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
    'tanh': np.tanh,
    'sin': np.sin,
    'relu': lambda z: np.maximum(0, z),
    'identity': lambda z: z,
    'abs': np.abs,
    'step': lambda z: (z > 0).astype(float),
    'inv': lambda z: 1 / (1 + np.abs(z)),
    'gauss': lambda z: np.exp(-z**2)
}


def activate_with_noise(cppn, x, y, noise_scale):
    """Activate CPPN with noise injection into hidden layers."""
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)
    values = {0: x, 1: y, 2: r, 3: bias}

    hidden_ids = sorted([n.id for n in cppn.nodes
                        if n.id not in cppn.input_ids and n.id != cppn.output_id])
    eval_order = hidden_ids + [cppn.output_id]

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        activation_val = ACTIVATIONS[node.activation](total)

        # Add noise to hidden layers only
        if noise_scale > 0 and nid != cppn.output_id:
            activation_val += np.random.randn(*activation_val.shape) * noise_scale

        values[nid] = activation_val

    return values[cppn.output_id]


def render_with_noise(cppn, size=32, noise_scale=0.0):
    """Render with optional noise injection."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return (activate_with_noise(cppn, x, y, noise_scale) > 0.5).astype(np.uint8)


def create_complex_cppn(n_hidden=3):
    """Create a CPPN with hidden nodes for meaningful noise injection."""
    cppn = CPPN()
    activations = ['sigmoid', 'tanh', 'sin', 'relu', 'gauss']

    # Add hidden nodes
    for h in range(n_hidden):
        node_id = 5 + h
        cppn.nodes.append(type(cppn.nodes[0])(
            id=node_id,
            activation=np.random.choice(activations),
            bias=np.random.randn() * 0.5
        ))
        # Connect from inputs
        for inp_id in cppn.input_ids:
            if np.random.rand() < 0.6:
                cppn.connections.append(type(cppn.connections[0])(
                    from_id=inp_id, to_id=node_id,
                    weight=np.random.randn() * PRIOR_SIGMA, enabled=True
                ))
        # Connect to output
        cppn.connections.append(type(cppn.connections[0])(
            from_id=node_id, to_id=cppn.output_id,
            weight=np.random.randn() * PRIOR_SIGMA, enabled=True
        ))
    return cppn


def search_standard(threshold, max_evals=1000, target_finds=5):
    """
    Standard search: mutate weights, accept if above threshold.
    Returns: (n_finds, n_evals, best_order)
    """
    cppn = create_complex_cppn()
    best_order = 0.0
    n_finds = 0
    found_orders = []

    for i in range(max_evals):
        # Render and evaluate
        img = cppn.render(32)
        order = order_multiplicative(img)

        if order > best_order:
            best_order = order

        if order >= threshold:
            n_finds += 1
            found_orders.append(order)
            if n_finds >= target_finds:
                return n_finds, i + 1, best_order

        # Mutate weights
        weights = cppn.get_weights()
        weights += np.random.randn(len(weights)) * 0.3
        cppn.set_weights(weights)

    return n_finds, max_evals, best_order


def search_noise_injected(threshold, max_evals=1000, target_finds=5, noise_scale=0.3):
    """
    Noise-injected search: keep weights fixed, use noise to explore.
    Each evaluation has different noise realization.
    Returns: (n_finds, n_evals, best_order)
    """
    cppn = create_complex_cppn()
    best_order = 0.0
    n_finds = 0
    found_orders = []

    for i in range(max_evals):
        # Render with noise injection
        img = render_with_noise(cppn, size=32, noise_scale=noise_scale)
        order = order_multiplicative(img)

        if order > best_order:
            best_order = order

        if order >= threshold:
            n_finds += 1
            found_orders.append(order)
            if n_finds >= target_finds:
                return n_finds, i + 1, best_order

        # Occasionally resample the base CPPN for long-range exploration
        if (i + 1) % 50 == 0:
            cppn = create_complex_cppn()

    return n_finds, max_evals, best_order


def search_hybrid(threshold, max_evals=1000, target_finds=5, noise_scale=0.2):
    """
    Hybrid: weight mutation + noise injection.
    Returns: (n_finds, n_evals, best_order)
    """
    cppn = create_complex_cppn()
    best_order = 0.0
    n_finds = 0

    for i in range(max_evals):
        # Render with noise
        img = render_with_noise(cppn, size=32, noise_scale=noise_scale)
        order = order_multiplicative(img)

        if order > best_order:
            best_order = order

        if order >= threshold:
            n_finds += 1
            if n_finds >= target_finds:
                return n_finds, i + 1, best_order

        # Mutate weights (smaller step since also using noise)
        weights = cppn.get_weights()
        weights += np.random.randn(len(weights)) * 0.2
        cppn.set_weights(weights)

    return n_finds, max_evals, best_order


def run_experiment():
    """Run comparison experiment."""
    np.random.seed(42)

    n_trials = 50
    threshold = 0.3  # Moderate order threshold
    max_evals = 500
    target_finds = 3

    results = {
        'standard': {'evals': [], 'finds': [], 'best': []},
        'noise': {'evals': [], 'finds': [], 'best': []},
        'hybrid': {'evals': [], 'finds': [], 'best': []}
    }

    print(f"Running {n_trials} trials with threshold={threshold}")
    print("="*60)

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

        # Standard search
        np.random.seed(1000 + trial)
        finds, evals, best = search_standard(threshold, max_evals, target_finds)
        results['standard']['finds'].append(finds)
        results['standard']['evals'].append(evals)
        results['standard']['best'].append(best)

        # Noise-injected search
        np.random.seed(1000 + trial)
        finds, evals, best = search_noise_injected(threshold, max_evals, target_finds)
        results['noise']['finds'].append(finds)
        results['noise']['evals'].append(evals)
        results['noise']['best'].append(best)

        # Hybrid search
        np.random.seed(1000 + trial)
        finds, evals, best = search_hybrid(threshold, max_evals, target_finds)
        results['hybrid']['finds'].append(finds)
        results['hybrid']['evals'].append(evals)
        results['hybrid']['best'].append(best)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    for method in ['standard', 'noise', 'hybrid']:
        finds = np.array(results[method]['finds'])
        evals = np.array(results[method]['evals'])
        best = np.array(results[method]['best'])

        success_rate = np.mean(finds >= target_finds)
        avg_evals = np.mean(evals)
        avg_best = np.mean(best)

        print(f"\n{method.upper()}:")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Avg evals to target: {avg_evals:.1f}")
        print(f"  Avg best order: {avg_best:.3f}")

    # Statistical comparison: standard vs noise
    std_evals = np.array(results['standard']['evals'])
    noise_evals = np.array(results['noise']['evals'])
    hybrid_evals = np.array(results['hybrid']['evals'])

    std_best = np.array(results['standard']['best'])
    noise_best = np.array(results['noise']['best'])
    hybrid_best = np.array(results['hybrid']['best'])

    # Test 1: Does noise find faster?
    t_stat, p_evals = stats.ttest_rel(std_evals, noise_evals)
    effect_evals = (np.mean(std_evals) - np.mean(noise_evals)) / np.std(std_evals)

    # Test 2: Does noise find higher order?
    t_stat2, p_best = stats.ttest_rel(std_best, noise_best)
    effect_best = (np.mean(noise_best) - np.mean(std_best)) / np.std(std_best)

    # Test 3: Hybrid vs standard
    t_stat3, p_hybrid = stats.ttest_rel(std_evals, hybrid_evals)
    effect_hybrid = (np.mean(std_evals) - np.mean(hybrid_evals)) / np.std(std_evals)

    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    print(f"\nNoise vs Standard (evals):")
    print(f"  Effect size: {effect_evals:.3f} (positive = noise faster)")
    print(f"  p-value: {p_evals:.6f}")

    print(f"\nNoise vs Standard (best order):")
    print(f"  Effect size: {effect_best:.3f} (positive = noise better)")
    print(f"  p-value: {p_best:.6f}")

    print(f"\nHybrid vs Standard (evals):")
    print(f"  Effect size: {effect_hybrid:.3f} (positive = hybrid faster)")
    print(f"  p-value: {p_hybrid:.6f}")

    # Determine verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Main hypothesis: noise finds faster
    if effect_evals > 0.5 and p_evals < 0.01:
        status = "VALIDATED"
        print(f"VALIDATED: Noise injection finds targets {effect_evals:.2f}σ faster")
    elif effect_evals < -0.5 and p_evals < 0.01:
        status = "REFUTED"
        print(f"REFUTED: Standard search is faster (effect={effect_evals:.2f})")
    else:
        status = "INCONCLUSIVE"
        print(f"INCONCLUSIVE: No clear winner (effect={effect_evals:.2f}, p={p_evals:.4f})")

    return {
        'status': status,
        'effect_size': effect_evals,
        'p_value': p_evals,
        'effect_best': effect_best,
        'p_best': p_best,
        'results': results
    }


if __name__ == '__main__':
    result = run_experiment()
    print(f"\nFinal status: {result['status']}")
