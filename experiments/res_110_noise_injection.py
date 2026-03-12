#!/usr/bin/env python3
"""
RES-110: Hidden Layer Noise Injection for Diversity

HYPOTHESIS: Adding noise to CPPN hidden layer activations during inference
increases output diversity (measured by pairwise image variance) without
significantly reducing average order score.

DOMAIN: noise_injection
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats


def activate_with_noise(cppn, x, y, noise_scale=0.0):
    """Modified activate that injects Gaussian noise into hidden activations."""
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

        # Add noise to hidden layers only (not output)
        if noise_scale > 0 and nid != cppn.output_id:
            activation_val += np.random.randn(*activation_val.shape) * noise_scale

        values[nid] = activation_val

    return values[cppn.output_id]


def render_with_noise(cppn, size=32, noise_scale=0.0):
    """Render CPPN image with optional noise injection."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return (activate_with_noise(cppn, x, y, noise_scale) > 0.5).astype(np.uint8)


def measure_diversity(images):
    """Measure diversity as mean pairwise L2 distance between images."""
    n = len(images)
    if n < 2:
        return 0.0

    flat = np.array([img.flatten().astype(float) for img in images])
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dist += np.linalg.norm(flat[i] - flat[j])
            count += 1
    return total_dist / count


def run_experiment():
    """Run the noise injection experiment."""
    np.random.seed(42)

    n_cppns = 100
    n_samples_per_cppn = 10
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

    results = {level: {'diversity': [], 'order': []} for level in noise_levels}

    print("Generating CPPNs and testing noise injection...")

    for i in range(n_cppns):
        # Create random CPPN with some hidden nodes
        cppn = CPPN()
        # Add hidden nodes
        n_hidden = np.random.randint(2, 5)
        activations = ['sigmoid', 'tanh', 'sin', 'relu', 'gauss']
        for h in range(n_hidden):
            node_id = 5 + h
            cppn.nodes.append(type(cppn.nodes[0])(
                id=node_id,
                activation=np.random.choice(activations),
                bias=np.random.randn() * 0.5
            ))
            # Connect from inputs
            for inp_id in cppn.input_ids:
                if np.random.rand() < 0.5:
                    cppn.connections.append(type(cppn.connections[0])(
                        from_id=inp_id, to_id=node_id,
                        weight=np.random.randn() * 1.0, enabled=True
                    ))
            # Connect to output
            if np.random.rand() < 0.7:
                cppn.connections.append(type(cppn.connections[0])(
                    from_id=node_id, to_id=cppn.output_id,
                    weight=np.random.randn() * 1.0, enabled=True
                ))

        for noise_level in noise_levels:
            # Generate multiple samples with same CPPN but potentially different noise
            images = []
            for _ in range(n_samples_per_cppn):
                img = render_with_noise(cppn, size=32, noise_scale=noise_level)
                images.append(img)

            # Measure diversity across samples
            diversity = measure_diversity(images)
            results[noise_level]['diversity'].append(diversity)

            # Measure average order
            orders = [compute_order(img) for img in images]
            results[noise_level]['order'].append(np.mean(orders))

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Baseline (no noise)
    baseline_diversity = np.mean(results[0.0]['diversity'])
    baseline_order = np.mean(results[0.0]['order'])

    print(f"\nBaseline (noise=0): diversity={baseline_diversity:.4f}, order={baseline_order:.4f}")

    # Compare each noise level to baseline
    best_noise_level = 0.0
    best_effect_size = 0.0
    best_p_value = 1.0
    best_order_preserved = True

    for noise_level in noise_levels[1:]:
        div = np.array(results[noise_level]['diversity'])
        div_base = np.array(results[0.0]['diversity'])

        ord_noise = np.array(results[noise_level]['order'])
        ord_base = np.array(results[0.0]['order'])

        # Test diversity increase
        t_div, p_div = stats.ttest_rel(div, div_base)
        effect_div = (np.mean(div) - np.mean(div_base)) / np.std(div_base)

        # Test order preservation (should not decrease significantly)
        t_ord, p_ord = stats.ttest_rel(ord_noise, ord_base)
        order_drop = (np.mean(ord_base) - np.mean(ord_noise)) / np.mean(ord_base)

        print(f"\nNoise={noise_level}:")
        print(f"  Diversity: {np.mean(div):.4f} (effect_size={effect_div:.3f}, p={p_div:.4f})")
        print(f"  Order: {np.mean(ord_noise):.4f} (drop={order_drop*100:.1f}%, p={p_ord:.4f})")

        # Track best noise level that increases diversity without hurting order too much
        order_preserved = order_drop < 0.1  # Less than 10% order drop
        if effect_div > best_effect_size and p_div < 0.01 and order_preserved:
            best_effect_size = effect_div
            best_p_value = p_div
            best_noise_level = noise_level
            best_order_preserved = order_preserved

    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)

    if best_effect_size > 0.5 and best_p_value < 0.01 and best_order_preserved:
        status = "VALIDATED"
        print(f"✓ VALIDATED: Noise level {best_noise_level} increases diversity")
        print(f"  effect_size={best_effect_size:.3f}, p={best_p_value:.6f}")
    elif best_effect_size > 0 and best_p_value < 0.05:
        status = "INCONCLUSIVE"
        print(f"? INCONCLUSIVE: Effect detected but below threshold")
        print(f"  effect_size={best_effect_size:.3f}, p={best_p_value:.6f}")
    else:
        status = "REFUTED"
        print(f"✗ REFUTED: No significant diversity increase with order preservation")
        print(f"  best_effect_size={best_effect_size:.3f}")

    return {
        'status': status,
        'best_noise_level': best_noise_level,
        'effect_size': best_effect_size,
        'p_value': best_p_value,
        'baseline_diversity': baseline_diversity,
        'baseline_order': baseline_order,
        'results': {k: {'mean_diversity': np.mean(v['diversity']),
                        'mean_order': np.mean(v['order'])}
                   for k, v in results.items()}
    }


if __name__ == '__main__':
    result = run_experiment()
    print(f"\nFinal: {result['status']}")
