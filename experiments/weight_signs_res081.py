"""
RES-081: Test if weight sign patterns affect CPPN order.

Hypothesis: uniform signs (all positive/negative) collapse order
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, Connection


def modify_weight_signs(cppn, pattern='all_positive'):
    """Modify weight signs according to pattern."""
    cppn_mod = CPPN(
        nodes=cppn.nodes,
        connections=[
            Connection(c.from_id, c.to_id, c.weight, c.enabled)
            for c in cppn.connections
        ],
        input_ids=cppn.input_ids,
        output_id=cppn.output_id
    )

    if pattern == 'all_positive':
        for conn in cppn_mod.connections:
            conn.weight = abs(conn.weight)
    elif pattern == 'all_negative':
        for conn in cppn_mod.connections:
            conn.weight = -abs(conn.weight)
    elif pattern == 'random_flip':
        for conn in cppn_mod.connections:
            if np.random.random() < 0.5:
                conn.weight = -conn.weight
    elif pattern == 'alternating':
        for i, conn in enumerate(cppn_mod.connections):
            if i % 2 == 0:
                conn.weight = abs(conn.weight)
            else:
                conn.weight = -abs(conn.weight)

    return cppn_mod


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    patterns = ['original', 'all_positive', 'all_negative', 'random_flip', 'alternating']
    results = {p: [] for p in patterns}

    print("Testing weight sign patterns...")
    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)

        for pattern in patterns:
            if pattern == 'original':
                cppn_test = cppn
            else:
                cppn_test = modify_weight_signs(cppn, pattern)

            img = cppn_test.activate(coords_x, coords_y)
            order_test = compute_order(img)
            results[pattern].append(order_test)

    # Convert to arrays
    for p in patterns:
        results[p] = np.array(results[p])

    # Statistical analysis
    original = results['original']
    all_pos = results['all_positive']
    all_neg = results['all_negative']
    rand_flip = results['random_flip']
    alt = results['alternating']

    t_pos, p_pos = stats.ttest_ind(all_pos, original)
    t_neg, p_neg = stats.ttest_ind(all_neg, original)
    t_rand, p_rand = stats.ttest_ind(rand_flip, original)
    t_alt, p_alt = stats.ttest_ind(alt, original)

    # Effect sizes
    pooled_std = np.sqrt((np.std(all_pos)**2 + np.std(original)**2) / 2)
    d_pos = (np.mean(all_pos) - np.mean(original)) / (pooled_std + 1e-10)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Original: {np.mean(original):.4f}")
    print(f"All positive: {np.mean(all_pos):.4f}, d={d_pos:.2f}, p={p_pos:.2e}")
    print(f"All negative: {np.mean(all_neg):.4f}")
    print(f"Random flip: {np.mean(rand_flip):.4f}, p={p_rand:.2e}")
    print(f"Alternating: {np.mean(alt):.4f}, p={p_alt:.2e}")

    # Collapse effect: uniform signs destroy structure
    collapse_effect = np.mean(original) - min(np.mean(all_pos), np.mean(all_neg))
    effect_size = abs(d_pos)

    validated = effect_size > 0.5 and (p_pos < 0.01 or p_neg < 0.01)
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_pos


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
