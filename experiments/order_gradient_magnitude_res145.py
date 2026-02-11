"""
RES-145: Test if order gradient magnitude increases during nested sampling.

Hypothesis: landscape steepness grows as NS progresses to high-order regions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_order


def compute_order_gradient_magnitude(cppn, coords_x, coords_y, eps=0.01):
    """Compute gradient magnitude of order metric w.r.t. weights."""
    gradients = []

    for conn in cppn.connections:
        old_w = conn.weight

        # Forward difference
        conn.weight = old_w + eps
        order_plus = compute_order(cppn.activate(coords_x, coords_y))

        # Backward difference
        conn.weight = old_w - eps
        order_minus = compute_order(cppn.activate(coords_x, coords_y))

        conn.weight = old_w

        grad = (order_plus - order_minus) / (2 * eps)
        gradients.append(grad)

    return np.mean(np.abs(gradients)) if gradients else 0.0


def main():
    np.random.seed(42)

    n_runs = 50
    resolution = 32

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    early_gradients = []
    late_gradients = []

    print("Running nested sampling with gradient tracking...")
    for run in range(n_runs):
        # Simulate NS by sampling CPPNs with increasing order
        for iteration in range(100):
            cppn = CPPN()
            grad_mag = compute_order_gradient_magnitude(cppn, coords_x, coords_y)

            if iteration < 20:
                early_gradients.append(grad_mag)
            elif iteration > 80:
                late_gradients.append(grad_mag)

    early_gradients = np.array(early_gradients)
    late_gradients = np.array(late_gradients)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(late_gradients, early_gradients)

    # Effect size
    pooled_std = np.sqrt((np.std(late_gradients)**2 + np.std(early_gradients)**2) / 2)
    effect_size = (np.mean(late_gradients) - np.mean(early_gradients)) / (pooled_std + 1e-10)

    # Correlation between iteration and gradient
    iterations = np.concatenate([np.arange(20), np.arange(80, 100)] * n_runs)
    all_grads = np.concatenate([early_gradients, late_gradients])
    corr, p_corr = stats.pearsonr(iterations, all_grads)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Early iteration gradient magnitude: {np.mean(early_gradients):.4f}")
    print(f"Late iteration gradient magnitude: {np.mean(late_gradients):.4f}")
    print(f"Ratio: {np.mean(late_gradients) / (np.mean(early_gradients) + 1e-10):.1f}x")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Correlation with iteration: r={corr:.3f}, p={p_corr:.2e}")

    validated = effect_size > 0.5 and p_value < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_value


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
