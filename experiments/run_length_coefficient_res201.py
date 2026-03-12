"""
RES-201: Test if run-length distribution statistics correlate with CPPN output order.

Hypothesis: structured images have variable (not uniform) run lengths
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_run_length_stats(image, threshold=0.5):
    """Compute run-length statistics."""
    binary = (image > threshold).astype(int)

    run_lengths = []
    for row in binary:
        current = 1
        for i in range(1, len(row)):
            if row[i] == row[i-1]:
                current += 1
            else:
                run_lengths.append(current)
                current = 1
        run_lengths.append(current)

    run_lengths = np.array(run_lengths)

    if len(run_lengths) == 0:
        return 0.0, 0.0

    mean_run = np.mean(run_lengths)
    cv_run = np.std(run_lengths) / (np.mean(run_lengths) + 1e-10)

    return mean_run, cv_run


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPNs
    print("Generating CPPNs...")
    orders = []
    mean_runs = []
    cv_runs = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)

        mean_run, cv_run = compute_run_length_stats(img)
        orders.append(order)
        mean_runs.append(mean_run)
        cv_runs.append(cv_run)

    orders = np.array(orders)
    mean_runs = np.array(mean_runs)
    cv_runs = np.array(cv_runs)

    # Correlations
    corr_mean, p_mean = stats.pearsonr(orders, mean_runs)
    corr_cv, p_cv = stats.pearsonr(orders, cv_runs)

    # Spearman for robustness
    spearman_mean, p_spearman = stats.spearmanr(orders, mean_runs)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Correlation (order vs mean run length): r={corr_mean:.3f}, p={p_mean:.2e}")
    print(f"Spearman: rho={spearman_mean:.3f}, p={p_spearman:.2e}")
    print(f"Correlation (order vs run CV): r={corr_cv:.3f}, p={p_cv:.2e}")
    print(f"Mean run length statistics:")
    print(f"  Mean: {np.mean(mean_runs):.2f}")
    print(f"  Std: {np.std(mean_runs):.2f}")

    # Effect size
    effect_size = abs(corr_mean)

    validated = effect_size > 0.5 and p_mean < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_mean


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: r={effect_size:.3f}, p={p_value:.2e}")
