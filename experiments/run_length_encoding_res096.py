"""
RES-096: Test if high-order CPPN images have longer run-lengths than random images.

Run-length = consecutive pixels of same value
Hypothesis: structured images have longer continuous regions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_run_lengths(image, threshold=0.5):
    """Compute run-length statistics for a binary image."""
    binary = (image > threshold).astype(int)

    run_lengths = []
    for row in binary:
        runs = []
        current_run = 1
        for i in range(1, len(row)):
            if row[i] == row[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        run_lengths.extend(runs)

    return np.array(run_lengths)


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    # Generate coordinates
    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPN images
    print("Generating CPPN images...")
    cppn_images = []
    cppn_orders = []
    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)
        cppn_images.append(img)
        cppn_orders.append(order)

    # Generate random images
    print("Generating random images...")
    random_images = [np.random.random((resolution, resolution)) for _ in range(n_samples)]

    # Compute run-length statistics
    print("Computing run-lengths...")
    cppn_run_lengths = [compute_run_lengths(img) for img in cppn_images]
    random_run_lengths = [compute_run_lengths(img) for img in random_images]

    cppn_mean_runs = np.array([np.mean(runs) for runs in cppn_run_lengths])
    random_mean_runs = np.array([np.mean(runs) for runs in random_run_lengths])

    cppn_max_runs = np.array([np.max(runs) for runs in cppn_run_lengths])
    random_max_runs = np.array([np.max(runs) for runs in random_run_lengths])

    # Statistical tests
    t_mean, p_mean = stats.ttest_ind(cppn_mean_runs, random_mean_runs)
    t_max, p_max = stats.ttest_ind(cppn_max_runs, random_max_runs)

    # Effect sizes
    pooled_std_mean = np.sqrt((np.std(cppn_mean_runs)**2 + np.std(random_mean_runs)**2) / 2)
    d_mean = (np.mean(cppn_mean_runs) - np.mean(random_mean_runs)) / pooled_std_mean

    pooled_std_max = np.sqrt((np.std(cppn_max_runs)**2 + np.std(random_max_runs)**2) / 2)
    d_max = (np.mean(cppn_max_runs) - np.mean(random_max_runs)) / pooled_std_max

    # Correlation with order
    corr_order, p_order = stats.pearsonr(cppn_orders, cppn_mean_runs)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"CPPN mean run length: {np.mean(cppn_mean_runs):.2f}")
    print(f"Random mean run length: {np.mean(random_mean_runs):.2f}")
    print(f"Effect size (Cohen's d): {d_mean:.2f}")
    print(f"p-value: {p_mean:.2e}")
    print(f"\nCPPN max run length: {np.mean(cppn_max_runs):.2f}")
    print(f"Random max run length: {np.mean(random_max_runs):.2f}")
    print(f"Max effect size: {d_max:.2f}")
    print(f"p-value: {p_max:.2e}")
    print(f"\nOrder correlation (within CPPN): r={corr_order:.3f}, p={p_order:.2e}")

    # Determine validation
    validated = d_mean > 0.5 and p_mean < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, d_mean, p_mean


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
