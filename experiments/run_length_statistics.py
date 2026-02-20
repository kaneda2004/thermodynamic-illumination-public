"""
RES-201: Run-length distribution statistics correlate with CPPN output order

Hypothesis: Run-length distribution statistics (mean run length, run-length
variability) correlate positively with CPPN output order. High-order images
should have longer runs due to spatial coherence.

Methodology:
- Generate 1000 CPPN images
- Compute run-length statistics for each (horizontal + vertical)
- Correlate with order metric
- Compare to random baseline
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats

set_global_seed(42)


def compute_run_lengths(img: np.ndarray) -> list:
    """Extract run lengths from binary image (both horizontal and vertical)."""
    runs = []

    # Horizontal runs
    for row in img:
        current_val = row[0]
        current_len = 1
        for val in row[1:]:
            if val == current_val:
                current_len += 1
            else:
                runs.append(current_len)
                current_val = val
                current_len = 1
        runs.append(current_len)

    # Vertical runs
    for col in img.T:
        current_val = col[0]
        current_len = 1
        for val in col[1:]:
            if val == current_val:
                current_len += 1
            else:
                runs.append(current_len)
                current_val = val
                current_len = 1
        runs.append(current_len)

    return runs


def run_length_stats(img: np.ndarray) -> dict:
    """Compute run-length statistics."""
    runs = compute_run_lengths(img)
    if not runs:
        return {'mean': 0, 'std': 0, 'max': 0, 'median': 0, 'cv': 0}

    runs = np.array(runs)
    mean_run = np.mean(runs)
    std_run = np.std(runs)
    max_run = np.max(runs)
    median_run = np.median(runs)
    cv = std_run / (mean_run + 1e-10)  # Coefficient of variation

    return {
        'mean': mean_run,
        'std': std_run,
        'max': max_run,
        'median': median_run,
        'cv': cv
    }


def main():
    n_samples = 1000
    size = 32

    print("Generating CPPN samples...")
    orders = []
    run_stats_list = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size)
        order = order_multiplicative(img)
        orders.append(order)
        run_stats_list.append(run_length_stats(img))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples} samples...")

    orders = np.array(orders)
    mean_runs = np.array([s['mean'] for s in run_stats_list])
    std_runs = np.array([s['std'] for s in run_stats_list])
    max_runs = np.array([s['max'] for s in run_stats_list])
    median_runs = np.array([s['median'] for s in run_stats_list])
    cv_runs = np.array([s['cv'] for s in run_stats_list])

    print(f"\n=== CPPN Run-Length Statistics ===")
    print(f"Order: mean={orders.mean():.4f}, std={orders.std():.4f}")
    print(f"Mean run: {mean_runs.mean():.2f} +/- {mean_runs.std():.2f}")
    print(f"Max run: {max_runs.mean():.2f} +/- {max_runs.std():.2f}")
    print(f"Median run: {median_runs.mean():.2f} +/- {median_runs.std():.2f}")

    # Correlations with order
    print(f"\n=== Correlations with Order ===")
    metrics = {
        'mean_run': mean_runs,
        'std_run': std_runs,
        'max_run': max_runs,
        'median_run': median_runs,
        'cv_run': cv_runs
    }

    for name, values in metrics.items():
        r_pearson, p_pearson = stats.pearsonr(orders, values)
        rho_spearman, p_spearman = stats.spearmanr(orders, values)
        print(f"{name}: r={r_pearson:.3f} (p={p_pearson:.2e}), rho={rho_spearman:.3f} (p={p_spearman:.2e})")

    # Primary hypothesis: mean run length correlates with order
    r, p = stats.pearsonr(orders, mean_runs)
    rho, p_rho = stats.spearmanr(orders, mean_runs)

    # Effect size: Cohen's d comparing high vs low order quartiles
    q1, q3 = np.percentile(orders, [25, 75])
    low_order_mask = orders <= q1
    high_order_mask = orders >= q3

    low_mean_run = mean_runs[low_order_mask].mean()
    high_mean_run = mean_runs[high_order_mask].mean()
    pooled_std = np.sqrt((mean_runs[low_order_mask].var() + mean_runs[high_order_mask].var()) / 2)
    d = (high_mean_run - low_mean_run) / (pooled_std + 1e-10)

    print(f"\n=== Primary Result: Mean Run Length ===")
    print(f"Pearson r = {r:.4f}, p = {p:.2e}")
    print(f"Spearman rho = {rho:.4f}, p = {p_rho:.2e}")
    print(f"Low-order quartile mean run: {low_mean_run:.2f}")
    print(f"High-order quartile mean run: {high_mean_run:.2f}")
    print(f"Cohen's d = {d:.2f}")

    # Mann-Whitney test
    mw_stat, mw_p = stats.mannwhitneyu(
        mean_runs[low_order_mask],
        mean_runs[high_order_mask],
        alternative='two-sided'
    )
    print(f"Mann-Whitney U p = {mw_p:.2e}")

    # Compare to random baseline
    print(f"\n=== Random Baseline ===")
    random_runs = []
    for _ in range(500):
        rand_img = (np.random.rand(size, size) > 0.5).astype(np.uint8)
        random_runs.append(run_length_stats(rand_img)['mean'])
    random_runs = np.array(random_runs)
    print(f"Random mean run: {random_runs.mean():.2f} +/- {random_runs.std():.2f}")

    # CPPN vs Random comparison
    d_vs_random = (mean_runs.mean() - random_runs.mean()) / np.sqrt((mean_runs.var() + random_runs.var()) / 2)
    mw_cppn_rand, p_cppn_rand = stats.mannwhitneyu(mean_runs, random_runs, alternative='two-sided')
    print(f"CPPN vs Random: d={d_vs_random:.2f}, MW p={p_cppn_rand:.2e}")

    # Summary
    print(f"\n=== SUMMARY ===")
    validated = abs(d) > 0.5 and p < 0.01
    print(f"Effect size d={d:.2f}, p={p:.2e}")
    print(f"Validated: {validated}")

    # Additional: check median run (might be more robust)
    r_med, p_med = stats.pearsonr(orders, median_runs)
    q1_med = median_runs[low_order_mask].mean()
    q3_med = median_runs[high_order_mask].mean()
    d_med = (q3_med - q1_med) / (np.sqrt((median_runs[low_order_mask].var() + median_runs[high_order_mask].var()) / 2) + 1e-10)
    print(f"\nMedian run: r={r_med:.3f}, d={d_med:.2f}")


if __name__ == "__main__":
    main()
