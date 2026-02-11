"""
RES-195: Weight vector norm predicts ESS contraction count during NS

Hypothesis: Weight vector norm predicts ESS contraction count during NS
- Higher norm weights may explore different regions requiring more/fewer contractions
- Tests if weight configuration magnitude affects optimization difficulty

Metrics:
- Weight vector L2 norm at each ESS step
- Number of contractions to find valid proposal
- Correlation between norm and contractions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample
)


def run_experiment(n_paths: int = 100, max_steps: int = 50, image_size: int = 32):
    """
    Run nested sampling paths and track weight norm vs ESS contractions.
    """

    all_norms = []
    all_contractions = []
    all_orders = []

    # Also track per-path data
    path_data = []

    print(f"Running {n_paths} NS paths, up to {max_steps} steps each...")

    for path in range(n_paths):
        if (path + 1) % 20 == 0:
            print(f"  Path {path + 1}/{n_paths}")

        # Start with fresh CPPN
        cppn = CPPN()
        img = cppn.render(image_size)
        current_order = order_multiplicative(img)

        path_norms = []
        path_contractions = []
        path_orders = []

        # Run NS-like progression
        threshold = current_order
        consecutive_failures = 0

        for step in range(max_steps):
            # Get current weight norm
            weights = cppn.get_weights()
            norm = np.linalg.norm(weights)

            # Try ESS step
            new_cppn, new_img, new_order, _, n_contractions, success = elliptical_slice_sample(
                cppn, threshold, image_size, order_multiplicative
            )

            # Record data for this step
            path_norms.append(norm)
            path_contractions.append(n_contractions)
            path_orders.append(current_order)

            all_norms.append(norm)
            all_contractions.append(n_contractions)
            all_orders.append(current_order)

            if success:
                cppn = new_cppn
                current_order = new_order
                threshold = new_order
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    break

        path_data.append({
            'norms': path_norms,
            'contractions': path_contractions,
            'orders': path_orders,
            'final_order': current_order
        })

    return all_norms, all_contractions, all_orders, path_data


def analyze_results(all_norms, all_contractions, all_orders, path_data):
    """Analyze correlation between weight norm and ESS contractions."""

    print("\n" + "="*60)
    print("RES-195: Weight Norm vs ESS Contractions Analysis")
    print("="*60)

    all_norms = np.array(all_norms)
    all_contractions = np.array(all_contractions)
    all_orders = np.array(all_orders)

    print(f"\nTotal observations: {len(all_norms)}")
    print(f"Norm range: {all_norms.min():.3f} - {all_norms.max():.3f}")
    print(f"Mean norm: {all_norms.mean():.3f} +/- {all_norms.std():.3f}")
    print(f"Contractions range: {all_contractions.min():.0f} - {all_contractions.max():.0f}")
    print(f"Mean contractions: {all_contractions.mean():.2f} +/- {all_contractions.std():.2f}")

    # Primary analysis: norm vs contractions
    print("\n" + "-"*60)
    print("Primary Analysis: Norm vs Contractions")

    r_spearman, p_spearman = stats.spearmanr(all_norms, all_contractions)
    r_pearson, p_pearson = stats.pearsonr(all_norms, all_contractions)

    print(f"  Spearman r = {r_spearman:.4f}, p = {p_spearman:.2e}")
    print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")

    # Split by norm median
    median_norm = np.median(all_norms)
    low_norm_mask = all_norms < median_norm
    high_norm_mask = all_norms >= median_norm

    low_norm_contractions = all_contractions[low_norm_mask]
    high_norm_contractions = all_contractions[high_norm_mask]

    mean_low = low_norm_contractions.mean()
    mean_high = high_norm_contractions.mean()

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((low_norm_contractions.var() + high_norm_contractions.var()) / 2)
    cohens_d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U test
    U, p_mw = stats.mannwhitneyu(low_norm_contractions, high_norm_contractions, alternative='two-sided')

    print(f"\n  Median split (norm = {median_norm:.3f}):")
    print(f"    Low norm (<median): mean contractions = {mean_low:.2f} +/- {low_norm_contractions.std():.2f}")
    print(f"    High norm (>=median): mean contractions = {mean_high:.2f} +/- {high_norm_contractions.std():.2f}")
    print(f"  Cohen's d = {cohens_d:.3f}")
    print(f"  Mann-Whitney U = {U:.0f}, p = {p_mw:.2e}")

    # Control for order (partial correlation)
    print("\n" + "-"*60)
    print("Controlling for Order (Partial Correlation)")

    # Linear regression to get residuals
    from scipy.stats import pearsonr

    # Residualize contractions on order
    slope_co, intercept_co, _, _, _ = stats.linregress(all_orders, all_contractions)
    contractions_residual = all_contractions - (slope_co * all_orders + intercept_co)

    # Residualize norm on order
    slope_no, intercept_no, _, _, _ = stats.linregress(all_orders, all_norms)
    norm_residual = all_norms - (slope_no * all_orders + intercept_no)

    # Partial correlation
    r_partial, p_partial = stats.pearsonr(norm_residual, contractions_residual)
    rho_partial, p_partial_spearman = stats.spearmanr(norm_residual, contractions_residual)

    print(f"  Order-Norm correlation: r = {stats.pearsonr(all_orders, all_norms)[0]:.4f}")
    print(f"  Order-Contractions correlation: r = {stats.pearsonr(all_orders, all_contractions)[0]:.4f}")
    print(f"  Partial correlation (norm-contractions | order):")
    print(f"    Pearson r = {r_partial:.4f}, p = {p_partial:.2e}")
    print(f"    Spearman rho = {rho_partial:.4f}, p = {p_partial_spearman:.2e}")

    # Within-path analysis (controls for path-level confounds)
    print("\n" + "-"*60)
    print("Within-Path Analysis")

    within_path_correlations = []
    for pd in path_data:
        if len(pd['norms']) >= 5:
            r, _ = stats.spearmanr(pd['norms'], pd['contractions'])
            if not np.isnan(r):
                within_path_correlations.append(r)

    if within_path_correlations:
        mean_within_r = np.mean(within_path_correlations)
        std_within_r = np.std(within_path_correlations)
        # One-sample t-test against 0
        t_stat, p_within = stats.ttest_1samp(within_path_correlations, 0)

        print(f"  N paths with sufficient data: {len(within_path_correlations)}")
        print(f"  Mean within-path correlation: r = {mean_within_r:.4f} +/- {std_within_r:.4f}")
        print(f"  One-sample t-test vs 0: t = {t_stat:.3f}, p = {p_within:.2e}")

    # Return results dict
    results = {
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'cohens_d': cohens_d,
        'p_mannwhitney': p_mw,
        'mean_low_norm': mean_low,
        'mean_high_norm': mean_high,
        'r_partial': r_partial,
        'p_partial': p_partial,
        'n_observations': len(all_norms),
        'mean_within_r': mean_within_r if within_path_correlations else np.nan,
        'p_within': p_within if within_path_correlations else np.nan
    }

    return results


def main():
    print("RES-195: Weight vector norm predicts ESS contraction count during NS")
    print("="*60)

    np.random.seed(42)

    all_norms, all_contractions, all_orders, path_data = run_experiment(
        n_paths=200, max_steps=50, image_size=32
    )

    results = analyze_results(all_norms, all_contractions, all_orders, path_data)

    print("\n" + "="*60)
    print("CONCLUSION:")

    # Primary criterion: significant correlation with meaningful effect size
    p_val = results['p_spearman']
    r_val = results['r_spearman']
    d_val = results['cohens_d']

    # Also check partial correlation (controlling for order)
    p_partial = results['p_partial']
    r_partial = results['r_partial']

    if p_val < 0.01 and abs(d_val) > 0.5:
        if r_val > 0:
            direction = "Higher norm -> more contractions"
        else:
            direction = "Higher norm -> fewer contractions"
        print(f"  VALIDATED: Weight norm predicts ESS contractions")
        print(f"  Direction: {direction}")
        print(f"  r = {r_val:.4f}, d = {d_val:.3f}, p = {p_val:.2e}")
        status = "validated"
    elif p_val < 0.01 and abs(d_val) < 0.5:
        print(f"  REFUTED: Significant but small effect (d = {d_val:.3f} < 0.5)")
        print(f"  r = {r_val:.4f}, p = {p_val:.2e}")
        status = "refuted"
    elif p_val >= 0.01:
        print(f"  REFUTED: No significant correlation")
        print(f"  r = {r_val:.4f}, p = {p_val:.2e}")
        status = "refuted"
    else:
        print(f"  INCONCLUSIVE")
        status = "inconclusive"

    # Additional note about partial correlation
    if p_partial < 0.01:
        print(f"\n  Note: Effect persists after controlling for order (r_partial = {r_partial:.4f})")
    else:
        print(f"\n  Note: Effect does NOT persist after controlling for order (r_partial = {r_partial:.4f}, p = {p_partial:.2e})")

    return status, results


if __name__ == "__main__":
    status, results = main()
    print(f"\nFinal status: {status}")
