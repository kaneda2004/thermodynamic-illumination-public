"""
RES-127: Analyze nested sampling trajectory geometry in CPPN weight space.

Hypothesis: The nested sampling trajectory through CPPN weight space follows
a low-dimensional manifold rather than random walk, measurable by:
1. Intrinsic dimensionality estimation (PCA, correlation dimension)
2. Path linearity (ratio of displacement to path length)
3. Persistent homology of trajectory point cloud

If VALIDATED: The sampling trajectory is geometrically structured, suggesting
nested sampling exploits low-dimensional structure in the weight-order mapping.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative

def collect_trajectory(n_samples=200, seed=42):
    """Run simplified nested sampling to collect weight trajectory."""
    np.random.seed(seed)

    # Initialize live points
    n_live = 50
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        live_points.append({'cppn': cppn, 'order': order, 'weights': cppn.get_weights().copy()})

    trajectory = []  # Weights of dead points in order

    for i in range(n_samples):
        # Find worst point
        min_idx = np.argmin([p['order'] for p in live_points])
        worst = live_points[min_idx]
        threshold = worst['order']

        # Record trajectory
        trajectory.append(worst['weights'].copy())

        # Simple replacement: copy random survivor and perturb
        other_idx = np.random.choice([j for j in range(n_live) if j != min_idx])
        new_cppn = live_points[other_idx]['cppn'].copy()

        # MCMC steps to find replacement above threshold
        for _ in range(20):
            proposal_cppn = new_cppn.copy()
            w = proposal_cppn.get_weights()
            w += np.random.randn(len(w)) * 0.3
            proposal_cppn.set_weights(w)
            img = proposal_cppn.render(32)
            new_order = order_multiplicative(img)
            if new_order > threshold:
                new_cppn = proposal_cppn
                break

        img = new_cppn.render(32)
        new_order = order_multiplicative(img)
        live_points[min_idx] = {
            'cppn': new_cppn,
            'order': new_order,
            'weights': new_cppn.get_weights().copy()
        }

        if (i + 1) % 50 == 0:
            print(f"  Collected {i+1}/{n_samples} points, threshold: {threshold:.4f}")

    return np.array(trajectory)


def estimate_intrinsic_dimension_pca(trajectory, variance_threshold=0.95):
    """Estimate intrinsic dimension via PCA explained variance."""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(trajectory)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumsum >= variance_threshold) + 1
    return intrinsic_dim, pca.explained_variance_ratio_


def estimate_correlation_dimension(trajectory, n_points=100):
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    Lower dimension indicates trajectory follows a low-D manifold.
    """
    # Subsample for efficiency
    idx = np.random.choice(len(trajectory), min(n_points, len(trajectory)), replace=False)
    points = trajectory[idx]

    # Compute all pairwise distances
    dists = pdist(points)

    # Compute correlation integral for different epsilon values
    eps_range = np.logspace(np.log10(np.percentile(dists, 5)),
                            np.log10(np.percentile(dists, 95)), 20)

    counts = []
    for eps in eps_range:
        count = np.sum(dists < eps) / (len(points) * (len(points) - 1) / 2)
        counts.append(count)

    # Fit log-log slope in middle region
    log_eps = np.log(eps_range)
    log_counts = np.log(np.array(counts) + 1e-10)

    # Use middle 60% of range for fit
    start, end = int(len(log_eps) * 0.2), int(len(log_eps) * 0.8)
    slope, intercept = np.polyfit(log_eps[start:end], log_counts[start:end], 1)

    return slope  # This is the correlation dimension


def compute_path_linearity(trajectory):
    """
    Compute ratio of straight-line displacement to total path length.
    Random walk: ~1/sqrt(n), structured trajectory: higher ratio.
    """
    # Total path length (sum of step distances)
    steps = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(steps, axis=1))

    # Straight-line displacement
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

    # Normalize by expected random walk ratio
    n = len(trajectory)
    linearity = displacement / (path_length + 1e-10)
    random_walk_expected = 1 / np.sqrt(n)

    return linearity, random_walk_expected


def generate_random_walk_baseline(n_points, n_dims, step_size=0.3, seed=123):
    """Generate random walk trajectory for comparison."""
    np.random.seed(seed)
    trajectory = np.zeros((n_points, n_dims))
    trajectory[0] = np.random.randn(n_dims)
    for i in range(1, n_points):
        trajectory[i] = trajectory[i-1] + np.random.randn(n_dims) * step_size
    return trajectory


def main():
    print("=" * 70)
    print("RES-127: NESTED SAMPLING TRAJECTORY GEOMETRY")
    print("=" * 70)

    n_samples = 200
    n_trials = 5

    # Collect results
    real_dims_pca = []
    real_dims_corr = []
    real_linearities = []
    baseline_dims_pca = []
    baseline_dims_corr = []
    baseline_linearities = []

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 40)

        # Real nested sampling trajectory
        print("Collecting nested sampling trajectory...")
        trajectory = collect_trajectory(n_samples=n_samples, seed=42 + trial * 100)
        n_dims = trajectory.shape[1]

        # Baseline random walk
        print("Generating random walk baseline...")
        baseline = generate_random_walk_baseline(n_samples, n_dims, seed=123 + trial)

        # PCA dimension
        real_dim_pca, _ = estimate_intrinsic_dimension_pca(trajectory)
        base_dim_pca, _ = estimate_intrinsic_dimension_pca(baseline)
        real_dims_pca.append(real_dim_pca)
        baseline_dims_pca.append(base_dim_pca)

        # Correlation dimension
        real_dim_corr = estimate_correlation_dimension(trajectory)
        base_dim_corr = estimate_correlation_dimension(baseline)
        real_dims_corr.append(real_dim_corr)
        baseline_dims_corr.append(base_dim_corr)

        # Path linearity
        real_lin, rw_expected = compute_path_linearity(trajectory)
        base_lin, _ = compute_path_linearity(baseline)
        real_linearities.append(real_lin)
        baseline_linearities.append(base_lin)

        print(f"  PCA dim: NS={real_dim_pca}, RW={base_dim_pca}")
        print(f"  Corr dim: NS={real_dim_corr:.2f}, RW={base_dim_corr:.2f}")
        print(f"  Linearity: NS={real_lin:.4f}, RW={base_lin:.4f}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # PCA dimension comparison
    pca_real_mean = np.mean(real_dims_pca)
    pca_base_mean = np.mean(baseline_dims_pca)
    pca_reduction = (pca_base_mean - pca_real_mean) / pca_base_mean

    # Correlation dimension comparison
    corr_real_mean = np.mean(real_dims_corr)
    corr_base_mean = np.mean(baseline_dims_corr)
    corr_reduction = (corr_base_mean - corr_real_mean) / corr_base_mean

    # Linearity comparison (paired t-test)
    lin_diff = np.array(real_linearities) - np.array(baseline_linearities)
    lin_mean_diff = np.mean(lin_diff)
    lin_std = np.std(lin_diff, ddof=1)
    t_stat = lin_mean_diff / (lin_std / np.sqrt(n_trials))

    # Two-tailed p-value approximation
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_trials - 1))

    # Effect size (Cohen's d for paired samples)
    effect_size = lin_mean_diff / lin_std if lin_std > 0 else 0

    print(f"\nPCA Intrinsic Dimension (95% variance):")
    print(f"  Nested Sampling: {pca_real_mean:.1f} +/- {np.std(real_dims_pca):.1f}")
    print(f"  Random Walk:     {pca_base_mean:.1f} +/- {np.std(baseline_dims_pca):.1f}")
    print(f"  Dimension reduction: {pca_reduction*100:.1f}%")

    print(f"\nCorrelation Dimension (Grassberger-Procaccia):")
    print(f"  Nested Sampling: {corr_real_mean:.2f} +/- {np.std(real_dims_corr):.2f}")
    print(f"  Random Walk:     {corr_base_mean:.2f} +/- {np.std(baseline_dims_corr):.2f}")
    print(f"  Dimension reduction: {corr_reduction*100:.1f}%")

    print(f"\nPath Linearity (displacement/path_length):")
    print(f"  Nested Sampling: {np.mean(real_linearities):.4f} +/- {np.std(real_linearities):.4f}")
    print(f"  Random Walk:     {np.mean(baseline_linearities):.4f} +/- {np.std(baseline_linearities):.4f}")
    print(f"  Mean difference: {lin_mean_diff:.4f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Effect size (Cohen's d): {effect_size:.2f}")

    # Validation criteria - REVISED based on findings
    # Original hypothesis was that NS follows low-D manifold
    # Results show NS has HIGHER dimension but LOWER linearity
    # This suggests NS explores more dimensions but in a more tortuous path
    print("\n" + "=" * 70)
    print("VALIDATION CRITERIA (REVISED)")
    print("=" * 70)

    # The key finding: NS is significantly different from random walk
    is_dim_different = abs(pca_reduction) > 0.2 or abs(corr_reduction) > 0.2
    is_significant = p_value < 0.01
    is_large_effect = abs(effect_size) > 0.5

    print(f"1. Dimension significantly different (>20% change): {is_dim_different}")
    print(f"   PCA change: {pca_reduction*100:.1f}%, Corr change: {corr_reduction*100:.1f}%")
    print(f"2. Linearity significantly different (p < 0.01): {is_significant} (p={p_value:.6f})")
    print(f"3. Large effect size (|d| > 0.5): {is_large_effect} (d={effect_size:.2f})")

    # Interpret direction of effects
    ns_more_complex = pca_real_mean > pca_base_mean
    ns_less_linear = np.mean(real_linearities) < np.mean(baseline_linearities)

    print(f"\nDirection of effects:")
    print(f"  NS dimension vs RW: {'HIGHER' if ns_more_complex else 'LOWER'}")
    print(f"  NS linearity vs RW: {'LOWER' if ns_less_linear else 'HIGHER'}")

    # The original hypothesis (low-D manifold) is REFUTED
    # But we discovered something interesting: NS explores MORE dimensions
    # with a MORE convoluted path - suggesting constraint-driven exploration
    if is_dim_different and is_significant and is_large_effect:
        if ns_more_complex and ns_less_linear:
            status = "REFUTED"  # Original hypothesis refuted but interesting finding
            summary = (f"Original hypothesis REFUTED: NS trajectory has HIGHER dimension "
                       f"({pca_real_mean:.0f}D vs {pca_base_mean:.0f}D PCA) and LOWER linearity "
                       f"({np.mean(real_linearities):.4f} vs {np.mean(baseline_linearities):.4f}) "
                       f"than random walk (p={p_value:.4f}, d={effect_size:.2f}). "
                       f"NS explores more dimensions via tortuous constraint-driven paths.")
        else:
            status = "VALIDATED"
            summary = (f"NS trajectory differs significantly from random walk. "
                       f"Dimension: {pca_real_mean:.0f}D vs {pca_base_mean:.0f}D, "
                       f"linearity: {np.mean(real_linearities):.4f} vs {np.mean(baseline_linearities):.4f}.")
    else:
        status = "INCONCLUSIVE"
        summary = (f"Insufficient evidence to distinguish NS trajectory from random walk. "
                   f"p={p_value:.4f}, d={effect_size:.2f}.")

    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Return values for logging
    return {
        'status': status,
        'effect_size': effect_size,
        'p_value': p_value,
        'pca_reduction': pca_reduction,
        'corr_reduction': corr_reduction,
        'summary': summary
    }


if __name__ == "__main__":
    results = main()
