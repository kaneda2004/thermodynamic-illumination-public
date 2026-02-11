"""
RES-118: Live points in nested sampling form distinct clusters in weight space
as iterations progress, with cluster separation increasing monotonically.

Tests whether CPPN weight vectors form clusters during nested sampling and
whether cluster separation (silhouette score) increases over iterations.
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample,
    PRIOR_SIGMA, set_global_seed
)


@dataclass
class SnapshotMetrics:
    iteration: int
    silhouette: float
    inertia: float  # Within-cluster sum of squares
    mean_order: float
    order_variance: float


def run_nested_sampling_with_snapshots(
    n_live: int = 50,
    n_iterations: int = 200,
    image_size: int = 32,
    snapshot_interval: int = 10,
    n_clusters: int = 3,
    seed: int = None
) -> list[SnapshotMetrics]:
    """Run nested sampling and track weight-space clustering at each snapshot."""

    if seed is not None:
        set_global_seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({'cppn': cppn, 'image': img, 'order': order})

    snapshots = []

    for iteration in range(n_iterations):
        # Record snapshot
        if iteration % snapshot_interval == 0 or iteration == n_iterations - 1:
            # Extract weight vectors
            weights = np.array([lp['cppn'].get_weights() for lp in live_points])
            orders = np.array([lp['order'] for lp in live_points])

            # Compute clustering metrics
            if weights.shape[0] >= n_clusters and weights.shape[1] > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(weights)

                # Silhouette score measures cluster separation
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(weights, labels)
                else:
                    sil = 0.0

                inertia = kmeans.inertia_
            else:
                sil = 0.0
                inertia = 0.0

            snapshots.append(SnapshotMetrics(
                iteration=iteration,
                silhouette=sil,
                inertia=inertia,
                mean_order=np.mean(orders),
                order_variance=np.var(orders)
            ))

        # Standard nested sampling step
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']

        # Select seed (any point except worst that meets threshold)
        valid_seeds = [i for i in range(n_live)
                       if i != worst_idx and live_points[i]['order'] >= threshold]
        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = np.random.choice(valid_seeds)

        # Replace worst via elliptical slice sampling
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx]['cppn'], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = {'cppn': new_cppn, 'image': new_img, 'order': new_order}
        else:
            # If ESS fails, clone a valid point with small perturbation
            new_cppn = CPPN()
            old_weights = live_points[seed_idx]['cppn'].get_weights()
            # Small perturbation
            new_weights = old_weights + np.random.randn(len(old_weights)) * 0.1
            new_cppn.set_weights(new_weights)
            new_img = new_cppn.render(image_size)
            new_order = order_multiplicative(new_img)
            if new_order >= threshold:
                live_points[worst_idx] = {'cppn': new_cppn, 'image': new_img, 'order': new_order}

    return snapshots


def analyze_monotonicity(values: list[float]) -> tuple[float, float]:
    """
    Analyze monotonic increasing trend.
    Returns: (monotonicity, p-value from Spearman correlation)
    """
    n = len(values)
    if n < 3:
        return 0.0, 1.0

    x = np.arange(n)
    rho, p = stats.spearmanr(x, values)

    # Count increasing pairs
    increases = sum(1 for i in range(n-1) if values[i+1] > values[i])
    monotonicity = increases / (n - 1)

    return monotonicity, p


def main():
    print("="*70)
    print("RES-118: Nested Sampling Cluster Formation")
    print("="*70)
    print()
    print("Hypothesis: Live points form clusters in weight space with")
    print("separation (silhouette score) increasing monotonically over iterations.")
    print()

    n_runs = 10
    n_live = 50
    n_iterations = 200
    snapshot_interval = 20
    n_clusters = 3

    all_monotonicity = []
    all_initial_sil = []
    all_final_sil = []
    all_sil_increase = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", end=" ", flush=True)

        snapshots = run_nested_sampling_with_snapshots(
            n_live=n_live,
            n_iterations=n_iterations,
            snapshot_interval=snapshot_interval,
            n_clusters=n_clusters,
            seed=run * 123
        )

        silhouettes = [s.silhouette for s in snapshots]

        mono, p = analyze_monotonicity(silhouettes)
        all_monotonicity.append(mono)

        all_initial_sil.append(silhouettes[0])
        all_final_sil.append(silhouettes[-1])
        all_sil_increase.append(silhouettes[-1] - silhouettes[0])

        print(f"mono={mono:.2f}, initial_sil={silhouettes[0]:.3f}, final_sil={silhouettes[-1]:.3f}")

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)

    # Test 1: Does silhouette increase over time?
    mean_initial = np.mean(all_initial_sil)
    mean_final = np.mean(all_final_sil)
    mean_increase = np.mean(all_sil_increase)
    std_increase = np.std(all_sil_increase)

    # Paired t-test for final > initial
    t_stat, p_paired = stats.ttest_rel(all_final_sil, all_initial_sil, alternative='greater')

    # Effect size (Cohen's d for paired samples)
    d_increase = mean_increase / std_increase if std_increase > 0 else 0

    print(f"\n1. Silhouette Score Change:")
    print(f"   Initial: {mean_initial:.3f} +/- {np.std(all_initial_sil):.3f}")
    print(f"   Final:   {mean_final:.3f} +/- {np.std(all_final_sil):.3f}")
    print(f"   Increase: {mean_increase:.3f} +/- {std_increase:.3f}")
    print(f"   Paired t-test: t={t_stat:.2f}, p={p_paired:.4f}")
    print(f"   Effect size (Cohen's d): {d_increase:.2f}")

    # Test 2: Is the increase monotonic?
    mean_mono = np.mean(all_monotonicity)

    # Test monotonicity > 0.5 (better than random)
    t_mono, p_mono = stats.ttest_1samp(all_monotonicity, 0.5, alternative='greater')
    d_mono = (mean_mono - 0.5) / np.std(all_monotonicity) if np.std(all_monotonicity) > 0 else 0

    print(f"\n2. Monotonicity of Silhouette Increase:")
    print(f"   Mean monotonicity: {mean_mono:.3f} +/- {np.std(all_monotonicity):.3f}")
    print(f"   (1.0 = always increasing, 0.5 = random)")
    print(f"   t-test vs 0.5: t={t_mono:.2f}, p={p_mono:.4f}")
    print(f"   Effect size: d={d_mono:.2f}")

    # Verdict
    print()
    print("="*70)
    print("VERDICT")
    print("="*70)

    # Primary hypothesis: silhouette increases monotonically
    # Requires: p < 0.01 for increase AND monotonicity > 0.5 significantly

    sil_validated = p_paired < 0.01 and d_increase > 0.5
    mono_validated = p_mono < 0.01 and d_mono > 0.5

    print(f"\nSilhouette increase: p={p_paired:.4f}, d={d_increase:.2f}")
    print(f"  {'VALIDATED' if sil_validated else 'NOT VALIDATED'} (requires p<0.01, d>0.5)")

    print(f"\nMonotonic increase: p={p_mono:.4f}, d={d_mono:.2f}")
    print(f"  {'VALIDATED' if mono_validated else 'NOT VALIDATED'} (requires p<0.01, d>0.5)")

    overall = sil_validated and mono_validated
    status = "validated" if overall else ("refuted" if (p_paired > 0.05 or mean_increase <= 0) else "inconclusive")

    print(f"\nOVERALL STATUS: {status.upper()}")
    print()

    # Summary for log
    print("="*70)
    print("LOG SUMMARY")
    print("="*70)
    print(f"Status: {status}")
    print(f"Silhouette: {mean_initial:.3f} -> {mean_final:.3f} (d={d_increase:.2f}, p={p_paired:.4f})")
    print(f"Monotonicity: {mean_mono:.2f} (d={d_mono:.2f}, p={p_mono:.4f})")

    return status, {
        'effect_size_increase': d_increase,
        'effect_size_mono': d_mono,
        'p_increase': p_paired,
        'p_mono': p_mono,
        'mean_initial_sil': mean_initial,
        'mean_final_sil': mean_final,
        'monotonicity': mean_mono
    }


if __name__ == "__main__":
    status, metrics = main()
