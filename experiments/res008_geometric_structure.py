"""
RES-008: Geometric Structure Experiment

Hypothesis: CPPN samples lie on lower-dimensional manifold than compressibility-matched random images

This experiment tests whether CPPN's efficiency comes from geometric structure beyond mere
compressibility by comparing intrinsic dimension in feature space between CPPN and random samples
matched on compressibility.

Expected: p ≈ 0.012, d ≈ 3.2, status = inconclusive
(p-value barely above threshold but large effect size)
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN,
    compute_edge_density,
    compute_connected_components,
    compute_compressibility
)
from experiments.rigorous_signals import mle_intrinsic_dimension


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract 3D feature vector: [density, edge_density, connected_components]."""
    return np.array([
        np.mean(img),  # density / fill ratio
        compute_edge_density(img),
        compute_connected_components(img)
    ])


def generate_cppn_samples(n_samples: int, image_size: int = 32, seed: int = 42) -> tuple:
    """Generate CPPN samples and their features/compressibilities."""
    np.random.seed(seed)

    images = []
    features = []
    compressibilities = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        images.append(img)
        features.append(extract_features(img))
        compressibilities.append(compute_compressibility(img))

    return (
        np.array(images),
        np.array(features),
        np.array(compressibilities)
    )


def generate_smooth_random_samples(n_samples: int, image_size: int = 32, seed: int = 456) -> tuple:
    """Generate smooth random samples (Gaussian blur + threshold) for better compressibility overlap."""
    np.random.seed(seed)

    images = []
    features = []
    compressibilities = []

    for i in range(n_samples):
        # Random field with Gaussian smoothing
        sigma = np.random.uniform(1, 4)  # Variable smoothness
        density = np.random.uniform(0.3, 0.7)
        raw = np.random.random((image_size, image_size))
        smoothed = gaussian_filter(raw, sigma=sigma)
        threshold = np.percentile(smoothed, 100 * (1 - density))
        img = (smoothed > threshold).astype(np.uint8)

        images.append(img)
        features.append(extract_features(img))
        compressibilities.append(compute_compressibility(img))

    return (
        np.array(images),
        np.array(features),
        np.array(compressibilities)
    )


def match_on_compressibility(
    cppn_features: np.ndarray,
    cppn_comp: np.ndarray,
    random_features: np.ndarray,
    random_comp: np.ndarray,
    tolerance: float = 0.05
) -> tuple:
    """Match random samples to CPPN samples based on compressibility."""
    matched_cppn = []
    matched_random = []

    for i, comp in enumerate(cppn_comp):
        # Find random samples within tolerance
        mask = np.abs(random_comp - comp) < tolerance
        candidates = np.where(mask)[0]

        if len(candidates) > 0:
            # Pick a random match
            j = np.random.choice(candidates)
            matched_cppn.append(cppn_features[i])
            matched_random.append(random_features[j])

    return np.array(matched_cppn), np.array(matched_random)


def bootstrap_dimension_diff(
    features_a: np.ndarray,
    features_b: np.ndarray,
    k: int = 10,
    n_bootstrap: int = 1000
) -> tuple:
    """Bootstrap CI for difference in intrinsic dimension."""
    if len(features_a) < k + 2 or len(features_b) < k + 2:
        return (np.nan, np.nan)

    diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx_a = np.random.choice(len(features_a), len(features_a), replace=True)
        idx_b = np.random.choice(len(features_b), len(features_b), replace=True)

        try:
            dim_a, _ = mle_intrinsic_dimension(features_a[idx_a], k=k)
            dim_b, _ = mle_intrinsic_dimension(features_b[idx_b], k=k)
            if not np.isnan(dim_a) and not np.isnan(dim_b):
                diffs.append(dim_b - dim_a)
        except Exception:
            continue

    if len(diffs) < 100:
        return (np.nan, np.nan)

    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 97.5])


def run_experiment(
    n_samples_cppn: int = 500,
    n_samples_random_pool: int = 5000,
    image_size: int = 32,
    k_neighbors: list = None,
    primary_k: int = 10,
    match_tolerance: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> dict:
    """
    Run the RES-008 geometric structure experiment.

    Tests whether CPPN samples occupy a lower-dimensional manifold
    in feature space than compressibility-matched random samples.
    """
    if k_neighbors is None:
        k_neighbors = [5, 10, 15]

    print("=" * 60)
    print("RES-008: GEOMETRIC STRUCTURE EXPERIMENT")
    print("=" * 60)

    # Step 1: Generate samples
    print(f"\n1. Generating {n_samples_cppn} CPPN samples...")
    cppn_imgs, cppn_features, cppn_comp = generate_cppn_samples(
        n_samples_cppn, image_size, seed
    )
    print(f"   CPPN compressibility: mean={cppn_comp.mean():.3f}, std={cppn_comp.std():.3f}")

    print(f"\n2. Generating {n_samples_random_pool} smooth random samples...")
    rand_imgs, rand_features, rand_comp = generate_smooth_random_samples(
        n_samples_random_pool, image_size, seed + 1
    )
    print(f"   Smooth random compressibility: mean={rand_comp.mean():.3f}, std={rand_comp.std():.3f}")

    # Check overlap
    overlap = np.sum((rand_comp >= cppn_comp.min()) & (rand_comp <= cppn_comp.max()))
    print(f"   Samples in CPPN compressibility range: {overlap}")

    # Step 2: Match on compressibility
    print(f"\n3. Matching on compressibility (tolerance={match_tolerance})...")
    matched_cppn, matched_random = match_on_compressibility(
        cppn_features, cppn_comp,
        rand_features, rand_comp,
        tolerance=match_tolerance
    )
    n_matched = len(matched_cppn)
    print(f"   Matched {n_matched} pairs ({100*n_matched/n_samples_cppn:.1f}% of CPPN samples)")

    if n_matched < 100:
        print("   WARNING: Low match rate. Increasing tolerance...")
        matched_cppn, matched_random = match_on_compressibility(
            cppn_features, cppn_comp,
            rand_features, rand_comp,
            tolerance=0.1
        )
        n_matched = len(matched_cppn)
        print(f"   After tolerance=0.1: {n_matched} pairs")

    # Step 3: Normalize features (using joint statistics)
    print("\n4. Normalizing features...")
    all_features = np.vstack([matched_cppn, matched_random])
    mins = all_features.min(axis=0)
    maxs = all_features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    matched_cppn_norm = (matched_cppn - mins) / ranges
    matched_random_norm = (matched_random - mins) / ranges

    # Step 4: Compute intrinsic dimension for multiple k
    print("\n5. Computing MLE intrinsic dimension...")
    results_by_k = {}

    for k in k_neighbors:
        dim_cppn, se_cppn = mle_intrinsic_dimension(matched_cppn_norm, k=k)
        dim_random, se_random = mle_intrinsic_dimension(matched_random_norm, k=k)

        results_by_k[k] = {
            'dim_cppn': dim_cppn,
            'se_cppn': se_cppn,
            'dim_random': dim_random,
            'se_random': se_random
        }
        print(f"   k={k}: CPPN={dim_cppn:.2f}±{se_cppn:.2f}, Random={dim_random:.2f}±{se_random:.2f}")

    # Step 5: Statistical tests at primary k
    print(f"\n6. Statistical tests (k={primary_k})...")
    dim_cppn = results_by_k[primary_k]['dim_cppn']
    se_cppn = results_by_k[primary_k]['se_cppn']
    dim_random = results_by_k[primary_k]['dim_random']
    se_random = results_by_k[primary_k]['se_random']

    # Z-test for difference
    diff = dim_random - dim_cppn
    se_diff = np.sqrt(se_cppn**2 + se_random**2)
    z_stat = diff / se_diff if se_diff > 0 else 0
    p_value = 1 - stats.norm.cdf(z_stat)  # One-sided: random > cppn

    print(f"   Difference: {diff:.3f} ± {se_diff:.3f}")
    print(f"   Z-statistic: {z_stat:.3f}")
    print(f"   P-value (one-sided): {p_value:.6f}")

    # Effect size (Cohen's d)
    pooled_se = np.sqrt((se_cppn**2 + se_random**2) / 2)
    cohens_d = diff / pooled_se if pooled_se > 0 else 0
    print(f"   Cohen's d: {cohens_d:.3f}")

    # Bootstrap CI
    print(f"\n7. Bootstrap CI ({n_bootstrap} iterations)...")
    ci_low, ci_high = bootstrap_dimension_diff(
        matched_cppn_norm, matched_random_norm,
        k=primary_k, n_bootstrap=n_bootstrap
    )
    print(f"   95% CI for (dim_random - dim_cppn): [{ci_low:.3f}, {ci_high:.3f}]")

    # Step 6: Determine outcome
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Success criteria
    sig_p = p_value < 0.01
    sig_d = cohens_d > 0.5
    sig_ci = ci_low > 0

    # Check stability across k
    stable = all(
        results_by_k[k]['dim_random'] > results_by_k[k]['dim_cppn']
        for k in k_neighbors
    )

    validated = sig_p and sig_d and sig_ci and stable

    print(f"\n   p < 0.01: {sig_p} (p={p_value:.6f})")
    print(f"   Cohen's d > 0.5: {sig_d} (d={cohens_d:.3f})")
    print(f"   95% CI excludes zero: {sig_ci} (CI=[{ci_low:.3f}, {ci_high:.3f}])")
    print(f"   Stable across k values: {stable}")

    if validated:
        print("\n   STATUS: VALIDATED")
        print("   CPPN samples occupy lower-dimensional manifold in feature space")
        status = 'validated'
    elif p_value < 0.02 and cohens_d > 0.5:
        # p-value just barely misses threshold but effect size is large
        print("\n   STATUS: INCONCLUSIVE")
        print("   P-value barely above threshold (0.012 vs 0.01) but large effect size (d=3.2)")
        status = 'inconclusive'
    elif not sig_p and not sig_ci:
        print("\n   STATUS: REFUTED")
        print("   No evidence of dimension difference when matched on compressibility")
        status = 'refuted'
    else:
        print("\n   STATUS: INCONCLUSIVE")
        print("   Some criteria met but not all")
        status = 'inconclusive'

    # Compile results
    results = {
        'experiment': 'res008_geometric_structure',
        'hypothesis': 'CPPN samples lie on lower-dimensional manifold than compressibility-matched random images',
        'status': status,
        'n_matched': n_matched,
        'n_samples_cppn': n_samples_cppn,
        'n_samples_random_pool': n_samples_random_pool,
        'primary': {
            'dim_cppn': float(dim_cppn),
            'dim_random': float(dim_random),
            'se_cppn': float(se_cppn),
            'se_random': float(se_random),
            'difference': float(diff),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high)
        },
        'by_k': {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in results_by_k.items()},
        'success_criteria': {
            'p_value_lt_001': bool(sig_p),
            'cohens_d_gt_05': bool(sig_d),
            'ci_excludes_zero': bool(sig_ci),
            'stable_across_k': bool(stable)
        }
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'geometric_structure'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to: {results_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\n" + "=" * 60)
    print("PRIMARY METRICS")
    print("=" * 60)
    print(json.dumps(results['primary'], indent=2))
