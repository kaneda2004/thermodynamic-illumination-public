"""
RES-109: CPPN Output Diversity Analysis

Hypothesis: CPPNs generate diverse output patterns rather than collapsing
to a limited number of modes, as measured by clustering analysis showing
>10 distinct pattern types across 500 random CPPNs.

Method:
1. Generate 500 random CPPNs with default architecture
2. Render each to 32x32 binary images
3. Extract features (flatten pixels + structural features)
4. Apply K-means clustering with varying K
5. Use silhouette score and elbow method to find optimal K
6. Test if optimal K > 10 with statistical significance
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, compute_compressibility, compute_edge_density,
    compute_spectral_coherence, compute_symmetry
)

def extract_features(img):
    """Extract rich feature vector from binary image."""
    # Flatten pixels (normalized)
    flat = img.flatten().astype(float)

    # Structural features
    compress = compute_compressibility(img)
    edge = compute_edge_density(img)
    spectral = compute_spectral_coherence(img)
    sym = compute_symmetry(img)

    # Density
    density = np.mean(img)

    # Quadrant densities (captures spatial distribution)
    h, w = img.shape
    q1 = np.mean(img[:h//2, :w//2])
    q2 = np.mean(img[:h//2, w//2:])
    q3 = np.mean(img[h//2:, :w//2])
    q4 = np.mean(img[h//2:, w//2:])

    return np.concatenate([
        flat,  # 1024 dims
        [compress, edge, spectral, sym, density, q1, q2, q3, q4]  # 9 dims
    ])

def main():
    np.random.seed(42)

    n_samples = 500
    print(f"Generating {n_samples} random CPPNs...")

    # Generate samples
    images = []
    features = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=32)
        images.append(img)
        features.append(extract_features(img))
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples}")

    X = np.array(features)
    print(f"Feature matrix shape: {X.shape}")

    # Filter out degenerate images (all black or all white)
    densities = [np.mean(img) for img in images]
    valid_mask = [(0.01 < d < 0.99) for d in densities]
    n_valid = sum(valid_mask)
    print(f"Valid images (non-degenerate): {n_valid}/{n_samples}")

    X_valid = X[valid_mask]

    # Reduce dimensionality for clustering
    print("\nReducing to 50 PCA components...")
    pca = PCA(n_components=min(50, X_valid.shape[0]-1))
    X_pca = pca.fit_transform(X_valid)
    variance_explained = sum(pca.explained_variance_ratio_)
    print(f"Variance explained: {variance_explained:.1%}")

    # Try different K values
    k_range = range(2, 31)
    silhouettes = []
    inertias = []

    print("\nClustering with K-means (K=2 to 30)...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        silhouettes.append(sil)
        inertias.append(kmeans.inertia_)

        if k % 5 == 0:
            print(f"  K={k}: silhouette={sil:.3f}")

    # Find optimal K by silhouette
    optimal_k_sil = k_range[np.argmax(silhouettes)]
    max_silhouette = max(silhouettes)

    # Elbow method: find where inertia curve bends
    # Use second derivative of normalized inertia
    inertias_norm = np.array(inertias) / inertias[0]
    diffs = np.diff(inertias_norm)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2  # +2 because we lost 2 points from derivatives
    optimal_k_elbow = list(k_range)[elbow_idx]

    print(f"\n=== Results ===")
    print(f"Optimal K (silhouette): {optimal_k_sil} (score={max_silhouette:.3f})")
    print(f"Optimal K (elbow): {optimal_k_elbow}")

    # Use silhouette-optimal K for analysis
    best_k = optimal_k_sil
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_best = kmeans_best.fit_predict(X_pca)

    # Cluster size distribution
    unique, counts = np.unique(labels_best, return_counts=True)
    print(f"\nCluster sizes at K={best_k}:")
    for c, n in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  Cluster {c}: {n} images ({100*n/len(labels_best):.1f}%)")

    # Test hypothesis: does K > 10 have significantly better silhouette than K=10?
    # Use bootstrap to get confidence interval on optimal K
    print("\n=== Bootstrap Analysis ===")
    n_bootstrap = 100
    optimal_ks = []

    for b in range(n_bootstrap):
        # Resample
        idx = np.random.choice(len(X_pca), len(X_pca), replace=True)
        X_boot = X_pca[idx]

        # Find best K
        best_sil = -1
        best_k_boot = 2
        for k in [5, 10, 15, 20, 25]:  # Subset for speed
            kmeans = KMeans(n_clusters=k, random_state=b, n_init=5)
            labels = kmeans.fit_predict(X_boot)
            sil = silhouette_score(X_boot, labels)
            if sil > best_sil:
                best_sil = sil
                best_k_boot = k
        optimal_ks.append(best_k_boot)

    mean_k = np.mean(optimal_ks)
    std_k = np.std(optimal_ks)
    ci_low = np.percentile(optimal_ks, 2.5)
    ci_high = np.percentile(optimal_ks, 97.5)

    print(f"Bootstrap optimal K: {mean_k:.1f} +/- {std_k:.1f}")
    print(f"95% CI: [{ci_low}, {ci_high}]")

    # Statistical test: is optimal K > 10?
    # One-sample t-test against null hypothesis K <= 10
    t_stat, p_value_2sided = stats.ttest_1samp(optimal_ks, 10)
    # One-sided p-value (testing if mean > 10)
    p_value = p_value_2sided / 2 if t_stat > 0 else 1 - p_value_2sided / 2

    print(f"\nTest: Optimal K > 10")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value (one-sided): {p_value:.4f}")

    # Effect size (Cohen's d for difference from 10)
    cohens_d = (mean_k - 10) / std_k
    print(f"Cohen's d: {cohens_d:.3f}")

    # Alternative metric: count distinct modes with minimum representation
    # A "true" cluster should have at least 2% of samples
    min_cluster_size = int(0.02 * n_valid)
    significant_clusters = sum(c >= min_cluster_size for c in counts)
    print(f"\nDistinct modes (>2% representation): {significant_clusters}")

    # Gap statistic for optimal K
    print("\n=== Gap Statistic ===")
    def compute_gap(X, k, n_ref=10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(X)
        W_k = kmeans.inertia_

        # Reference distribution (uniform in bounding box)
        W_refs = []
        for _ in range(n_ref):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans_ref.fit(X_ref)
            W_refs.append(kmeans_ref.inertia_)

        gap = np.mean(np.log(W_refs)) - np.log(W_k)
        s_k = np.std(np.log(W_refs)) * np.sqrt(1 + 1/n_ref)
        return gap, s_k

    gaps = []
    s_ks = []
    for k in [5, 10, 15, 20, 25, 30]:
        gap, s_k = compute_gap(X_pca, k)
        gaps.append((k, gap, s_k))
        print(f"  K={k}: gap={gap:.3f}, s_k={s_k:.3f}")

    # Find optimal by gap statistic (first k where gap(k) >= gap(k+1) - s(k+1))
    optimal_k_gap = gaps[0][0]
    for i in range(len(gaps)-1):
        k, gap, s = gaps[i]
        k_next, gap_next, s_next = gaps[i+1]
        if gap >= gap_next - s_next:
            optimal_k_gap = k
            break
    print(f"Optimal K (gap statistic): {optimal_k_gap}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Valid samples: {n_valid}/{n_samples}")
    print(f"Optimal K (silhouette): {optimal_k_sil}")
    print(f"Optimal K (elbow): {optimal_k_elbow}")
    print(f"Optimal K (bootstrap mean): {mean_k:.1f}")
    print(f"Optimal K (gap statistic): {optimal_k_gap}")
    print(f"Distinct modes (>2% rep): {significant_clusters}")
    print(f"p-value (K>10): {p_value:.4f}")
    print(f"Effect size: {cohens_d:.3f}")

    # Verdict
    hypothesis_supported = (optimal_k_sil > 10 or mean_k > 10) and p_value < 0.01 and abs(cohens_d) > 0.5
    print(f"\nHypothesis (K>10) supported: {hypothesis_supported}")

    return {
        'n_samples': n_samples,
        'n_valid': n_valid,
        'optimal_k_silhouette': optimal_k_sil,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_bootstrap_mean': mean_k,
        'optimal_k_gap': optimal_k_gap,
        'significant_clusters': significant_clusters,
        'p_value': p_value,
        'effect_size': cohens_d,
        'hypothesis_supported': hypothesis_supported
    }

if __name__ == '__main__':
    results = main()
