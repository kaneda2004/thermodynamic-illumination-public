"""
RES-109: CPPN Output Diversity Analysis (v2 - refined)

The initial analysis showed contradictory results between silhouette (K=2)
and bootstrap (K=24). This suggests the data may be:
1. Continuously distributed (no clear clusters)
2. Or has some clusters but with fuzzy boundaries

Let's use multiple complementary methods to assess true diversity.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, compute_compressibility, compute_edge_density,
    compute_spectral_coherence, compute_symmetry
)

def structural_features(img):
    """Extract only structural features (not raw pixels)."""
    compress = compute_compressibility(img)
    edge = compute_edge_density(img)
    spectral = compute_spectral_coherence(img)
    sym = compute_symmetry(img)
    density = np.mean(img)

    # Quadrant densities
    h, w = img.shape
    q1 = np.mean(img[:h//2, :w//2])
    q2 = np.mean(img[:h//2, w//2:])
    q3 = np.mean(img[h//2:, :w//2])
    q4 = np.mean(img[h//2:, w//2:])

    # Row and column means for profile diversity
    row_var = np.var(np.mean(img, axis=1))
    col_var = np.var(np.mean(img, axis=0))

    return np.array([compress, edge, spectral, sym, density, q1, q2, q3, q4, row_var, col_var])

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
        features.append(structural_features(img))

    X = np.array(features)
    print(f"Structural feature matrix: {X.shape}")

    # Filter degenerate (all black/white)
    densities = X[:, 4]  # density is index 4
    valid_mask = (0.01 < densities) & (densities < 0.99)
    n_valid = np.sum(valid_mask)
    print(f"Non-degenerate samples: {n_valid}/{n_samples}")

    X_valid = X[valid_mask]
    images_valid = [img for img, m in zip(images, valid_mask) if m]

    # Normalize features
    X_norm = (X_valid - X_valid.mean(axis=0)) / (X_valid.std(axis=0) + 1e-10)

    print("\n=== Method 1: Pairwise Diversity ===")
    # Compute average pairwise distance
    dists = pdist(X_norm, metric='euclidean')
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    print(f"Mean pairwise distance: {mean_dist:.3f}")
    print(f"Std pairwise distance: {std_dist:.3f}")

    # Compare to uniform distribution
    X_uniform = np.random.randn(n_valid, X_norm.shape[1])
    dists_uniform = pdist(X_uniform, metric='euclidean')
    mean_uniform = np.mean(dists_uniform)

    # If CPPN diversity is similar to uniform, they're maximally diverse
    diversity_ratio = mean_dist / mean_uniform
    print(f"Diversity ratio (vs uniform): {diversity_ratio:.3f}")

    print("\n=== Method 2: Hopkins Statistic ===")
    # Hopkins statistic: measures clustering tendency
    # H close to 0.5 = uniform, H close to 1 = clustered
    def hopkins_statistic(X, n_points=50):
        n = len(X)
        n_points = min(n_points, n)

        # Sample n_points from X
        sample_idx = np.random.choice(n, n_points, replace=False)
        X_sample = X[sample_idx]

        # Generate n_points uniform random points
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_random = np.random.uniform(X_min, X_max, (n_points, X.shape[1]))

        # Nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)

        # Distance from random points to nearest data point
        d_random, _ = nbrs.kneighbors(X_random)
        u = d_random[:, 0].sum()

        # Distance from sample to nearest other data point (use 2nd neighbor since 1st is itself)
        d_sample, _ = nbrs.kneighbors(X_sample)
        w = d_sample[:, 1].sum()

        return u / (u + w)

    hopkins_scores = [hopkins_statistic(X_norm) for _ in range(30)]
    hopkins_mean = np.mean(hopkins_scores)
    hopkins_std = np.std(hopkins_scores)
    print(f"Hopkins statistic: {hopkins_mean:.3f} +/- {hopkins_std:.3f}")
    print("  (0.5 = uniform/no clusters, >0.7 = clustered)")

    print("\n=== Method 3: K-means with Multiple Metrics ===")
    k_range = [2, 3, 5, 8, 10, 15, 20, 25, 30]
    results_kmeans = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)

        sil = silhouette_score(X_norm, labels)
        ch = calinski_harabasz_score(X_norm, labels)
        inertia = kmeans.inertia_

        # Count clusters with >5% of data
        _, counts = np.unique(labels, return_counts=True)
        significant = sum(c > 0.05 * n_valid for c in counts)

        results_kmeans.append({
            'k': k, 'silhouette': sil, 'calinski': ch,
            'inertia': inertia, 'significant_clusters': significant
        })
        print(f"  K={k:2d}: sil={sil:.3f}, CH={ch:.1f}, sig_clusters={significant}")

    print("\n=== Method 4: Hierarchical Clustering ===")
    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(X_norm, method='ward')

    # Try different cut heights
    for n_clusters in [5, 10, 15, 20]:
        labels_hier = fcluster(Z, n_clusters, criterion='maxclust')
        sil_hier = silhouette_score(X_norm, labels_hier)
        _, counts = np.unique(labels_hier, return_counts=True)
        significant = sum(c > 0.05 * n_valid for c in counts)
        print(f"  n_clust={n_clusters}: silhouette={sil_hier:.3f}, significant={significant}")

    print("\n=== Method 5: Local Intrinsic Dimension ===")
    # Estimate intrinsic dimension using nearest neighbors
    def estimate_intrinsic_dim(X, k=10):
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        # Use ratio of k-th to 1st neighbor distance
        dims = []
        for i in range(len(X)):
            if distances[i, 1] > 0:
                ratio = distances[i, k] / distances[i, 1]
                dim = np.log(k) / np.log(ratio + 1e-10)
                if 0 < dim < X.shape[1]:
                    dims.append(dim)
        return np.mean(dims), np.std(dims)

    intrinsic_dim, intrinsic_std = estimate_intrinsic_dim(X_norm)
    print(f"Local intrinsic dimension: {intrinsic_dim:.2f} +/- {intrinsic_std:.2f}")
    print(f"  (Feature space dim: {X_norm.shape[1]})")

    print("\n=== Method 6: Feature Range Coverage ===")
    # How much of the feature space do CPPNs cover?
    for i, name in enumerate(['compress', 'edge', 'spectral', 'sym', 'density']):
        feature_vals = X_valid[:, i]
        coverage = (feature_vals.max() - feature_vals.min())
        print(f"  {name}: range=[{feature_vals.min():.3f}, {feature_vals.max():.3f}], std={feature_vals.std():.3f}")

    print("\n=== Method 7: Unique Pattern Hashes ===")
    # Downsample and hash to count truly unique patterns
    def hash_pattern(img, downsample=8):
        # Downsample to 8x8
        step = img.shape[0] // downsample
        small = img[::step, ::step][:downsample, :downsample]
        return tuple(small.flatten())

    hashes = [hash_pattern(img) for img in images_valid]
    unique_patterns = len(set(hashes))
    print(f"Unique 8x8 patterns: {unique_patterns}/{n_valid} ({100*unique_patterns/n_valid:.1f}%)")

    # At 4x4 resolution
    hashes_4 = [hash_pattern(img, 4) for img in images_valid]
    unique_4 = len(set(hashes_4))
    print(f"Unique 4x4 patterns: {unique_4}/{n_valid} ({100*unique_4/n_valid:.1f}%)")

    # Summary and statistical test
    print("\n" + "="*60)
    print("SUMMARY: CPPN OUTPUT DIVERSITY")
    print("="*60)

    # Key metrics
    print(f"\nKey findings:")
    print(f"  1. Hopkins statistic: {hopkins_mean:.3f} (0.5=uniform, >0.7=clustered)")
    print(f"  2. Diversity ratio vs uniform: {diversity_ratio:.3f}")
    print(f"  3. Unique 8x8 patterns: {unique_patterns}/{n_valid} ({100*unique_patterns/n_valid:.1f}%)")
    print(f"  4. Intrinsic dimension: {intrinsic_dim:.1f}/{X_norm.shape[1]} features")
    print(f"  5. Best silhouette K: {k_range[np.argmax([r['silhouette'] for r in results_kmeans])]}")

    # The real question: are outputs diverse or collapsed?
    # Collapsed would mean: high Hopkins (>0.7), few unique patterns, low diversity ratio

    # Test: Is Hopkins significantly different from 0.5 (uniform)?
    t_hopkins, p_hopkins = stats.ttest_1samp(hopkins_scores, 0.5)
    print(f"\nHopkins test (H0: uniform at 0.5):")
    print(f"  t={t_hopkins:.3f}, p={p_hopkins:.4f}")
    print(f"  Hopkins {hopkins_mean:.3f} {'>' if hopkins_mean > 0.5 else '<'} 0.5")

    # Effect size for clustering tendency
    cohens_d_hopkins = (hopkins_mean - 0.5) / hopkins_std

    # Verdict
    # If Hopkins < 0.5: data is MORE uniform than random (very diverse)
    # If Hopkins ~ 0.5: data is uniformly distributed (diverse)
    # If Hopkins > 0.7: data is clearly clustered (less diverse)

    is_diverse = hopkins_mean < 0.65  # Not strongly clustered
    unique_ratio_high = unique_patterns / n_valid > 0.8  # Most patterns unique

    print(f"\n=== VERDICT ===")
    print(f"CPPNs generate {'DIVERSE' if is_diverse else 'MODE-COLLAPSED'} outputs")
    print(f"  - Hopkins {hopkins_mean:.3f} < 0.65: {hopkins_mean < 0.65}")
    print(f"  - Unique pattern ratio {unique_patterns/n_valid:.2f} > 0.8: {unique_ratio_high}")

    # The hypothesis was about >10 distinct clusters
    # But if data is continuously distributed (Hopkins~0.5), clusters aren't meaningful
    # The better conclusion is: diversity is high, patterns form a continuum

    print(f"\n=== CONCLUSION ===")
    if hopkins_mean < 0.55:
        print("CPPNs generate a CONTINUOUS DIVERSITY of patterns (no discrete modes)")
        print("Outputs are uniformly distributed in feature space")
        status = "refuted"  # Original hypothesis about >10 modes is wrong framing
    elif hopkins_mean > 0.65:
        print("CPPNs show DISCRETE CLUSTERING into distinct pattern types")
        status = "validated" if unique_patterns > 50 else "refuted"
    else:
        print("CPPNs show MODERATE STRUCTURE with some clustering tendency")
        status = "inconclusive"

    # Effect size: how different from uniform?
    effect_size = abs(cohens_d_hopkins)
    p_value = p_hopkins

    print(f"\nFinal statistics:")
    print(f"  p-value (Hopkins != 0.5): {p_value:.6f}")
    print(f"  Effect size (Cohen's d): {effect_size:.3f}")
    print(f"  Status: {status.upper()}")

    return {
        'hopkins_mean': hopkins_mean,
        'hopkins_std': hopkins_std,
        'diversity_ratio': diversity_ratio,
        'unique_8x8_patterns': unique_patterns,
        'unique_ratio': unique_patterns / n_valid,
        'intrinsic_dim': intrinsic_dim,
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'status': status
    }

if __name__ == '__main__':
    results = main()
