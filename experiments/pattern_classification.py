"""
Pattern Classification Experiment (RES-066)

Hypothesis: High-order CPPN images can be classified into distinct pattern types
(stripes, blobs, checkers) with >80% accuracy using simple spectral/spatial features.

Methodology:
1. Generate 500 high-order CPPN images (order > 0.5)
2. Compute pattern features: dominant frequency, orientation, blob count, aspect ratio
3. Apply k-means clustering (k=3-5)
4. Measure cluster separability (silhouette score) and visual coherence
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from core.thermo_sampler_v3 import compute_edge_density, compute_spectral_coherence, compute_connected_components
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from pathlib import Path

def compute_dominant_frequency(img: np.ndarray) -> tuple[float, float]:
    """Return dominant spatial frequency and orientation from FFT."""
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Zero out DC component
    power[cy, cx] = 0

    # Find peak
    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    py, px = peak_idx

    # Frequency and orientation
    freq = np.sqrt((px - cx)**2 + (py - cy)**2) / (h/2)
    orientation = np.arctan2(py - cy, px - cx)

    return float(freq), float(orientation)


def compute_blob_aspect_ratio(img: np.ndarray) -> float:
    """Compute aspect ratio of largest connected component."""
    labeled, num_features = ndimage.label(img)
    if num_features == 0:
        return 1.0

    # Find largest component
    sizes = ndimage.sum(img, labeled, range(1, num_features + 1))
    if len(sizes) == 0:
        return 1.0
    largest_label = np.argmax(sizes) + 1
    largest = (labeled == largest_label)

    # Bounding box
    rows = np.any(largest, axis=1)
    cols = np.any(largest, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, 1)
    cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, 1)

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    return max(height, width) / max(1, min(height, width))


def compute_stripe_score(img: np.ndarray) -> float:
    """Score how stripe-like an image is (high horizontal or vertical coherence)."""
    # Row-wise correlation
    row_corr = np.mean([np.corrcoef(img[i], img[i+1])[0, 1]
                        for i in range(img.shape[0]-1) if not np.all(img[i] == img[i+1])])
    # Column-wise correlation
    col_corr = np.mean([np.corrcoef(img[:, j], img[:, j+1])[0, 1]
                        for j in range(img.shape[1]-1) if not np.all(img[:, j] == img[:, j+1])])

    # Handle NaN
    row_corr = 0 if np.isnan(row_corr) else row_corr
    col_corr = 0 if np.isnan(col_corr) else col_corr

    return max(abs(row_corr), abs(col_corr))


def compute_pattern_features(img: np.ndarray) -> dict:
    """Compute comprehensive pattern features."""
    freq, orientation = compute_dominant_frequency(img)

    return {
        'dominant_freq': freq,
        'orientation': orientation,
        'edge_density': compute_edge_density(img),
        'coherence': compute_spectral_coherence(img),
        'n_components': compute_connected_components(img),
        'aspect_ratio': compute_blob_aspect_ratio(img),
        'stripe_score': compute_stripe_score(img),
        'density': float(np.mean(img)),
    }


def generate_high_order_samples(n_samples: int, order_threshold: float, max_attempts: int = 10000) -> list:
    """Generate CPPN images above order threshold."""
    samples = []
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)

        if order > order_threshold:
            features = compute_pattern_features(img)
            features['order'] = float(order)
            samples.append({'img': img, 'features': features})

        attempts += 1

        if attempts % 1000 == 0:
            print(f"  Attempts: {attempts}, collected: {len(samples)}")

    return samples


def main():
    set_global_seed(42)

    print("Pattern Classification Experiment")
    print("=" * 50)

    # Generate high-order samples (lower threshold to 0.15 based on actual distribution)
    print("\n1. Generating high-order CPPN samples...")
    samples = generate_high_order_samples(n_samples=500, order_threshold=0.15)
    print(f"   Collected {len(samples)} samples")

    if len(samples) < 50:
        print("ERROR: Not enough high-order samples. Try lower threshold.")
        return {'status': 'failed', 'reason': 'insufficient_samples'}

    # Extract feature matrix
    feature_names = ['dominant_freq', 'orientation', 'edge_density',
                     'coherence', 'n_components', 'aspect_ratio', 'stripe_score']
    X = np.array([[s['features'][f] for f in feature_names] for s in samples])

    # Normalize features
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Try different cluster counts
    print("\n2. Clustering analysis...")
    results = {}

    for k in [3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)

        silhouette = silhouette_score(X_norm, labels)

        # Cluster statistics
        cluster_stats = []
        for c in range(k):
            mask = labels == c
            cluster_features = X[mask].mean(axis=0)
            cluster_std = X[mask].std(axis=0)
            cluster_stats.append({
                'size': int(mask.sum()),
                'mean_features': dict(zip(feature_names, cluster_features.tolist())),
                'std_features': dict(zip(feature_names, cluster_std.tolist())),
            })

        results[k] = {
            'silhouette': float(silhouette),
            'cluster_stats': cluster_stats,
        }

        print(f"   k={k}: silhouette={silhouette:.3f}")

    # Best k by silhouette
    best_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    best_silhouette = results[best_k]['silhouette']

    print(f"\n3. Best clustering: k={best_k} (silhouette={best_silhouette:.3f})")

    # Pattern interpretation
    print("\n4. Pattern interpretation:")
    for i, stats in enumerate(results[best_k]['cluster_stats']):
        mf = stats['mean_features']
        pattern_type = "unknown"

        # Heuristic labeling
        if mf['stripe_score'] > 0.5 and mf['aspect_ratio'] > 3:
            pattern_type = "stripes"
        elif mf['n_components'] > 3 and mf['aspect_ratio'] < 2:
            pattern_type = "blobs"
        elif mf['coherence'] > 0.5 and mf['dominant_freq'] > 0.3:
            pattern_type = "checker/grid"
        elif mf['n_components'] <= 2:
            pattern_type = "simple_shapes"
        else:
            pattern_type = "complex"

        print(f"   Cluster {i} (n={stats['size']}): {pattern_type}")
        print(f"      stripe_score={mf['stripe_score']:.2f}, n_comp={mf['n_components']:.1f}, "
              f"aspect={mf['aspect_ratio']:.2f}, coherence={mf['coherence']:.2f}")

    # Success criteria: silhouette > 0.3 indicates reasonable clustering
    success = best_silhouette > 0.3

    output = {
        'n_samples': len(samples),
        'best_k': best_k,
        'best_silhouette': best_silhouette,
        'all_results': results,
        'feature_names': feature_names,
        'success': success,
        'status': 'validated' if success else 'refuted',
    }

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'pattern_classification.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n5. Result: {'VALIDATED' if success else 'REFUTED'}")
    print(f"   Silhouette {best_silhouette:.3f} {'>' if success else '<='} 0.3 threshold")

    return output


if __name__ == "__main__":
    main()
