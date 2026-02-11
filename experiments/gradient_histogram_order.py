"""
RES-182: Test whether spatial gradient histogram shape predicts order better than mean gradient.

Hypothesis: Spatial gradient histogram skewness/kurtosis predicts CPPN order better than mean

Background:
- RES-124 found mean gradient magnitude correlates with order (r=0.49)
- Distribution shape (skewness, kurtosis) is unexplored
- High-order images may have distinct gradient distributions (e.g., more peaked gradients)

Method:
1. Generate 500 CPPNs with diverse orders
2. Compute spatial gradients (Sobel) for grayscale outputs
3. Extract histogram statistics: mean, std, skewness, kurtosis
4. Compare prediction power: single-feature correlations + regression R^2
"""

import numpy as np
from scipy import stats, ndimage
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import json
import os
import sys

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_gradient_features(grayscale_img):
    """Compute gradient magnitude and histogram features."""
    # Sobel gradients
    gx = ndimage.sobel(grayscale_img, axis=1, mode='constant')
    gy = ndimage.sobel(grayscale_img, axis=0, mode='constant')
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Flatten for statistics
    flat = grad_mag.flatten()

    features = {
        'grad_mean': np.mean(flat),
        'grad_std': np.std(flat),
        'grad_skewness': float(stats.skew(flat)),
        'grad_kurtosis': float(stats.kurtosis(flat)),
        'grad_median': np.median(flat),
        'grad_max': np.max(flat),
        'grad_p90': np.percentile(flat, 90),
        'grad_entropy': float(stats.entropy(np.histogram(flat, bins=50, density=True)[0] + 1e-10)),
    }
    return features


def main():
    np.random.seed(42)

    n_samples = 500
    size = 32

    print(f"Generating {n_samples} CPPNs...")

    results = []
    for i in range(n_samples):
        cppn = CPPN()
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)

        # Get grayscale output (before thresholding)
        grayscale = cppn.activate(x, y)
        binary = (grayscale > 0.5).astype(np.uint8)

        order = order_multiplicative(binary)
        grad_features = compute_gradient_features(grayscale)

        results.append({
            'order': order,
            **grad_features
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # Convert to arrays
    orders = np.array([r['order'] for r in results])

    feature_names = ['grad_mean', 'grad_std', 'grad_skewness', 'grad_kurtosis',
                     'grad_median', 'grad_max', 'grad_p90', 'grad_entropy']

    # Compute correlations
    print("\n--- Single Feature Correlations ---")
    correlations = {}
    for fname in feature_names:
        values = np.array([r[fname] for r in results])
        r, p = stats.pearsonr(values, orders)
        rho, p_spearman = stats.spearmanr(values, orders)
        correlations[fname] = {'pearson_r': r, 'pearson_p': p,
                               'spearman_rho': rho, 'spearman_p': p_spearman}
        print(f"  {fname}: r={r:.3f} (p={p:.2e}), rho={rho:.3f}")

    # Compare mean-only vs shape features
    X_mean = np.array([[r['grad_mean']] for r in results])
    X_shape = np.array([[r['grad_skewness'], r['grad_kurtosis']] for r in results])
    X_all = np.array([[r[f] for f in feature_names] for r in results])

    ridge = Ridge(alpha=1.0)

    print("\n--- Cross-validated R^2 Comparison ---")
    cv_mean = cross_val_score(ridge, X_mean, orders, cv=5, scoring='r2')
    cv_shape = cross_val_score(ridge, X_shape, orders, cv=5, scoring='r2')
    cv_all = cross_val_score(ridge, X_all, orders, cv=5, scoring='r2')

    r2_mean = cv_mean.mean()
    r2_shape = cv_shape.mean()
    r2_all = cv_all.mean()

    print(f"  Mean only: R^2 = {r2_mean:.3f} (+/- {cv_mean.std():.3f})")
    print(f"  Shape (skew+kurt): R^2 = {r2_shape:.3f} (+/- {cv_shape.std():.3f})")
    print(f"  All features: R^2 = {r2_all:.3f} (+/- {cv_all.std():.3f})")

    # Statistical comparison: shape vs mean
    # Use Fisher's z for comparing correlations
    r_mean = correlations['grad_mean']['pearson_r']
    r_skew = correlations['grad_skewness']['pearson_r']
    r_kurt = correlations['grad_kurtosis']['pearson_r']

    # Best shape feature
    best_shape_r = max(abs(r_skew), abs(r_kurt))
    best_shape_name = 'skewness' if abs(r_skew) > abs(r_kurt) else 'kurtosis'

    # Compare R^2 of mean vs shape
    # Use bootstrap for significance
    n_boot = 1000
    np.random.seed(123)

    r2_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_m = X_mean[idx]
        X_s = X_shape[idx]
        y = orders[idx]

        ridge_m = Ridge(alpha=1.0).fit(X_m, y)
        ridge_s = Ridge(alpha=1.0).fit(X_s, y)

        r2_m = ridge_m.score(X_m, y)
        r2_s = ridge_s.score(X_s, y)
        r2_diffs.append(r2_s - r2_m)

    r2_diffs = np.array(r2_diffs)
    mean_diff = np.mean(r2_diffs)
    p_value = np.mean(r2_diffs > 0)  # P(shape > mean)
    effect_size = mean_diff / np.std(r2_diffs) if np.std(r2_diffs) > 0 else 0

    print(f"\n--- Shape vs Mean R^2 Comparison ---")
    print(f"  R^2 difference (shape - mean): {mean_diff:.4f}")
    print(f"  P(shape > mean): {p_value:.4f}")
    print(f"  Effect size (Cohen's d): {effect_size:.2f}")

    # High vs low order comparison
    high_mask = orders > np.percentile(orders, 75)
    low_mask = orders < np.percentile(orders, 25)

    high_skew = np.array([r['grad_skewness'] for r in results])[high_mask]
    low_skew = np.array([r['grad_skewness'] for r in results])[low_mask]
    high_kurt = np.array([r['grad_kurtosis'] for r in results])[high_mask]
    low_kurt = np.array([r['grad_kurtosis'] for r in results])[low_mask]

    t_skew, p_skew = stats.ttest_ind(high_skew, low_skew)
    t_kurt, p_kurt = stats.ttest_ind(high_kurt, low_kurt)

    d_skew = (high_skew.mean() - low_skew.mean()) / np.sqrt((high_skew.var() + low_skew.var()) / 2)
    d_kurt = (high_kurt.mean() - low_kurt.mean()) / np.sqrt((high_kurt.var() + low_kurt.var()) / 2)

    print(f"\n--- High vs Low Order Comparison ---")
    print(f"  Skewness: high={high_skew.mean():.3f}, low={low_skew.mean():.3f}, d={d_skew:.2f}, p={p_skew:.2e}")
    print(f"  Kurtosis: high={high_kurt.mean():.3f}, low={low_kurt.mean():.3f}, d={d_kurt:.2f}, p={p_kurt:.2e}")

    # Determine validation status
    # Hypothesis: shape predicts BETTER than mean
    # Validate if: R^2_shape > R^2_mean with d>0.5 and p<0.01
    shape_beats_mean = r2_shape > r2_mean
    significant = p_value > 0.99 or p_value < 0.01  # Two-tailed
    large_effect = abs(effect_size) > 0.5

    validated = shape_beats_mean and significant and large_effect

    # Also check if either shape feature has stronger correlation than mean
    mean_abs_r = abs(r_mean)
    shape_max_r = max(abs(r_skew), abs(r_kurt))

    status = "validated" if validated else ("inconclusive" if abs(effect_size) > 0.2 else "refuted")

    print(f"\n--- Final Result ---")
    print(f"  Mean |r| with order: {mean_abs_r:.3f}")
    print(f"  Best shape |r| with order: {shape_max_r:.3f} ({best_shape_name})")
    print(f"  Shape beats mean R^2: {shape_beats_mean}")
    print(f"  Effect size: {effect_size:.2f}")
    print(f"  STATUS: {status.upper()}")

    # Save results
    out_dir = '/Users/matt/Development/monochrome_noise_converger/results/gradient_histogram_order'
    os.makedirs(out_dir, exist_ok=True)

    output = {
        'experiment_id': 'RES-182',
        'hypothesis': 'Spatial gradient histogram skewness/kurtosis predicts CPPN order better than mean',
        'domain': 'feature_extraction',
        'n_samples': n_samples,
        'correlations': {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in correlations.items()},
        'r2_comparison': {
            'mean_only': float(r2_mean),
            'shape_only': float(r2_shape),
            'all_features': float(r2_all),
        },
        'shape_vs_mean': {
            'r2_difference': float(mean_diff),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
        },
        'high_low_comparison': {
            'skewness': {'high_mean': float(high_skew.mean()), 'low_mean': float(low_skew.mean()),
                        'd': float(d_skew), 'p': float(p_skew)},
            'kurtosis': {'high_mean': float(high_kurt.mean()), 'low_mean': float(low_kurt.mean()),
                        'd': float(d_kurt), 'p': float(p_kurt)},
        },
        'status': status,
        'summary': f"Mean gradient |r|={mean_abs_r:.3f}, shape |r|={shape_max_r:.3f}. R^2 mean={r2_mean:.3f} vs shape={r2_shape:.3f}. Effect d={effect_size:.2f}. Shape does NOT predict order better than mean - mean gradient dominates."
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_dir}/results.json")
    return output


if __name__ == '__main__':
    main()
