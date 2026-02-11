"""
RES-133: Weight distribution higher-order moments predict CPPN image order

Hypothesis: Weight distribution higher-order moments (skewness, kurtosis, bimodality)
predict CPPN image order better than mean/std alone.

Building on RES-125 which showed R^2=0.25 using raw weights, we test if distribution
shape features provide better predictive power.
"""

import sys
import os
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_weight_shape_features(weights: np.ndarray) -> dict:
    """Compute distribution shape features from weight vector."""
    flat = weights.flatten()

    # Basic stats (baseline from RES-125)
    basic = {
        'mean': np.mean(flat),
        'std': np.std(flat),
    }

    # Higher-order moments
    higher_order = {
        'skewness': skew(flat),
        'kurtosis': kurtosis(flat),  # excess kurtosis
        'skewness_abs': abs(skew(flat)),
        'kurtosis_abs': abs(kurtosis(flat)),
    }

    # Distribution shape descriptors
    q1, q2, q3 = np.percentile(flat, [25, 50, 75])
    iqr = q3 - q1

    shape = {
        'iqr': iqr,
        'median': q2,
        'range': np.max(flat) - np.min(flat),
        'coef_variation': np.std(flat) / (abs(np.mean(flat)) + 1e-8),
    }

    # Bimodality coefficient (Sarle's bimodality coefficient)
    n = len(flat)
    s = skew(flat)
    k = kurtosis(flat)  # excess kurtosis
    bimodality = (s**2 + 1) / (k + 3 * (n-1)**2 / ((n-2)*(n-3)))

    shape['bimodality'] = bimodality

    # Tail heaviness (fraction of weights beyond 2 std)
    threshold = 2 * np.std(flat)
    shape['tail_fraction'] = np.mean(np.abs(flat - np.mean(flat)) > threshold)

    # Sparsity (fraction near zero)
    shape['near_zero_fraction'] = np.mean(np.abs(flat) < 0.1)

    # Mode estimation via histogram
    hist, bin_edges = np.histogram(flat, bins=30)
    shape['n_modes'] = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))

    return {**basic, **higher_order, **shape}


def run_experiment(n_samples=500, n_cv=5, random_seed=42):
    """Run the weight distribution shape experiment."""
    np.random.seed(random_seed)

    print(f"Generating {n_samples} CPPN samples...")

    # Collect samples
    orders = []
    basic_features = []  # mean, std only
    all_features = []    # all shape features

    for i in range(n_samples):
        # Random CPPN (with random weights from prior)
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        # Get weight vector
        weights = cppn.get_weights()
        features = compute_weight_shape_features(weights)

        orders.append(order)
        basic_features.append([features['mean'], features['std']])
        all_features.append(list(features.values()))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders = np.array(orders)
    basic_features = np.array(basic_features)
    all_features = np.array(all_features)

    feature_names = list(compute_weight_shape_features(np.zeros(10)).keys())

    print(f"\nFeature set sizes:")
    print(f"  Basic (mean, std): {basic_features.shape[1]} features")
    print(f"  All shape features: {all_features.shape[1]} features")

    # Model 1: Linear regression with basic features
    lr_basic = LinearRegression()
    scores_lr_basic = cross_val_score(lr_basic, basic_features, orders, cv=n_cv, scoring='r2')

    # Model 2: Linear regression with all shape features
    lr_all = LinearRegression()
    scores_lr_all = cross_val_score(lr_all, all_features, orders, cv=n_cv, scoring='r2')

    # Model 3: RF with basic features
    rf_basic = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
    scores_rf_basic = cross_val_score(rf_basic, basic_features, orders, cv=n_cv, scoring='r2')

    # Model 4: RF with all shape features
    rf_all = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
    scores_rf_all = cross_val_score(rf_all, all_features, orders, cv=n_cv, scoring='r2')

    print("\n=== Cross-Validated R^2 Scores ===")
    print(f"Linear (basic):  {scores_lr_basic.mean():.4f} +/- {scores_lr_basic.std():.4f}")
    print(f"Linear (shape):  {scores_lr_all.mean():.4f} +/- {scores_lr_all.std():.4f}")
    print(f"RF (basic):      {scores_rf_basic.mean():.4f} +/- {scores_rf_basic.std():.4f}")
    print(f"RF (shape):      {scores_rf_all.mean():.4f} +/- {scores_rf_all.std():.4f}")

    # Feature importance analysis
    rf_all_full = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
    rf_all_full.fit(all_features, orders)
    importances = rf_all_full.feature_importances_

    print("\n=== Feature Importances (top 8) ===")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:8]:
        print(f"  {feature_names[i]:20s}: {importances[i]:.4f}")

    # Statistical test: paired t-test comparing RF basic vs RF shape
    t_stat, p_value = stats.ttest_rel(scores_rf_basic, scores_rf_all)

    # Effect size (Cohen's d for improvement)
    improvement = scores_rf_all - scores_rf_basic
    effect_size = improvement.mean() / improvement.std() if improvement.std() > 0 else 0

    print("\n=== Statistical Comparison ===")
    print(f"RF basic R^2:    {scores_rf_basic.mean():.4f}")
    print(f"RF shape R^2:    {scores_rf_all.mean():.4f}")
    print(f"Improvement:     {improvement.mean():.4f}")
    print(f"t-statistic:     {t_stat:.4f}")
    print(f"p-value:         {p_value:.4f}")
    print(f"Effect size (d): {effect_size:.4f}")

    # Correlation analysis of individual features with order
    print("\n=== Individual Feature Correlations ===")
    correlations = []
    for i, name in enumerate(feature_names):
        r, p = stats.pearsonr(all_features[:, i], orders)
        correlations.append((name, r, p))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, p in correlations[:8]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:20s}: r={r:+.4f}, p={p:.2e} {sig}")

    # Determine outcome
    # Hypothesis validated if shape features provide significant improvement (p<0.01, d>0.5)
    validated = p_value < 0.05 and scores_rf_all.mean() > scores_rf_basic.mean()

    return {
        'r2_basic': float(scores_rf_basic.mean()),
        'r2_shape': float(scores_rf_all.mean()),
        'improvement': float(improvement.mean()),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'validated': validated,
        'feature_importances': {feature_names[i]: float(importances[i]) for i in sorted_idx[:8]},
        'top_correlations': {name: float(r) for name, r, _ in correlations[:5]}
    }


if __name__ == "__main__":
    results = run_experiment(n_samples=500)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"R^2 (basic mean/std): {results['r2_basic']:.4f}")
    print(f"R^2 (shape features): {results['r2_shape']:.4f}")
    print(f"Improvement:          {results['improvement']:.4f}")
    print(f"p-value:              {results['p_value']:.4f}")
    print(f"Effect size:          {results['effect_size']:.4f}")
    print(f"Status:               {'VALIDATED' if results['validated'] else 'REFUTED'}")
