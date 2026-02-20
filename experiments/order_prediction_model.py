"""
RES-125: Order Prediction from CPPN Weights

Hypothesis: Image order can be predicted from CPPN weight vectors using a
simple random forest model with R^2 > 0.5, and weight statistics (mean, std,
L1/L2 norm) are the most predictive features.

Domain: order_prediction_model
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative

def generate_dataset(n_samples=500, seed=42):
    """Generate CPPN samples with weights and order values."""
    np.random.seed(seed)

    weights_list = []
    orders = []

    for i in range(n_samples):
        cppn = CPPN()  # Generates random weights from prior
        weights = cppn.get_weights()
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        weights_list.append(weights)
        orders.append(order)

    return np.array(weights_list), np.array(orders)

def compute_weight_features(weights):
    """Compute interpretable features from weight vectors."""
    features = []
    feature_names = []

    # Basic statistics
    features.append(np.mean(weights))
    feature_names.append('mean')

    features.append(np.std(weights))
    feature_names.append('std')

    features.append(np.min(weights))
    feature_names.append('min')

    features.append(np.max(weights))
    feature_names.append('max')

    # Norms
    features.append(np.linalg.norm(weights, 1))  # L1
    feature_names.append('L1_norm')

    features.append(np.linalg.norm(weights, 2))  # L2
    feature_names.append('L2_norm')

    features.append(np.linalg.norm(weights, np.inf))  # L-inf
    feature_names.append('Linf_norm')

    # Higher moments
    features.append(stats.skew(weights))
    feature_names.append('skewness')

    features.append(stats.kurtosis(weights))
    feature_names.append('kurtosis')

    # Sparsity-related
    features.append(np.sum(np.abs(weights) < 0.1) / len(weights))
    feature_names.append('near_zero_frac')

    features.append(np.sum(np.abs(weights) > 1.0) / len(weights))
    feature_names.append('large_frac')

    return np.array(features), feature_names

def run_experiment():
    print("=" * 60)
    print("RES-125: Order Prediction from CPPN Weights")
    print("=" * 60)

    # Generate data
    print("\nGenerating dataset...")
    weights_raw, orders = generate_dataset(n_samples=500, seed=42)
    print(f"  Samples: {len(orders)}")
    print(f"  Weight dimensions: {weights_raw.shape[1]}")
    print(f"  Order range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"  Order mean: {orders.mean():.3f} +/- {orders.std():.3f}")

    # Compute features
    print("\nComputing weight features...")
    features_list = []
    for w in weights_raw:
        feats, feat_names = compute_weight_features(w)
        features_list.append(feats)
    X_features = np.array(features_list)
    print(f"  Features: {len(feat_names)}")
    print(f"  Feature names: {feat_names}")

    # Model 1: Raw weights -> Order (Random Forest)
    print("\n--- Model 1: Raw Weights -> Order (RF) ---")
    rf_raw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_rf_raw = cross_val_score(rf_raw, weights_raw, orders, cv=cv, scoring='r2')
    print(f"  Cross-val R^2: {scores_rf_raw.mean():.4f} +/- {scores_rf_raw.std():.4f}")

    # Model 2: Features -> Order (Random Forest)
    print("\n--- Model 2: Weight Features -> Order (RF) ---")
    rf_feat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores_rf_feat = cross_val_score(rf_feat, X_features, orders, cv=cv, scoring='r2')
    print(f"  Cross-val R^2: {scores_rf_feat.mean():.4f} +/- {scores_rf_feat.std():.4f}")

    # Model 3: Raw weights -> Order (Linear Ridge)
    print("\n--- Model 3: Raw Weights -> Order (Ridge) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(weights_raw)
    ridge_raw = Ridge(alpha=1.0)
    scores_ridge_raw = cross_val_score(ridge_raw, X_scaled, orders, cv=cv, scoring='r2')
    print(f"  Cross-val R^2: {scores_ridge_raw.mean():.4f} +/- {scores_ridge_raw.std():.4f}")

    # Model 4: Features -> Order (Linear Ridge)
    print("\n--- Model 4: Weight Features -> Order (Ridge) ---")
    X_feat_scaled = StandardScaler().fit_transform(X_features)
    ridge_feat = Ridge(alpha=1.0)
    scores_ridge_feat = cross_val_score(ridge_feat, X_feat_scaled, orders, cv=cv, scoring='r2')
    print(f"  Cross-val R^2: {scores_ridge_feat.mean():.4f} +/- {scores_ridge_feat.std():.4f}")

    # Feature importance from RF on features
    print("\n--- Feature Importance (RF on features) ---")
    rf_feat.fit(X_features, orders)
    importances = rf_feat.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:5]:
        print(f"  {feat_names[i]}: {importances[i]:.4f}")

    # Statistical test: Is best R^2 > 0.5?
    best_r2 = scores_rf_raw.mean()  # Use raw weights RF
    print(f"\n--- Hypothesis Test: R^2 > 0.5 ---")
    print(f"  Best model: RF on raw weights")
    print(f"  R^2 = {best_r2:.4f} (95% CI: [{best_r2 - 2*scores_rf_raw.std():.4f}, {best_r2 + 2*scores_rf_raw.std():.4f}])")

    # Bootstrap test for p-value
    n_bootstrap = 1000
    bootstrap_r2s = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(orders), len(orders), replace=True)
        rf_boot = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_boot.fit(weights_raw[idx], orders[idx])
        # Use OOB-like test on non-sampled
        oob_idx = np.setdiff1d(np.arange(len(orders)), idx)
        if len(oob_idx) > 10:
            preds = rf_boot.predict(weights_raw[oob_idx])
            ss_res = np.sum((orders[oob_idx] - preds) ** 2)
            ss_tot = np.sum((orders[oob_idx] - orders[oob_idx].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            bootstrap_r2s.append(r2)

    bootstrap_r2s = np.array(bootstrap_r2s)
    p_value = np.mean(bootstrap_r2s < 0.5)  # Fraction below 0.5 (null hypothesis)
    print(f"  Bootstrap p-value (R^2 < 0.5): {p_value:.4f}")

    # Effect size: Cohen's d for R^2 vs 0
    effect_size = best_r2 / scores_rf_raw.std() if scores_rf_raw.std() > 0 else float('inf')
    print(f"  Effect size (R^2/std): {effect_size:.2f}")

    # Final verdict
    print("\n" + "=" * 60)
    if best_r2 > 0.5 and p_value < 0.01:
        status = "VALIDATED"
        print(f"STATUS: VALIDATED (R^2={best_r2:.3f} > 0.5, p={p_value:.4f} < 0.01)")
    elif best_r2 > 0.3:
        status = "INCONCLUSIVE"
        print(f"STATUS: INCONCLUSIVE (R^2={best_r2:.3f}, partial predictability)")
    else:
        status = "REFUTED"
        print(f"STATUS: REFUTED (R^2={best_r2:.3f} <= 0.5)")

    # Summary metrics
    return {
        'status': status,
        'r2_rf_raw': float(best_r2),
        'r2_rf_features': float(scores_rf_feat.mean()),
        'r2_ridge_raw': float(scores_ridge_raw.mean()),
        'r2_ridge_features': float(scores_ridge_feat.mean()),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'top_features': [(feat_names[i], float(importances[i])) for i in sorted_idx[:3]]
    }

if __name__ == "__main__":
    results = run_experiment()
    print(f"\nResults: {results}")
