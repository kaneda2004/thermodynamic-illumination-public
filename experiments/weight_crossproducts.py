"""
RES-173: Cross-products of weight pairs predict order better than individual weights

Hypothesis: The multiplicative interaction terms (w_i * w_j) between CPPN weights
predict order better than just individual weights, since the CPPN computes weighted
sums that are then passed through nonlinear activations, creating interaction effects.

Method:
1. Generate many random CPPNs
2. For each, extract individual weights and all pairwise cross-products
3. Train regression models on:
   a) Individual weights only
   b) Cross-products only
   c) Both combined
4. Compare R^2 values to see if interactions add predictive power

Effect size: Compare R^2 of cross-product model vs individual weights model
"""

import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def generate_cppn_data(n_samples=500, seed=42):
    """Generate random CPPNs and compute their orders."""
    np.random.seed(seed)

    data = []
    for _ in range(n_samples):
        cppn = CPPN()  # Random initialization
        img = cppn.render(32)
        order = order_multiplicative(img)
        weights = cppn.get_weights()
        data.append({
            'weights': weights,
            'order': order
        })

    return data


def compute_cross_products(weights):
    """Compute all pairwise cross-products of weights."""
    n = len(weights)
    cross_prods = []
    for i in range(n):
        for j in range(i, n):  # Upper triangle including diagonal (squares)
            cross_prods.append(weights[i] * weights[j])
    return np.array(cross_prods)


def run_experiment():
    print("Generating CPPN samples...")
    data = generate_cppn_data(n_samples=500)

    # Extract features
    X_individual = np.array([d['weights'] for d in data])
    X_crossprods = np.array([compute_cross_products(d['weights']) for d in data])
    X_combined = np.hstack([X_individual, X_crossprods])
    y = np.array([d['order'] for d in data])

    print(f"Sample size: {len(y)}")
    print(f"Individual features: {X_individual.shape[1]}")
    print(f"Cross-product features: {X_crossprods.shape[1]}")
    print(f"Combined features: {X_combined.shape[1]}")
    print(f"Order range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

    # Models to test
    models = {
        'ridge': Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{model_name.upper()} regression:")

        # Individual weights only
        scores_ind = cross_val_score(model, X_individual, y, cv=5, scoring='r2')
        r2_individual = scores_ind.mean()
        print(f"  Individual weights R^2: {r2_individual:.4f} (+/- {scores_ind.std():.4f})")

        # Cross-products only
        scores_cp = cross_val_score(model, X_crossprods, y, cv=5, scoring='r2')
        r2_crossprods = scores_cp.mean()
        print(f"  Cross-products R^2: {r2_crossprods:.4f} (+/- {scores_cp.std():.4f})")

        # Combined
        scores_combined = cross_val_score(model, X_combined, y, cv=5, scoring='r2')
        r2_combined = scores_combined.mean()
        print(f"  Combined R^2: {r2_combined:.4f} (+/- {scores_combined.std():.4f})")

        results[model_name] = {
            'r2_individual': float(r2_individual),
            'r2_crossprods': float(r2_crossprods),
            'r2_combined': float(r2_combined),
            'r2_ind_std': float(scores_ind.std()),
            'r2_cp_std': float(scores_cp.std()),
            'r2_combined_std': float(scores_combined.std()),
            'improvement_cp_vs_ind': float(r2_crossprods - r2_individual),
            'improvement_combined_vs_ind': float(r2_combined - r2_individual)
        }

    # Statistical test: is cross-product R^2 significantly better than individual?
    # Use best model (RF typically)
    best_model_key = max(results.keys(), key=lambda k: results[k]['r2_combined'])
    best = results[best_model_key]

    # Compute effect size (difference in R^2)
    delta_r2 = best['r2_crossprods'] - best['r2_individual']

    # Bootstrap for significance of R^2 difference
    print("\nBootstrap test for R^2 difference...")
    n_bootstrap = 1000
    deltas = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y), len(y), replace=True)
        X_ind_b = X_individual[idx]
        X_cp_b = X_crossprods[idx]
        y_b = y[idx]

        # Quick Ridge for speed
        ridge = Ridge(alpha=1.0)
        scaler = StandardScaler()

        X_ind_s = scaler.fit_transform(X_ind_b)
        ridge.fit(X_ind_s, y_b)
        pred_ind = ridge.predict(X_ind_s)
        ss_res_ind = np.sum((y_b - pred_ind)**2)
        ss_tot = np.sum((y_b - y_b.mean())**2)
        r2_ind = 1 - ss_res_ind/ss_tot if ss_tot > 0 else 0

        X_cp_s = scaler.fit_transform(X_cp_b)
        ridge.fit(X_cp_s, y_b)
        pred_cp = ridge.predict(X_cp_s)
        ss_res_cp = np.sum((y_b - pred_cp)**2)
        r2_cp = 1 - ss_res_cp/ss_tot if ss_tot > 0 else 0

        deltas.append(r2_cp - r2_ind)

    deltas = np.array(deltas)
    p_value = np.mean(deltas <= 0)  # proportion of times cross-prods NOT better
    mean_delta = np.mean(deltas)

    print(f"Bootstrap R^2 difference: {mean_delta:.4f} (p={p_value:.4f})")

    # Effect size: Cohen's d for the delta distribution
    cohens_d = mean_delta / (deltas.std() + 1e-10)

    # Additional analysis: which cross-products matter most?
    print("\nTop predictive cross-products (RF feature importance):")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_crossprods, y)
    importances = rf.feature_importances_

    # Map feature indices to weight pairs
    n_weights = X_individual.shape[1]
    feature_names = []
    idx = 0
    for i in range(n_weights):
        for j in range(i, n_weights):
            feature_names.append(f"w{i}*w{j}")
            idx += 1

    top_k = 5
    top_indices = np.argsort(importances)[-top_k:][::-1]
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. {feature_names[idx]}: importance={importances[idx]:.4f}")

    # Summary
    summary = {
        'n_samples': len(y),
        'n_individual_features': int(X_individual.shape[1]),
        'n_crossproduct_features': int(X_crossprods.shape[1]),
        'best_model': best_model_key,
        'r2_individual': best['r2_individual'],
        'r2_crossprods': best['r2_crossprods'],
        'r2_combined': best['r2_combined'],
        'delta_r2': float(delta_r2),
        'bootstrap_delta_mean': float(mean_delta),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'top_crossproducts': [feature_names[i] for i in top_indices],
        'top_importances': [float(importances[i]) for i in top_indices],
        'model_results': results
    }

    # Determine outcome
    # Hypothesis: cross-products predict BETTER than individual
    # Validated if: delta_r2 > 0, p < 0.01, effect size > 0.5
    if delta_r2 > 0 and p_value < 0.01 and abs(cohens_d) > 0.5:
        status = 'validated'
    elif delta_r2 <= 0:
        status = 'refuted'
    else:
        status = 'inconclusive'

    summary['status'] = status

    print(f"\n{'='*60}")
    print(f"RESULT: {status.upper()}")
    print(f"R^2 individual: {best['r2_individual']:.4f}")
    print(f"R^2 cross-products: {best['r2_crossprods']:.4f}")
    print(f"Delta R^2: {delta_r2:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Save results
    with open('/Users/matt/Development/monochrome_noise_converger/results/weight_crossproducts/results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to results/weight_crossproducts/results.json")

    return summary


if __name__ == '__main__':
    run_experiment()
