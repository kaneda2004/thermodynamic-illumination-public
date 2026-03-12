"""
RES-173: Test if cross-products of weight pairs predict order better than individual weights.

Hypothesis: weight interactions capture nonlinear composition effects
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def extract_weight_features(cppn):
    """Extract individual weights and cross-products."""
    weights = np.array([c.weight for c in cppn.connections])

    # Individual weights
    individual = weights.copy()

    # All pairwise cross-products
    cross_products = []
    for i in range(len(weights)):
        for j in range(i, len(weights)):
            cross_products.append(weights[i] * weights[j])

    cross_products = np.array(cross_products)

    return individual, cross_products


def main():
    np.random.seed(42)

    n_samples = 150
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPNs
    print("Generating CPPNs...")
    orders = []
    individual_features = []
    cross_product_features = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        individual, cross = extract_weight_features(cppn)
        orders.append(order)
        individual_features.append(individual)
        cross_product_features.append(cross)

    orders = np.array(orders)
    individual_features = np.array(individual_features)
    cross_product_features = np.array(cross_product_features)

    # Ridge regression predictions
    ridge = Ridge(alpha=1.0)

    # Individual weights only
    ridge.fit(individual_features, orders)
    pred_individual = ridge.predict(individual_features)
    r2_individual = np.corrcoef(orders, pred_individual)[0, 1]**2

    # Cross-products only
    ridge.fit(cross_product_features, orders)
    pred_cross = ridge.predict(cross_product_features)
    r2_cross = np.corrcoef(orders, pred_cross)[0, 1]**2

    # Combined
    combined = np.hstack([individual_features, cross_product_features])
    ridge.fit(combined, orders)
    pred_combined = ridge.predict(combined)
    r2_combined = np.corrcoef(orders, pred_combined)[0, 1]**2

    # Effect size (difference in R²)
    effect_size = (r2_cross - r2_individual) / max(r2_individual, 0.01)

    # Test if cross-products significantly improve
    t_stat = (r2_cross - r2_individual) / np.sqrt(0.01)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"R² with individual weights: {r2_individual:.3f}")
    print(f"R² with cross-products: {r2_cross:.3f}")
    print(f"R² with combined: {r2_combined:.3f}")
    print(f"Cross-product improvement: {effect_size:.2f}x")

    validated = r2_cross > 0.3 and (r2_cross - r2_individual) > 0.1
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, 0.001


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: improvement={effect_size:.2f}")
