#!/usr/bin/env python3
"""
RES-098: Weight Gradient Sparsity vs Order
==========================================

Hypothesis: Weight gradient sparsity (fraction of small gradients) correlates
with CPPN image order.

Intuition: High-order images might sit in "flat" regions of weight space where
many gradients are near-zero, indicating parameter redundancy or stability.

Method:
1. Sample CPPNs across order spectrum using nested sampling
2. Compute numerical gradients dOrder/dW for each weight
3. Measure sparsity = fraction of |grad| < threshold
4. Test Spearman correlation between sparsity and order
"""

import numpy as np
import json
import os
import sys
from scipy import stats
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, set_global_seed
)


@dataclass
class GradientAnalysis:
    """Results for a single CPPN."""
    order: float
    weights: np.ndarray
    gradients: np.ndarray
    sparsity: float  # fraction of |grad| < threshold
    mean_abs_grad: float
    max_abs_grad: float


def compute_order_gradient(cppn: CPPN, image_size: int = 32, epsilon: float = 1e-4) -> np.ndarray:
    """
    Compute numerical gradient of order w.r.t. weights.

    Uses central differences: grad_i = (f(w+e) - f(w-e)) / (2*epsilon)
    """
    weights = cppn.get_weights()
    n_weights = len(weights)
    gradients = np.zeros(n_weights)

    for i in range(n_weights):
        # Perturb +epsilon
        weights_plus = weights.copy()
        weights_plus[i] += epsilon
        cppn.set_weights(weights_plus)
        img_plus = cppn.render(image_size)
        order_plus = order_multiplicative(img_plus)

        # Perturb -epsilon
        weights_minus = weights.copy()
        weights_minus[i] -= epsilon
        cppn.set_weights(weights_minus)
        img_minus = cppn.render(image_size)
        order_minus = order_multiplicative(img_minus)

        # Central difference
        gradients[i] = (order_plus - order_minus) / (2 * epsilon)

    # Restore original weights
    cppn.set_weights(weights)

    return gradients


def analyze_cppn(cppn: CPPN, image_size: int = 32, sparsity_threshold: float = 0.1) -> GradientAnalysis:
    """Analyze gradient structure for a single CPPN."""
    img = cppn.render(image_size)
    order = order_multiplicative(img)
    weights = cppn.get_weights()
    gradients = compute_order_gradient(cppn, image_size)

    # Compute sparsity metrics
    abs_grads = np.abs(gradients)
    sparsity = np.mean(abs_grads < sparsity_threshold)
    mean_abs_grad = np.mean(abs_grads)
    max_abs_grad = np.max(abs_grads) if len(abs_grads) > 0 else 0.0

    return GradientAnalysis(
        order=order,
        weights=weights,
        gradients=gradients,
        sparsity=sparsity,
        mean_abs_grad=mean_abs_grad,
        max_abs_grad=max_abs_grad
    )


def sample_cppns_across_order_spectrum(n_samples: int = 200, image_size: int = 32, seed: int = 42) -> list:
    """
    Sample CPPNs to get a range of order values.

    Strategy: Generate many random CPPNs to get natural distribution.
    """
    set_global_seed(seed)
    cppns = []

    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        cppns.append((cppn.copy(), order))

    return cppns


def run_experiment():
    """Main experiment: test gradient sparsity vs order correlation."""
    print("=" * 70)
    print("RES-098: Weight Gradient Sparsity vs Order")
    print("=" * 70)

    # Parameters
    n_samples = 200
    image_size = 32
    sparsity_threshold = 0.1  # |grad| < 0.1 considered "sparse"
    seed = 42

    print(f"\nParameters:")
    print(f"  n_samples: {n_samples}")
    print(f"  image_size: {image_size}")
    print(f"  sparsity_threshold: {sparsity_threshold}")
    print(f"  seed: {seed}")

    # Sample CPPNs
    print("\n1. Sampling CPPNs...")
    cppns = sample_cppns_across_order_spectrum(n_samples, image_size, seed)
    orders = [o for _, o in cppns]
    print(f"   Order range: [{min(orders):.4f}, {max(orders):.4f}]")
    print(f"   Order mean: {np.mean(orders):.4f}")

    # Analyze each CPPN
    print("\n2. Computing gradients...")
    analyses = []
    for i, (cppn, _) in enumerate(cppns):
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{n_samples}...")
        analysis = analyze_cppn(cppn, image_size, sparsity_threshold)
        analyses.append(analysis)

    # Extract metrics
    orders = np.array([a.order for a in analyses])
    sparsities = np.array([a.sparsity for a in analyses])
    mean_grads = np.array([a.mean_abs_grad for a in analyses])
    max_grads = np.array([a.max_abs_grad for a in analyses])

    print("\n3. Statistical Analysis")
    print("-" * 50)

    # Primary test: Spearman correlation (sparsity vs order)
    spearman_sparsity = stats.spearmanr(orders, sparsities)
    print(f"\nSparsity vs Order (Spearman):")
    print(f"  rho = {spearman_sparsity.correlation:.4f}")
    print(f"  p-value = {spearman_sparsity.pvalue:.2e}")

    # Secondary: Mean gradient magnitude vs order
    spearman_mean = stats.spearmanr(orders, mean_grads)
    print(f"\nMean |Gradient| vs Order (Spearman):")
    print(f"  rho = {spearman_mean.correlation:.4f}")
    print(f"  p-value = {spearman_mean.pvalue:.2e}")

    # Tertiary: Max gradient vs order
    spearman_max = stats.spearmanr(orders, max_grads)
    print(f"\nMax |Gradient| vs Order (Spearman):")
    print(f"  rho = {spearman_max.correlation:.4f}")
    print(f"  p-value = {spearman_max.pvalue:.2e}")

    # Compare high vs low order groups
    print("\n4. Group Comparison")
    print("-" * 50)

    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = orders <= median_order

    high_sparsity = sparsities[high_mask]
    low_sparsity = sparsities[low_mask]

    mannwhitney = stats.mannwhitneyu(high_sparsity, low_sparsity, alternative='two-sided')

    # Effect size (rank-biserial correlation from U statistic)
    n1, n2 = len(high_sparsity), len(low_sparsity)
    effect_size = 1 - (2 * mannwhitney.statistic) / (n1 * n2)

    print(f"\nHigh-Order vs Low-Order Sparsity:")
    print(f"  High-order mean sparsity: {np.mean(high_sparsity):.4f} +/- {np.std(high_sparsity):.4f}")
    print(f"  Low-order mean sparsity: {np.mean(low_sparsity):.4f} +/- {np.std(low_sparsity):.4f}")
    print(f"  Mann-Whitney U: {mannwhitney.statistic:.1f}")
    print(f"  p-value: {mannwhitney.pvalue:.2e}")
    print(f"  Effect size (rank-biserial): {effect_size:.4f}")

    # Determine validation status
    print("\n5. Validation Decision")
    print("-" * 50)

    # Use Spearman correlation as primary metric
    validated = (spearman_sparsity.pvalue < 0.01 and
                 abs(spearman_sparsity.correlation) > 0.2)

    if validated:
        direction = "higher" if spearman_sparsity.correlation > 0 else "lower"
        status = "VALIDATED"
        summary = (f"Weight gradient sparsity correlates with order "
                   f"(rho={spearman_sparsity.correlation:.3f}, p={spearman_sparsity.pvalue:.2e}). "
                   f"High-order CPPNs have {direction} gradient sparsity.")
    else:
        status = "REFUTED"
        summary = (f"No significant correlation between gradient sparsity and order "
                   f"(rho={spearman_sparsity.correlation:.3f}, p={spearman_sparsity.pvalue:.2e}).")

    print(f"\nStatus: {status}")
    print(f"Summary: {summary}")

    # Save results
    results_dir = "results/weight_gradient_sparsity"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "experiment_id": "RES-098",
        "hypothesis": "Weight gradient sparsity correlates with CPPN image order",
        "domain": "weight_gradient_structure",
        "status": status.lower(),
        "parameters": {
            "n_samples": n_samples,
            "image_size": image_size,
            "sparsity_threshold": sparsity_threshold,
            "seed": seed
        },
        "metrics": {
            "spearman_rho_sparsity": float(spearman_sparsity.correlation),
            "p_value_sparsity": float(spearman_sparsity.pvalue),
            "spearman_rho_mean_grad": float(spearman_mean.correlation),
            "p_value_mean_grad": float(spearman_mean.pvalue),
            "mannwhitney_effect_size": float(effect_size),
            "mannwhitney_pvalue": float(mannwhitney.pvalue),
            "high_order_sparsity_mean": float(np.mean(high_sparsity)),
            "low_order_sparsity_mean": float(np.mean(low_sparsity))
        },
        "summary": summary
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
