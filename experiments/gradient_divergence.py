"""
RES-162: Order gradient field divergence analysis

Hypothesis: Order gradient field has negative divergence near high-order regions (sink behavior)

If the order landscape has "attractors" at high-order regions, the gradient field
should converge toward those points (negative divergence = sink).
Low-order regions might show positive divergence (source) or near-zero.

Divergence of vector field F = div(F) = dFx/dx + dFy/dy + ...
We compute this via finite differences on the gradient field itself.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
import json
import os

def compute_order_gradient(cppn: CPPN, eps: float = 0.01) -> np.ndarray:
    """Compute gradient of order w.r.t. weights via finite differences."""
    w = cppn.get_weights()
    n = len(w)
    grad = np.zeros(n)

    for i in range(n):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps

        cppn.set_weights(w_plus)
        order_plus = order_multiplicative(cppn.render(32))

        cppn.set_weights(w_minus)
        order_minus = order_multiplicative(cppn.render(32))

        grad[i] = (order_plus - order_minus) / (2 * eps)

    cppn.set_weights(w)  # Restore
    return grad

def compute_gradient_divergence(cppn: CPPN, eps: float = 0.01, h: float = 0.05) -> float:
    """
    Compute divergence of the gradient field at current position.

    div(grad(order)) = sum_i d/dw_i (d(order)/dw_i)
    This is the trace of the Hessian.
    """
    w = cppn.get_weights()
    n = len(w)
    div = 0.0

    for i in range(n):
        # Compute gradient at w + h*e_i and w - h*e_i
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += h
        w_minus[i] -= h

        cppn.set_weights(w_plus)
        grad_plus = compute_order_gradient(cppn, eps)

        cppn.set_weights(w_minus)
        grad_minus = compute_order_gradient(cppn, eps)

        # d(grad_i)/dw_i
        div += (grad_plus[i] - grad_minus[i]) / (2 * h)

    cppn.set_weights(w)  # Restore
    return div

def run_experiment(n_samples: int = 100, seed: int = 42):
    np.random.seed(seed)

    results = []

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"Sample {i}/{n_samples}")

        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute divergence (trace of Hessian)
        div = compute_gradient_divergence(cppn, eps=0.01, h=0.05)

        results.append({
            'sample': i,
            'order': order,
            'divergence': div
        })

    return results

def analyze_results(results):
    orders = np.array([r['order'] for r in results])
    divs = np.array([r['divergence'] for r in results])

    # Filter valid (non-zero order)
    valid = orders > 0.01
    orders_valid = orders[valid]
    divs_valid = divs[valid]

    # Correlation
    if len(orders_valid) > 5:
        corr, pval = stats.pearsonr(orders_valid, divs_valid)
    else:
        corr, pval = 0, 1

    # Split by order level
    median_order = np.median(orders_valid) if len(orders_valid) > 0 else 0.1
    high_order = divs_valid[orders_valid > median_order]
    low_order = divs_valid[orders_valid <= median_order]

    # Effect size
    if len(high_order) > 1 and len(low_order) > 1:
        pooled_std = np.sqrt((np.std(high_order)**2 + np.std(low_order)**2) / 2)
        d = (np.mean(high_order) - np.mean(low_order)) / (pooled_std + 1e-10)
        t_stat, t_pval = stats.ttest_ind(high_order, low_order)
    else:
        d, t_pval = 0, 1

    analysis = {
        'n_valid': int(np.sum(valid)),
        'correlation': float(corr),
        'correlation_pval': float(pval),
        'mean_div_high_order': float(np.mean(high_order)) if len(high_order) > 0 else 0,
        'mean_div_low_order': float(np.mean(low_order)) if len(low_order) > 0 else 0,
        'cohens_d': float(d),
        'ttest_pval': float(t_pval),
        'median_order': float(median_order),
        'high_negative_div': float(np.mean(high_order < 0)) if len(high_order) > 0 else 0,
        'low_negative_div': float(np.mean(low_order < 0)) if len(low_order) > 0 else 0,
    }

    return analysis

if __name__ == '__main__':
    print("Running gradient divergence experiment (RES-162)...")

    results = run_experiment(n_samples=100, seed=42)
    analysis = analyze_results(results)

    print("\n=== RESULTS ===")
    print(f"Valid samples: {analysis['n_valid']}")
    print(f"Order-Divergence correlation: r={analysis['correlation']:.3f} (p={analysis['correlation_pval']:.4f})")
    print(f"Mean divergence (high order): {analysis['mean_div_high_order']:.4f}")
    print(f"Mean divergence (low order): {analysis['mean_div_low_order']:.4f}")
    print(f"Cohen's d: {analysis['cohens_d']:.3f}")
    print(f"T-test p-value: {analysis['ttest_pval']:.4f}")
    print(f"High order negative div rate: {analysis['high_negative_div']:.2%}")
    print(f"Low order negative div rate: {analysis['low_negative_div']:.2%}")

    # Save results
    os.makedirs('results/gradient_landscape', exist_ok=True)
    with open('results/gradient_landscape/divergence_results.json', 'w') as f:
        json.dump({'results': results, 'analysis': analysis}, f, indent=2)

    print("\nResults saved to results/gradient_landscape/divergence_results.json")
