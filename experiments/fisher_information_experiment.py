"""
RES-049: Fisher Information Geometry of Order Landscape

Hypothesis: Fisher information (curvature of order landscape) is higher in
high-order regions, indicating steeper geometry that explains why gradient
methods fail.

Fisher information measures local curvature: F_ij = E[(d log p / d theta_i)(d log p / d theta_j)]
For our order metric, we approximate this as the Hessian of log(order) w.r.t. weights.

Method:
1. Sample CPPNs across order spectrum (low, medium, high)
2. Compute numerical Hessian of order at each point
3. Measure Fisher information via trace(H^2) and condition number
4. Test: High-order regions have larger Fisher information (steeper curvature)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_order_for_weights(base_cppn, weights):
    """Helper to compute order given weight vector."""
    cppn = base_cppn.copy()
    cppn.set_weights(weights)
    img = cppn.render(32)
    return compute_order(img)

def numerical_hessian(base_cppn, weights, eps=0.01):
    """
    Compute numerical Hessian of order w.r.t. weights.
    Uses central differences for accuracy.
    """
    n = len(weights)
    H = np.zeros((n, n))
    f0 = compute_order_for_weights(base_cppn, weights)

    for i in range(n):
        for j in range(i, n):
            # Four-point stencil for mixed partials
            w_pp = weights.copy()
            w_pp[i] += eps
            w_pp[j] += eps

            w_pm = weights.copy()
            w_pm[i] += eps
            w_pm[j] -= eps

            w_mp = weights.copy()
            w_mp[i] -= eps
            w_mp[j] += eps

            w_mm = weights.copy()
            w_mm[i] -= eps
            w_mm[j] -= eps

            f_pp = compute_order_for_weights(base_cppn, w_pp)
            f_pm = compute_order_for_weights(base_cppn, w_pm)
            f_mp = compute_order_for_weights(base_cppn, w_mp)
            f_mm = compute_order_for_weights(base_cppn, w_mm)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            H[j, i] = H[i, j]

    return H

def compute_fisher_metrics(H):
    """
    Compute Fisher information metrics from Hessian.
    - trace_sq: Trace of H^2 (total curvature)
    - max_eigenvalue: Maximum curvature direction
    - condition_number: Ratio of max/min eigenvalues (anisotropy)
    - frobenius: Frobenius norm (overall magnitude)
    """
    eigenvalues = np.linalg.eigvalsh(H)
    abs_eigenvalues = np.abs(eigenvalues)

    return {
        'trace_sq': np.trace(H @ H),
        'max_eigenvalue': np.max(abs_eigenvalues),
        'min_eigenvalue': np.min(abs_eigenvalues) + 1e-10,
        'condition_number': np.max(abs_eigenvalues) / (np.min(abs_eigenvalues) + 1e-10),
        'frobenius': np.linalg.norm(H, 'fro'),
        'mean_curvature': np.mean(abs_eigenvalues),
    }

def run_experiment(n_samples=50, seed=42):
    """
    Sample CPPNs, compute Hessian, and test Fisher information vs order.
    """
    np.random.seed(seed)

    results = []
    print(f"Computing Fisher information for {n_samples} CPPNs...")

    for i in range(n_samples):
        # Create random CPPN
        cppn = CPPN()
        weights = cppn.get_weights()
        img = cppn.render(32)
        order = compute_order(img)

        # Compute Hessian
        H = numerical_hessian(cppn, weights)
        fisher = compute_fisher_metrics(H)

        results.append({
            'order': order,
            **fisher
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    return results

def analyze_results(results):
    """Statistical analysis of Fisher information vs order."""
    orders = np.array([r['order'] for r in results])

    # Key metric: Frobenius norm of Hessian (overall curvature magnitude)
    frobenius = np.array([r['frobenius'] for r in results])
    max_eig = np.array([r['max_eigenvalue'] for r in results])
    condition = np.array([r['condition_number'] for r in results])
    mean_curv = np.array([r['mean_curvature'] for r in results])

    # Spearman correlations (robust to non-linearity)
    rho_frob, p_frob = stats.spearmanr(orders, frobenius)
    rho_max, p_max = stats.spearmanr(orders, max_eig)
    rho_cond, p_cond = stats.spearmanr(orders, condition)
    rho_mean, p_mean = stats.spearmanr(orders, mean_curv)

    # Group comparison: low vs high order
    median_order = np.median(orders)
    low_mask = orders < median_order
    high_mask = orders >= median_order

    low_frob = frobenius[low_mask]
    high_frob = frobenius[high_mask]

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(high_frob, low_frob, alternative='greater')

    # Effect size (Cohen's d)
    cohens_d = (np.mean(high_frob) - np.mean(low_frob)) / np.sqrt(
        (np.var(high_frob) + np.var(low_frob)) / 2
    )

    return {
        'n_samples': len(results),
        'order_range': [float(np.min(orders)), float(np.max(orders))],
        'frobenius_correlation': {
            'spearman_rho': float(rho_frob),
            'p_value': float(p_frob),
        },
        'max_eigenvalue_correlation': {
            'spearman_rho': float(rho_max),
            'p_value': float(p_max),
        },
        'condition_number_correlation': {
            'spearman_rho': float(rho_cond),
            'p_value': float(p_cond),
        },
        'mean_curvature_correlation': {
            'spearman_rho': float(rho_mean),
            'p_value': float(p_mean),
        },
        'group_comparison': {
            'low_order_mean_frobenius': float(np.mean(low_frob)),
            'high_order_mean_frobenius': float(np.mean(high_frob)),
            'mann_whitney_u': float(u_stat),
            'mann_whitney_p': float(u_p),
            'cohens_d': float(cohens_d),
        },
        'summary_stats': {
            'mean_order': float(np.mean(orders)),
            'mean_frobenius': float(np.mean(frobenius)),
            'mean_max_eigenvalue': float(np.mean(max_eig)),
            'mean_condition_number': float(np.mean(condition)),
        }
    }

if __name__ == '__main__':
    # Run experiment
    results = run_experiment(n_samples=50, seed=42)

    # Analyze
    analysis = analyze_results(results)

    # Save results
    os.makedirs('results/fisher_information', exist_ok=True)
    with open('results/fisher_information/fisher_results.json', 'w') as f:
        json.dump({
            'raw_results': results,
            'analysis': analysis
        }, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("FISHER INFORMATION GEOMETRY RESULTS")
    print("="*60)
    print(f"\nSamples: {analysis['n_samples']}")
    print(f"Order range: {analysis['order_range'][0]:.3f} - {analysis['order_range'][1]:.3f}")

    print(f"\nCorrelations with Order:")
    print(f"  Frobenius norm: rho={analysis['frobenius_correlation']['spearman_rho']:.3f}, p={analysis['frobenius_correlation']['p_value']:.2e}")
    print(f"  Max eigenvalue: rho={analysis['max_eigenvalue_correlation']['spearman_rho']:.3f}, p={analysis['max_eigenvalue_correlation']['p_value']:.2e}")
    print(f"  Condition number: rho={analysis['condition_number_correlation']['spearman_rho']:.3f}, p={analysis['condition_number_correlation']['p_value']:.2e}")
    print(f"  Mean curvature: rho={analysis['mean_curvature_correlation']['spearman_rho']:.3f}, p={analysis['mean_curvature_correlation']['p_value']:.2e}")

    print(f"\nGroup Comparison (High vs Low Order):")
    gc = analysis['group_comparison']
    print(f"  Low order mean Frobenius: {gc['low_order_mean_frobenius']:.3f}")
    print(f"  High order mean Frobenius: {gc['high_order_mean_frobenius']:.3f}")
    print(f"  Mann-Whitney p: {gc['mann_whitney_p']:.2e}")
    print(f"  Cohen's d: {gc['cohens_d']:.2f}")

    # Determine verdict
    primary_p = analysis['frobenius_correlation']['p_value']
    primary_rho = analysis['frobenius_correlation']['spearman_rho']
    cohens_d = gc['cohens_d']

    print("\n" + "="*60)
    if primary_p < 0.01 and primary_rho > 0 and cohens_d > 0.5:
        print("VERDICT: VALIDATED")
        print("High-order regions have significantly higher Fisher information")
    elif primary_p < 0.01 and primary_rho < 0:
        print("VERDICT: REFUTED (opposite direction)")
        print("High-order regions have LOWER Fisher information")
    else:
        print("VERDICT: INCONCLUSIVE")
        print(f"Effect not significant (p={primary_p:.3f}, d={cohens_d:.2f})")
    print("="*60)
