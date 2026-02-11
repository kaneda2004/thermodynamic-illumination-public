#!/usr/bin/env python3
"""
RES-073: Test effect of prior weight scale (sigma) on achievable order.

Hypothesis: Larger sigma enables higher-order structures via increased nonlinearity.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, compute_compressibility,
    Connection, Node
)

def create_cppn_with_sigma(sigma: float) -> CPPN:
    """Create CPPN with specified weight scale."""
    nodes = [
        Node(0, 'input'),
        Node(1, 'input'),
        Node(2, 'sin'),
        Node(3, 'tanh'),
        Node(4, 'sigmoid', np.random.randn() * sigma),
    ]
    connections = [
        Connection(0, 2, np.random.randn() * sigma),
        Connection(1, 2, np.random.randn() * sigma),
        Connection(0, 3, np.random.randn() * sigma),
        Connection(1, 3, np.random.randn() * sigma),
        Connection(2, 4, np.random.randn() * sigma),
        Connection(3, 4, np.random.randn() * sigma),
    ]
    for inp in [0, 1]:
        connections.append(Connection(inp, 4, np.random.randn() * sigma))

    cppn = CPPN(nodes=nodes, connections=connections, output_id=4)
    return cppn

def sample_best_order(sigma: float, n_samples: int = 200) -> tuple:
    """Sample CPPNs and return best order achieved."""
    orders = []
    for _ in range(n_samples):
        cppn = create_cppn_with_sigma(sigma)
        img = cppn.render(64)
        order = compute_compressibility(img)
        orders.append(order)
    return max(orders), np.mean(orders), np.std(orders)

def run_experiment():
    """Test sigma effect on order - focus on mean and variance."""
    np.random.seed(42)

    sigmas = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    n_trials = 10
    n_samples = 300

    results = {s: {'all_orders': []} for s in sigmas}

    print("Testing weight scales (collecting all orders)...")
    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}")
        for sigma in sigmas:
            for _ in range(n_samples // n_trials):
                cppn = create_cppn_with_sigma(sigma)
                img = cppn.render(64)
                order = compute_compressibility(img)
                results[sigma]['all_orders'].append(order)

    # Compute statistics
    print("\n=== RESULTS ===")
    print(f"Sigma -> Mean Order (std):")
    mean_orders = {}
    std_orders = {}
    for s in sigmas:
        orders = results[s]['all_orders']
        mean_orders[s] = np.mean(orders)
        std_orders[s] = np.std(orders)
        print(f"  {s:.1f}: {mean_orders[s]:.4f} (std={std_orders[s]:.4f})")

    # Correlation between sigma and mean order
    sigma_vals = []
    order_vals = []
    for s in sigmas:
        for o in results[s]['all_orders']:
            sigma_vals.append(s)
            order_vals.append(o)

    corr, p_val = stats.spearmanr(sigma_vals, order_vals)
    print(f"\nCorrelation (sigma vs order): r={corr:.3f}, p={p_val:.6f}")

    # Is there a U-shaped or inverted-U relationship?
    # Test quadratic fit
    log_sigmas = np.log(sigmas)
    means = [mean_orders[s] for s in sigmas]
    coeffs = np.polyfit(log_sigmas, means, 2)
    r2 = 1 - np.sum((np.array(means) - np.polyval(coeffs, log_sigmas))**2) / np.var(means) / len(means)

    print(f"Quadratic fit (log sigma): a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}")

    # ANOVA across sigma groups
    groups = [results[s]['all_orders'] for s in sigmas]
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.2f}, p={anova_p:.6f}")

    # Effect size: eta-squared
    ss_between = sum(len(g) * (np.mean(g) - np.mean(order_vals))**2 for g in groups)
    ss_total = np.var(order_vals) * len(order_vals)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"Effect size (eta-squared): {eta_sq:.4f}")

    # Post-hoc: is sigma=1.0 different from extremes?
    t1, p1 = stats.ttest_ind(results[1.0]['all_orders'], results[0.1]['all_orders'])
    t2, p2 = stats.ttest_ind(results[1.0]['all_orders'], results[5.0]['all_orders'])
    print(f"sigma=1.0 vs 0.1: t={t1:.2f}, p={p1:.6f}")
    print(f"sigma=1.0 vs 5.0: t={t2:.2f}, p={p2:.6f}")

    status = "VALIDATED" if anova_p < 0.01 and eta_sq > 0.01 else "REFUTED"
    print(f"\nSTATUS: {status}")

    return {
        'sigmas': sigmas,
        'mean_orders': mean_orders,
        'std_orders': std_orders,
        'correlation': corr,
        'p_value': p_val,
        'anova_p': anova_p,
        'eta_squared': eta_sq,
        'status': status
    }

if __name__ == '__main__':
    run_experiment()
