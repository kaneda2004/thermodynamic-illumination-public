"""
RES-049: Fisher information is higher in high-order regions of order landscape

Hypothesis: Fisher information correlates with order in CPPN images
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_fisher_metrics(cppn, weights, eps=0.01):
    """Compute Frobenius norm of Hessian as Fisher information."""
    n = len(weights)
    H = np.zeros((n, n))

    def order_at(w):
        c = CPPN()
        c.set_weights(w)
        return compute_order(c.render(32))

    f0 = order_at(weights)
    for i in range(min(n, 5)):  # Sample subset for speed
        for j in range(i, min(n, 5)):
            w_pp, w_pm, w_mp, w_mm = weights.copy(), weights.copy(), weights.copy(), weights.copy()
            w_pp[i] += eps; w_pp[j] += eps
            w_pm[i] += eps; w_pm[j] -= eps
            w_mp[i] -= eps; w_mp[j] += eps
            w_mm[i] -= eps; w_mm[j] -= eps
            H[i,j] = (order_at(w_pp) - order_at(w_pm) - order_at(w_mp) + order_at(w_mm)) / (4*eps**2)
            H[j,i] = H[i,j]
    return np.linalg.norm(H, 'fro')

def run_experiment(n_samples=30):
    np.random.seed(42)
    results = []
    for i in range(n_samples):
        cppn = CPPN()
        w = cppn.get_weights()
        o = compute_order(cppn.render(32))
        f = compute_fisher_metrics(cppn, w)
        results.append({'order': o, 'fisher': f})
    return results

def analyze(results):
    orders = np.array([r['order'] for r in results])
    fisher = np.array([r['fisher'] for r in results])

    rho, p = stats.spearmanr(orders, fisher)
    low = orders < np.median(orders)
    d = (np.mean(fisher[~low]) - np.mean(fisher[low])) / np.sqrt((np.var(fisher[~low]) + np.var(fisher[low]))/2)

    return {
        'correlation': float(rho),
        'p_value': float(p),
        'effect_size': float(d),
        'status': 'validated' if p < 0.01 and rho > 0.5 and d > 0.5 else 'inconclusive'
    }

if __name__ == '__main__':
    results = run_experiment()
    analysis = analyze(results)
    os.makedirs('results/information_geometry', exist_ok=True)
    with open('results/information_geometry/results.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: correlation={analysis['correlation']:.3f}, p={analysis['p_value']:.2e}, d={analysis['effect_size']:.2f}")
