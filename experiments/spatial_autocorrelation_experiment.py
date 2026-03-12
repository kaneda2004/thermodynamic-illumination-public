"""
RES-056: High-order CPPN images have longer spatial correlation lengths
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats, signal
import json

def compute_correlation_length(img):
    """Estimate spatial correlation length via autocorrelation decay."""
    row = img[img.shape[0]//2, :]
    ac = np.correlate(row - np.mean(row), row - np.mean(row), mode='full')
    ac = ac[len(ac)//2:]
    ac = ac / ac[0] if ac[0] > 0 else ac

    length = 1
    for i in range(1, len(ac)):
        if ac[i] < 0.4:
            length = i
            break

    return float(length)

def run_experiment(n_samples=40):
    np.random.seed(42)
    cppn_lengths = []
    random_lengths = []
    cppn_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img_cppn = cppn.render(32)
        o = compute_order(img_cppn)
        length = compute_correlation_length(img_cppn)
        cppn_lengths.append(length)
        cppn_orders.append(o)

        img_random = np.random.rand(32, 32)
        random_lengths.append(compute_correlation_length(img_random))

    return {
        'cppn_lengths': cppn_lengths,
        'random_lengths': random_lengths,
        'cppn_orders': cppn_orders
    }

def analyze(results):
    cppn = np.array(results['cppn_lengths'])
    rand = np.array(results['random_lengths'])
    orders = np.array(results['cppn_orders'])

    rho, p_corr = stats.spearmanr(orders, cppn)
    d = (np.mean(cppn) - np.mean(rand)) / np.sqrt((np.var(cppn) + np.var(rand))/2)

    return {
        'effect_size': float(d),
        'correlation': float(rho),
        'p_value': float(p_corr),
        'cppn_mean': float(np.mean(cppn)),
        'random_mean': float(np.mean(rand)),
        'status': 'validated' if p_corr < 0.01 and rho > 0.5 and d > 0.5 else 'inconclusive'
    }

if __name__ == '__main__':
    results = run_experiment()
    analysis = analyze(results)
    os.makedirs('results/spatial_autocorrelation', exist_ok=True)
    with open('results/spatial_autocorrelation/results.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: d={analysis['effect_size']:.2f}, r={analysis['correlation']:.3f}")
