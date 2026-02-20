"""
RES-056: Spatial Autocorrelation and Order Correlation

Hypothesis: High-order CPPN images have longer spatial correlation lengths than low-order

Method:
1. Generate CPPN samples across order spectrum
2. Compute autocorrelation length (decay to 0.1) for each image
3. Correlate autocorrelation length with order
4. Test: High-order should show longer correlation lengths
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_autocorr_length(img, direction='horizontal'):
    """Compute spatial autocorrelation decay length."""
    if direction == 'horizontal':
        slices = [img[i, :] for i in range(img.shape[0])]
    else:
        slices = [img[:, j] for j in range(img.shape[1])]

    lengths = []
    for s in slices:
        acf = np.correlate(s - np.mean(s), s - np.mean(s), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        decay_idx = np.where(acf < 0.1)[0]
        length = decay_idx[0] if len(decay_idx) > 0 else len(acf)
        lengths.append(length)

    return np.mean(lengths)

def run_experiment(n_samples=100, seed=42):
    """Sample CPPNs and correlate autocorrelation length with order."""
    np.random.seed(seed)

    orders = []
    corr_lengths = []

    print(f"Sampling {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img = cppn.render(64)

        order = compute_order(img)
        corr_len = compute_autocorr_length(img)

        orders.append(order)
        corr_lengths.append(corr_len)

    orders = np.array(orders)
    corr_lengths = np.array(corr_lengths)

    # Test correlation
    pearson_r, p_value = stats.pearsonr(orders, corr_lengths)

    # Cohens d: high-order vs low-order
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_corr = corr_lengths[high_order_mask]
    low_corr = corr_lengths[low_order_mask]

    pooled_std = np.sqrt((np.std(high_corr)**2 + np.std(low_corr)**2) / 2)
    cohens_d = (np.mean(high_corr) - np.mean(low_corr)) / pooled_std if pooled_std > 0 else 0

    # Determine status
    status = 'validated' if abs(pearson_r) > 0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPN images have longer spatial correlation lengths',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={pearson_r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'spatial_correlation')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
