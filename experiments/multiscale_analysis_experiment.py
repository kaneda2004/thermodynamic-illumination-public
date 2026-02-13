"""
RES-117: Multiscale Analysis and Order

Hypothesis: CPPN image order correlates with wavelet energy in low-frequency bands

Method:
1. Generate CPPNs and random images
2. Compute wavelet decomposition
3. Measure energy decay rate
4. Correlate with order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_wavelet_decay(img, levels=3):
    """Compute wavelet energy decay rate (simplified)."""
    energies = []

    for level in range(levels):
        # Simplified: compute variance at each scale
        if level == 0:
            energy = np.var(img)
        else:
            # Downsample and measure
            downsampled = img[::2**level, ::2**level]
            energy = np.var(downsampled) if downsampled.size > 0 else 0

        energies.append(energy)

    # Decay rate (negative slope)
    if len(energies) > 1:
        decay_rate = (energies[-1] - energies[0]) / len(energies)
    else:
        decay_rate = 0

    return -decay_rate  # Return negative for decreasing energy

def run_experiment(n_samples=100, seed=42):
    """Test wavelet decay vs order."""
    np.random.seed(seed)

    orders = []
    decay_rates = []

    print(f"Analyzing wavelet decay in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        decay = compute_wavelet_decay(img)
        decay_rates.append(decay)

    orders = np.array(orders)
    decay_rates = np.array(decay_rates)

    # Correlation
    r, p_value = stats.pearsonr(orders, decay_rates)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_decay = decay_rates[high_order_mask]
    low_decay = decay_rates[low_order_mask]

    pooled_std = np.sqrt((np.std(high_decay)**2 + np.std(low_decay)**2) / 2)
    cohens_d = (np.mean(high_decay) - np.mean(low_decay)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if r > 0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'CPPN image order correlates with wavelet energy in low-frequency bands',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'multiscale_analysis')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
