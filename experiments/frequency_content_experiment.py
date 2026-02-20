"""
RES-054: High-order CPPN images concentrate energy in low frequencies
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_low_freq_energy(img, k_cutoff=8):
    """Fraction of energy in low frequencies (k <= k_cutoff)."""
    fft = np.fft.fft2(img)
    power = np.abs(fft) ** 2

    h, w = img.shape
    low_energy = 0.0
    for i in range(h):
        for j in range(w):
            k = np.sqrt((i - h//2)**2 + (j - w//2)**2)
            if k <= k_cutoff:
                low_energy += power[i,j]

    return low_energy / np.sum(power) if np.sum(power) > 0 else 0.0

def run_experiment(n_samples=40):
    np.random.seed(42)
    cppn_energies = []
    random_energies = []
    cppn_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img_cppn = cppn.render(32)
        o = compute_order(img_cppn)
        e = compute_low_freq_energy(img_cppn)
        cppn_energies.append(e)
        cppn_orders.append(o)

        img_random = np.random.rand(32, 32)
        random_energies.append(compute_low_freq_energy(img_random))

    return {
        'cppn_energies': cppn_energies,
        'random_energies': random_energies,
        'cppn_orders': cppn_orders
    }

def analyze(results):
    cppn = np.array(results['cppn_energies'])
    rand = np.array(results['random_energies'])

    d = (np.mean(cppn) - np.mean(rand)) / np.sqrt((np.var(cppn) + np.var(rand))/2)
    t, p = stats.ttest_ind(cppn, rand)

    return {
        'effect_size': float(d),
        'p_value': float(p),
        'cppn_mean': float(np.mean(cppn)),
        'random_mean': float(np.mean(rand)),
        'status': 'validated' if p < 0.01 and d > 0.5 else 'inconclusive'
    }

if __name__ == '__main__':
    results = run_experiment()
    analysis = analyze(results)
    os.makedirs('results/frequency_content', exist_ok=True)
    with open('results/frequency_content/results.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: d={analysis['effect_size']:.2f}, p={analysis['p_value']:.2e}")
