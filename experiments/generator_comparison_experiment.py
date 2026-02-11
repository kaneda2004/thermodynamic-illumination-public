"""
RES-119: Generator Comparison (CPPN vs Other Generators)

Hypothesis: CPPNs achieve orders of magnitude higher efficiency for high-order generation

Method:
1. Generate images from CPPN, Perlin, sinusoidal, Gabor
2. Compare order distributions
3. Test: CPPN should dominate
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def perlin_noise(size=64, scale=10):
    """Simplified Perlin-like noise."""
    x = np.linspace(0, scale, size)
    y = np.linspace(0, scale, size)
    xx, yy = np.meshgrid(x, y)

    img = np.sin(xx) * np.cos(yy)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)

def sinusoidal(size=64):
    """Sinusoidal pattern."""
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    xx, yy = np.meshgrid(x, y)

    img = np.sin(xx) * np.sin(yy)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)

def gabor(size=64, frequency=0.1, angle=0):
    """Gabor-like filter."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    img = np.exp(-0.5 * (xx**2 + yy**2)) * np.cos(2*np.pi*frequency*xx)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)

def run_experiment(n_samples=50, seed=42):
    """Compare generator types."""
    np.random.seed(seed)

    cppn_orders = []
    perlin_orders = []
    sinusoidal_orders = []
    gabor_orders = []

    print(f"Comparing generators ({n_samples} samples each)...")
    for i in range(n_samples):
        # CPPN
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img = cppn.render(64)
        cppn_orders.append(compute_order(img))

        # Perlin
        img_perlin = perlin_noise()
        perlin_orders.append(compute_order(img_perlin))

        # Sinusoidal
        img_sin = sinusoidal()
        sinusoidal_orders.append(compute_order(img_sin))

        # Gabor
        img_gabor = gabor()
        gabor_orders.append(compute_order(img_gabor))

    cppn_orders = np.array(cppn_orders)
    perlin_orders = np.array(perlin_orders)
    sinusoidal_orders = np.array(sinusoidal_orders)
    gabor_orders = np.array(gabor_orders)

    # Compare CPPN vs Perlin
    t_stat, p_value = stats.ttest_ind(cppn_orders, perlin_orders)

    pooled_std = np.sqrt((np.std(cppn_orders)**2 + np.std(perlin_orders)**2) / 2)
    cohens_d = (np.mean(cppn_orders) - np.mean(perlin_orders)) / pooled_std if pooled_std > 0 else 0

    status = 'refuted' if cohens_d < 0 else 'validated'

    results = {
        'hypothesis': 'CPPNs achieve orders of magnitude higher efficiency for high-order generation',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'CPPN={np.mean(cppn_orders):.3f}, Perlin={np.mean(perlin_orders):.3f}, Sinusoid={np.mean(sinusoidal_orders):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'generator_comparison')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
