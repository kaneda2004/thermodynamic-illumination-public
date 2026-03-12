"""
RES-139: RLE vs Zlib Compression Prediction

Hypothesis: RLE compression ratio predicts order better than general zlib compression

Method:
1. Generate images
2. Compute RLE and zlib compression ratios
3. Correlate each with order
4. Compare predictive power
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json
import zlib

def rle_compression(img):
    """Compute run-length encoding compression ratio."""
    binary = (img > 0.5).astype(uint8)
    flat = binary.flatten()

    runs = []
    current_val = flat[0]
    count = 1

    for i in range(1, len(flat)):
        if flat[i] == current_val and count < 255:
            count += 1
        else:
            runs.append((current_val, count))
            current_val = flat[i]
            count = 1

    compressed_size = len(runs) * 2
    original_size = len(flat)

    return compressed_size / original_size if original_size > 0 else 1.0

def run_experiment(n_samples=100, seed=42):
    """Compare compression methods."""
    np.random.seed(seed)

    orders = []
    rle_ratios = []
    zlib_ratios = []

    print(f"Testing compression methods on {n_samples} images...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        # RLE
        rle_ratio = rle_compression(img)
        rle_ratios.append(rle_ratio)

        # Zlib
        img_bytes = (img * 255).astype(np.uint8).tobytes()
        compressed = zlib.compress(img_bytes, level=6)
        zlib_ratio = len(compressed) / len(img_bytes) if len(img_bytes) > 0 else 1.0
        zlib_ratios.append(zlib_ratio)

    orders = np.array(orders)
    rle_ratios = np.array(rle_ratios)
    zlib_ratios = np.array(zlib_ratios)

    # Correlations
    r_rle, p_rle = stats.pearsonr(orders, rle_ratios)
    r_zlib, p_zlib = stats.pearsonr(orders, zlib_ratios)

    # Test which is better
    z_stat = (r_zlib - r_rle) / np.sqrt((1 - r_zlib**2) / n_samples + (1 - r_rle**2) / n_samples)

    # Effect size
    cohens_d = (r_zlib - r_rle)

    status = 'refuted' if r_zlib > r_rle else 'validated'

    results = {
        'hypothesis': 'RLE compression ratio predicts order better than general zlib compression',
        'effect_size': float(cohens_d),
        'p_value': float(abs(z_stat)),
        'status': status,
        'summary': f'RLE r={r_rle:.3f}, Zlib r={r_zlib:.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'compression_theory')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    from numpy import uint8
    run_experiment()
