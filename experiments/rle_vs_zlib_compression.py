"""
RES-139: RLE compression ratio predicts order better than general zlib compression

Hypothesis: Run-length encoding (RLE) compression ratio predicts CPPN image order
better than general-purpose zlib compression because RLE specifically captures
horizontal/vertical structure which the order metric measures.

Approach:
1. Generate N=500 CPPN images with varying order
2. Compute both RLE compression ratio and zlib compression ratio
3. Correlate each with order metric
4. Compare correlation strengths using Fisher z-test
"""

import numpy as np
import zlib
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, Node, Connection, PRIOR_SIGMA
from scipy import stats
import json
import os


def generate_cppn_image(cppn, resolution=64):
    """Generate continuous image from CPPN."""
    coords = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)


def rle_compress(img_binary):
    """Compute run-length encoding compression ratio for binary image."""
    flat = img_binary.flatten()
    if len(flat) == 0:
        return 1.0

    # Count runs
    runs = 1
    for i in range(1, len(flat)):
        if flat[i] != flat[i-1]:
            runs += 1

    # RLE stores (value, count) pairs - simplistically 2 bytes per run
    # Original: 1 bit per pixel (8 pixels per byte)
    original_bytes = len(flat) / 8
    rle_bytes = runs * 2  # 1 byte value, 1 byte count (simplified)

    return rle_bytes / max(original_bytes, 1)


def zlib_compress_ratio(img_binary):
    """Compute zlib compression ratio for binary image."""
    flat = img_binary.flatten().astype(np.uint8)
    original = len(flat)
    compressed = len(zlib.compress(flat.tobytes(), level=9))
    return compressed / max(original, 1)


def run_experiment(n_samples=500, resolution=64, seed=42):
    """Run the RLE vs zlib compression experiment."""
    np.random.seed(seed)

    orders = []
    rle_ratios = []
    zlib_ratios = []

    for i in range(n_samples):
        # Generate CPPN with varying depth for order diversity
        depth = np.random.randint(0, 5)
        cppn = CPPN()
        for _ in range(depth):
            hidden_id = len(cppn.nodes)
            act = np.random.choice(['sin', 'tanh', 'gauss', 'sigmoid'])
            cppn.nodes.append(Node(hidden_id, act, np.random.randn() * PRIOR_SIGMA))
            # Connect random input to hidden
            inp = np.random.choice(cppn.input_ids)
            cppn.connections.append(Connection(inp, hidden_id, np.random.randn() * PRIOR_SIGMA))
            # Connect hidden to output
            cppn.connections.append(Connection(hidden_id, cppn.output_id, np.random.randn() * PRIOR_SIGMA))

        # Generate image and binarize
        img = generate_cppn_image(cppn, resolution)
        binary = (img > 0.5).astype(np.uint8)

        # Compute order on binary image
        order = order_multiplicative(binary)

        # Compute compression ratios
        rle = rle_compress(binary)
        zlb = zlib_compress_ratio(binary)

        orders.append(order)
        rle_ratios.append(rle)
        zlib_ratios.append(zlb)

    orders = np.array(orders)
    rle_ratios = np.array(rle_ratios)
    zlib_ratios = np.array(zlib_ratios)

    # Correlations
    rle_corr, rle_p = stats.spearmanr(rle_ratios, orders)
    zlib_corr, zlib_p = stats.spearmanr(zlib_ratios, orders)

    # Fisher z-test to compare correlations
    n = len(orders)
    z_rle = np.arctanh(rle_corr)
    z_zlib = np.arctanh(zlib_corr)
    se = np.sqrt(2 / (n - 3))
    z_diff = (z_rle - z_zlib) / se
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    # Effect size: difference in absolute correlations
    effect_size = abs(rle_corr) - abs(zlib_corr)

    results = {
        'n_samples': n_samples,
        'resolution': resolution,
        'rle_order_corr': float(rle_corr),
        'rle_order_p': float(rle_p),
        'zlib_order_corr': float(zlib_corr),
        'zlib_order_p': float(zlib_p),
        'fisher_z_diff': float(z_diff),
        'p_diff': float(p_diff),
        'effect_size_corr_diff': float(effect_size),
        'order_mean': float(np.mean(orders)),
        'order_std': float(np.std(orders)),
        'rle_mean': float(np.mean(rle_ratios)),
        'zlib_mean': float(np.mean(zlib_ratios)),
    }

    # Determine outcome
    # Hypothesis: RLE predicts order BETTER than zlib (higher correlation magnitude)
    if p_diff < 0.01 and abs(rle_corr) > abs(zlib_corr):
        results['validated'] = True
        results['status'] = 'validated'
    elif p_diff < 0.01 and abs(zlib_corr) > abs(rle_corr):
        results['validated'] = False
        results['status'] = 'refuted'  # zlib is actually better
    else:
        results['validated'] = False
        results['status'] = 'inconclusive'  # no significant difference

    return results


if __name__ == '__main__':
    results = run_experiment()

    # Save results
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/rle_vs_zlib', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/rle_vs_zlib/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== RES-139: RLE vs Zlib Compression ===")
    print(f"Samples: {results['n_samples']}")
    print(f"\nCorrelations with order:")
    print(f"  RLE ratio:  r = {results['rle_order_corr']:.3f} (p = {results['rle_order_p']:.2e})")
    print(f"  Zlib ratio: r = {results['zlib_order_corr']:.3f} (p = {results['zlib_order_p']:.2e})")
    print(f"\nFisher z-test for correlation difference:")
    print(f"  z = {results['fisher_z_diff']:.3f}, p = {results['p_diff']:.4f}")
    print(f"  Effect (|r_rle| - |r_zlib|) = {results['effect_size_corr_diff']:.3f}")
    print(f"\nStatus: {results['status'].upper()}")
