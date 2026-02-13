#!/usr/bin/env python3
"""
RES-066: Test if PNG compression ratio correlates with order metric.

Hypothesis: Higher order images (structured) should compress better with PNG
(lower file size / raw size ratio) due to spatial redundancy.

Related: RES-031 showed gzip/bz2/lzma all agree on compressibility.
"""

import numpy as np
import io
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def generate_samples(n_samples=200, size=64):
    """Generate CPPN and random samples with varying order."""
    samples = []
    orders = []

    # CPPN samples (high order)
    for _ in range(n_samples // 2):
        cppn = CPPN()  # Random CPPN
        img = cppn.render(size=size)
        order = order_multiplicative(img)
        samples.append(img)
        orders.append(order)

    # Random samples with varying blur (varying order)
    for sigma in np.linspace(0, 5, n_samples // 2):
        img = np.random.rand(size, size)
        if sigma > 0:
            img = gaussian_filter(img, sigma=sigma)
        # Threshold to binary
        img = (img > 0.5).astype(np.uint8)
        order = order_multiplicative(img)
        samples.append(img)
        orders.append(order)

    return samples, orders


def compute_png_ratio(img):
    """Compute PNG compression ratio: compressed_size / raw_size."""
    # Convert to 8-bit grayscale
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')

    # Save to bytes buffer with PNG
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', compress_level=9)
    compressed_size = buf.tell()

    # Raw size
    raw_size = img.shape[0] * img.shape[1]

    return compressed_size / raw_size


def main():
    np.random.seed(42)

    print("Generating samples...")
    samples, orders = generate_samples(n_samples=200, size=64)

    print("Computing PNG compression ratios...")
    png_ratios = [compute_png_ratio(img) for img in samples]

    orders = np.array(orders)
    png_ratios = np.array(png_ratios)

    # Pearson correlation
    r, p = stats.pearsonr(orders, png_ratios)

    # Spearman correlation (rank-based, robust)
    rho, p_spearman = stats.spearmanr(orders, png_ratios)

    # Effect size interpretation
    effect = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"

    print(f"\n=== RES-066: PNG Compression vs Order ===")
    print(f"N samples: {len(samples)}")
    print(f"Order range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"PNG ratio range: [{png_ratios.min():.3f}, {png_ratios.max():.3f}]")
    print(f"\nPearson r = {r:.4f} (p = {p:.2e})")
    print(f"Spearman rho = {rho:.4f} (p = {p_spearman:.2e})")
    print(f"Effect size: {effect}")

    # Determine status
    if p < 0.01 and abs(r) > 0.5:
        status = "VALIDATED"
    elif p < 0.05:
        status = "INCONCLUSIVE"
    else:
        status = "REFUTED"

    print(f"\nStatus: {status}")

    # Analyze CPPN vs random separately
    n_cppn = len(samples) // 2
    cppn_orders = orders[:n_cppn]
    cppn_ratios = png_ratios[:n_cppn]
    rand_orders = orders[n_cppn:]
    rand_ratios = png_ratios[n_cppn:]

    print(f"\n=== Subgroup Analysis ===")
    print(f"CPPN: order mean={np.mean(cppn_orders):.3f}, PNG ratio mean={np.mean(cppn_ratios):.3f}")
    print(f"Random: order mean={np.mean(rand_orders):.3f}, PNG ratio mean={np.mean(rand_ratios):.3f}")

    # Within-group correlations
    r_cppn, p_cppn = stats.pearsonr(cppn_orders, cppn_ratios)
    r_rand, p_rand = stats.pearsonr(rand_orders, rand_ratios)
    print(f"CPPN within-group r={r_cppn:.3f} (p={p_cppn:.2e})")
    print(f"Random within-group r={r_rand:.3f} (p={p_rand:.2e})")

    # Expected direction: higher order should mean LOWER PNG ratio (better compression)
    # BUT order_multiplicative uses compressibility as a gate, so high order = intermediate compressibility
    # PNG measures lossless compression, so high order images might have different characteristics
    direction = "negative" if r < 0 else "positive"
    print(f"\nCorrelation direction: {direction}")
    print("Note: Positive correlation expected if order metric favors intermediate PNG ratios")

    return {
        'r': float(r),
        'p': float(p),
        'rho': float(rho),
        'p_spearman': float(p_spearman),
        'n': len(samples),
        'status': status
    }


if __name__ == '__main__':
    main()
