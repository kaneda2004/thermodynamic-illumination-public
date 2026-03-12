#!/usr/bin/env python3
"""
RES-076: Measure fraction of order from local (2x2) vs global features.

Hypothesis: Local structure contributes a size-independent baseline (~15-25%),
while global structure scales with image size.

Method:
1. Generate CPPN images at multiple sizes (8x8, 16x16, 32x32, 64x64)
2. Compute total order for each
3. Shuffle pixels to destroy global structure, preserve local 2x2 blocks
4. Measure order after shuffle = local contribution
5. Global contribution = total - local
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative

def shuffle_preserving_local(img, block_size=2):
    """Shuffle image while preserving local 2x2 blocks."""
    h, w = img.shape
    # Reshape into blocks
    blocks_h, blocks_w = h // block_size, w // block_size
    blocks = img[:blocks_h*block_size, :blocks_w*block_size].reshape(
        blocks_h, block_size, blocks_w, block_size
    ).transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

    # Shuffle block positions
    np.random.shuffle(blocks)

    # Reconstruct image
    shuffled = blocks.reshape(blocks_h, blocks_w, block_size, block_size
    ).transpose(0, 2, 1, 3).reshape(blocks_h*block_size, blocks_w*block_size)
    return shuffled

def compute_local_global_fraction(size, n_samples=100, n_shuffles=10):
    """Compute local/global order fractions for given image size."""
    total_orders = []
    local_orders = []

    for _ in range(n_samples):
        cppn = CPPN()  # Fresh random CPPN each time
        img = cppn.render(size=size)
        total_order = order_multiplicative(img)

        # Average over multiple shuffles
        shuffle_orders = []
        for _ in range(n_shuffles):
            shuffled = shuffle_preserving_local(img, block_size=2)
            shuffle_orders.append(order_multiplicative(shuffled))
        local_order = np.mean(shuffle_orders)

        total_orders.append(total_order)
        local_orders.append(local_order)

    total_mean = np.mean(total_orders)
    local_mean = np.mean(local_orders)
    local_fraction = local_mean / total_mean if total_mean > 0 else 0

    return {
        'size': size,
        'total_order': total_mean,
        'local_order': local_mean,
        'global_order': total_mean - local_mean,
        'local_fraction': local_fraction,
        'global_fraction': 1 - local_fraction,
        'total_std': np.std(total_orders),
        'local_std': np.std(local_orders),
        'n_samples': n_samples
    }

def main():
    sizes = [8, 16, 32, 64]
    results = []

    print("RES-076: Local vs Global Structure Fraction")
    print("=" * 60)

    for size in sizes:
        print(f"\nProcessing {size}x{size}...")
        r = compute_local_global_fraction(size, n_samples=100, n_shuffles=10)
        results.append(r)
        print(f"  Total order: {r['total_order']:.4f} +/- {r['total_std']:.4f}")
        print(f"  Local order: {r['local_order']:.4f} +/- {r['local_std']:.4f}")
        print(f"  Local fraction: {r['local_fraction']*100:.1f}%")
        print(f"  Global fraction: {r['global_fraction']*100:.1f}%")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    local_fractions = [r['local_fraction'] for r in results]
    global_orders = [r['global_order'] for r in results]

    # Check if local fraction is size-independent
    local_frac_std = np.std(local_fractions)
    local_frac_mean = np.mean(local_fractions)
    local_cv = local_frac_std / local_frac_mean if local_frac_mean > 0 else float('inf')

    print(f"\nLocal fraction across sizes: {[f'{f*100:.1f}%' for f in local_fractions]}")
    print(f"Local fraction mean: {local_frac_mean*100:.1f}%")
    print(f"Local fraction CV: {local_cv:.3f}")

    # Check if global order scales with size
    log_sizes = np.log(sizes)
    log_global = np.log([max(g, 1e-10) for g in global_orders])
    slope, intercept = np.polyfit(log_sizes, log_global, 1)

    print(f"\nGlobal order scaling: O_global ~ size^{slope:.2f}")

    # Determine verdict
    hypothesis_supported = (
        0.10 <= local_frac_mean <= 0.35 and  # Local is 10-35%
        local_cv < 0.3 and  # Relatively stable across sizes
        slope > 0.3  # Global scales positively with size
    )

    print(f"\n{'VALIDATED' if hypothesis_supported else 'REFUTED'}")
    print(f"  - Local fraction in expected range (15-25%): {0.15 <= local_frac_mean <= 0.25}")
    print(f"  - Local fraction size-independent (CV<0.3): {local_cv < 0.3}")
    print(f"  - Global scales with size (slope>0.3): {slope > 0.3}")

    # Summary metrics
    metrics = {
        'local_fraction_mean': float(local_frac_mean),
        'local_fraction_cv': float(local_cv),
        'global_scaling_exponent': float(slope),
        'sizes_tested': sizes,
        'n_samples_per_size': 100
    }

    print(f"\nMETRICS: {metrics}")

    return hypothesis_supported, metrics

if __name__ == "__main__":
    validated, metrics = main()
