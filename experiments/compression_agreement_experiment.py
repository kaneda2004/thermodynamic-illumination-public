#!/usr/bin/env python3
"""
EXPERIMENT RES-023: Compression Algorithm Agreement on Image Orderliness

HYPOTHESIS: Different compression algorithms (gzip, bz2, lzma) exhibit
high pairwise rank correlation (Spearman rho > 0.8, large effect) when
ranking images by compressibility, validating compression as a robust
complexity measure.

NULL HYPOTHESIS: Compression algorithms disagree on image ordering - pairwise
Spearman correlations are weak (rho < 0.5).

THEORETICAL BACKGROUND:
Compression ratio is used as a proxy for Kolmogorov complexity throughout
this project (RES-007, RES-012). However, this assumes that different
compression algorithms agree on which images are "more compressible."

If different algorithms give inconsistent rankings, the choice of algorithm
would be arbitrary and conclusions about complexity would be algorithm-dependent.

If they agree strongly (rho > 0.8, "large" effect by Cohen's guidelines),
compression ratio is a ROBUST measure that reflects intrinsic image
complexity rather than algorithm quirks.

METHODOLOGY:
1. Generate n=500 images with varying complexity (mix of CPPN and random)
2. Compute compressibility using 3 algorithms: gzip, bz2, lzma
3. Compute all 3 pairwise Spearman rank correlations
4. Test if all correlations exceed 0.8 threshold (large effect)

SUCCESS CRITERIA:
- All pairwise rho > 0.8 (large effect threshold)
- All p < 0.01 (significance)
- Effect size d > 0.5 for structured vs unstructured images (all algorithms)

NOVELTY:
- RES-007 used single compression (zlib) to correlate with order
- RES-012 used single compression (zlib) for algorithmic MI
- This is first systematic comparison of compression algorithms
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import zlib
import bz2
import lzma
import json
from pathlib import Path
from datetime import datetime

# Try to import zstd, fall back gracefully if not available
try:
    import zstandard as zstd
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compress_gzip(data: bytes) -> int:
    """Compress using gzip (zlib) and return compressed size in bits."""
    compressed = zlib.compress(data, level=9)
    return len(compressed) * 8


def compress_bz2(data: bytes) -> int:
    """Compress using bz2 and return compressed size in bits."""
    compressed = bz2.compress(data, compresslevel=9)
    return len(compressed) * 8


def compress_lzma(data: bytes) -> int:
    """Compress using lzma and return compressed size in bits."""
    compressed = lzma.compress(data, preset=9)
    return len(compressed) * 8


def compress_zstd(data: bytes) -> int:
    """Compress using zstd and return compressed size in bits."""
    if not HAVE_ZSTD:
        return 0
    cctx = zstd.ZstdCompressor(level=22)  # Max compression
    compressed = cctx.compress(data)
    return len(compressed) * 8


def compute_compressibilities(img: np.ndarray) -> dict:
    """
    Compute compressibility using multiple algorithms.

    Compressibility = 1 - (compressed_bits / raw_bits)
    Higher = more compressible = more structured
    """
    # Tile 2x2 to overcome header overhead (same as core implementation)
    tiled = np.tile(img, (2, 2))
    packed = np.packbits(tiled.flatten())
    data = bytes(packed)
    raw_bits = tiled.size

    results = {}

    # gzip (zlib)
    gzip_bits = compress_gzip(data)
    results['gzip'] = max(0, 1 - gzip_bits / raw_bits)

    # bz2
    bz2_bits = compress_bz2(data)
    results['bz2'] = max(0, 1 - bz2_bits / raw_bits)

    # lzma
    lzma_bits = compress_lzma(data)
    results['lzma'] = max(0, 1 - lzma_bits / raw_bits)

    # zstd (if available)
    if HAVE_ZSTD:
        zstd_bits = compress_zstd(data)
        results['zstd'] = max(0, 1 - zstd_bits / raw_bits)

    return results


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000,
                 alpha: float = 0.05) -> tuple:
    """Bootstrap confidence interval for Spearman correlation."""
    n = len(x)
    correlations = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = spearmanr(x[idx], y[idx])
        correlations.append(rho)
    correlations = np.array(correlations)
    lower = np.percentile(correlations, alpha/2 * 100)
    upper = np.percentile(correlations, (1 - alpha/2) * 100)
    return lower, upper


def run_experiment():
    """
    Main experiment: Test agreement between compression algorithms.
    """
    print("=" * 70)
    print("EXPERIMENT RES-023: COMPRESSION ALGORITHM AGREEMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Different compression algorithms agree on image")
    print("            orderliness ranking (all pairwise Spearman rho > 0.8)")
    print()

    # Check for zstd
    if not HAVE_ZSTD:
        print("WARNING: zstd not available, using 3 algorithms (gzip, bz2, lzma)")
        algorithms = ['gzip', 'bz2', 'lzma']
    else:
        print("Using 4 compression algorithms: gzip, bz2, lzma, zstd")
        algorithms = ['gzip', 'bz2', 'lzma', 'zstd']
    print()

    n_samples = 500
    image_size = 32
    np.random.seed(42)

    # Generate diverse image set: mix of CPPN (structured) and random (unstructured)
    # with varying levels of Gaussian blur for random to create complexity gradient
    print(f"Generating {n_samples} images with varying complexity...")

    data = {alg: [] for alg in algorithms}
    orders = []
    image_types = []

    for i in range(n_samples):
        if i < n_samples // 2:
            # CPPN images (structured)
            cppn = CPPN()
            img = cppn.render(image_size)
            image_types.append('cppn')
        else:
            # Random images (unstructured)
            img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
            image_types.append('random')

        # Compute compressibility with all algorithms
        comp = compute_compressibilities(img)
        for alg in algorithms:
            data[alg].append(comp[alg])

        # Also compute order for reference
        orders.append(order_multiplicative(img))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    # Convert to arrays
    for alg in algorithms:
        data[alg] = np.array(data[alg])
    orders = np.array(orders)

    # Compute pairwise correlations
    print("\n" + "=" * 70)
    print("PAIRWISE SPEARMAN CORRELATIONS")
    print("=" * 70)

    n_algs = len(algorithms)
    n_pairs = n_algs * (n_algs - 1) // 2

    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'algorithms': algorithms,
        'pairwise_correlations': {},
        'all_pass_threshold': True,
        'threshold': 0.8  # Large effect threshold
    }

    print(f"\n{'Pair':<15} {'Spearman rho':>12} {'p-value':>12} {'95% CI':>20} {'Pass':>8}")
    print("-" * 70)

    all_rhos = []
    all_pass = True

    for i in range(n_algs):
        for j in range(i + 1, n_algs):
            alg1, alg2 = algorithms[i], algorithms[j]
            x, y = data[alg1], data[alg2]

            rho, p_value = spearmanr(x, y)
            ci_low, ci_high = bootstrap_ci(x, y)

            pair_name = f"{alg1}-{alg2}"
            passes = rho > 0.8 and p_value < 0.01  # Large effect threshold
            all_pass = all_pass and passes
            all_rhos.append(rho)

            results['pairwise_correlations'][pair_name] = {
                'spearman_rho': float(rho),
                'p_value': float(p_value),
                'ci_95': [float(ci_low), float(ci_high)],
                'passes_threshold': bool(passes)
            }

            status = "PASS" if passes else "FAIL"
            print(f"{pair_name:<15} {rho:>12.4f} {p_value:>12.2e} [{ci_low:>6.3f}, {ci_high:>6.3f}] {status:>8}")

    results['all_pass_threshold'] = bool(all_pass)
    results['mean_correlation'] = float(np.mean(all_rhos))
    results['min_correlation'] = float(np.min(all_rhos))

    # Summary statistics per algorithm
    print("\n" + "=" * 70)
    print("COMPRESSION STATISTICS BY ALGORITHM")
    print("=" * 70)

    print(f"\n{'Algorithm':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)

    results['algorithm_stats'] = {}
    for alg in algorithms:
        vals = data[alg]
        results['algorithm_stats'][alg] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals))
        }
        print(f"{alg:<10} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    # Correlation with order metric
    print("\n" + "=" * 70)
    print("CORRELATION WITH ORDER METRIC")
    print("=" * 70)

    print(f"\n{'Algorithm':<10} {'Spearman rho':>12} {'p-value':>12}")
    print("-" * 40)

    results['order_correlations'] = {}
    for alg in algorithms:
        rho, p = spearmanr(data[alg], orders)
        results['order_correlations'][alg] = {
            'spearman_rho': float(rho),
            'p_value': float(p)
        }
        print(f"{alg:<10} {rho:>12.4f} {p:>12.2e}")

    # CPPN vs Random comparison
    print("\n" + "=" * 70)
    print("CPPN vs RANDOM SEPARATION")
    print("=" * 70)

    cppn_mask = np.array([t == 'cppn' for t in image_types])
    random_mask = ~cppn_mask

    print(f"\n{'Algorithm':<10} {'CPPN mean':>12} {'Random mean':>12} {'Cohen d':>10}")
    print("-" * 50)

    results['cppn_vs_random'] = {}
    for alg in algorithms:
        cppn_vals = data[alg][cppn_mask]
        random_vals = data[alg][random_mask]
        d = (np.mean(cppn_vals) - np.mean(random_vals)) / np.sqrt(
            (np.var(cppn_vals) + np.var(random_vals)) / 2
        )
        results['cppn_vs_random'][alg] = {
            'cppn_mean': float(np.mean(cppn_vals)),
            'random_mean': float(np.mean(random_vals)),
            'cohens_d': float(d)
        }
        print(f"{alg:<10} {np.mean(cppn_vals):>12.4f} {np.mean(random_vals):>12.4f} {d:>10.2f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    min_rho = min(all_rhos)
    mean_rho = np.mean(all_rhos)

    if all_pass:
        print(f"\nALL pairwise correlations exceed 0.8 (large effect) threshold!")
        print(f"Mean correlation: {mean_rho:.3f}, Min: {min_rho:.3f}")
        print()
        print("Compression algorithms STRONGLY AGREE on image orderliness.")
        print()
        print("This validates compression ratio as a ROBUST complexity measure:")
        print("  - The choice of compression algorithm does not change conclusions")
        print("  - High compressibility reflects intrinsic image structure")
        print("  - Kolmogorov complexity proxy is algorithm-independent")
        print()
        print("Key insight: gzip, bz2, and lzma use fundamentally different")
        print("algorithms (LZ77, Burrows-Wheeler, LZMA), yet agree on ranking.")
        status = 'validated'
    else:
        if min_rho > 0.5:
            print(f"\nModerate agreement (min rho = {min_rho:.3f})")
            print("Algorithms agree on broad ordering but disagree on fine details.")
            status = 'inconclusive'
        else:
            print(f"\nLow agreement detected (min rho = {min_rho:.3f})")
            print("Compression algorithms DISAGREE on image ordering!")
            print("Choice of algorithm significantly affects conclusions.")
            status = 'refuted'

    results['status'] = status

    # Summary for research log
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 70)

    print(f"\nStatus: {status.upper()}")
    print(f"Mean pairwise correlation: {np.mean(all_rhos):.4f}")
    print(f"Min pairwise correlation: {np.min(all_rhos):.4f}")
    print(f"All pairs pass threshold (rho > 0.8): {all_pass}")
    print(f"N algorithms: {len(algorithms)}")
    print(f"N pairs tested: {n_pairs}")

    results['summary'] = {
        'status': status,
        'mean_rho': float(np.mean(all_rhos)),
        'min_rho': float(np.min(all_rhos)),
        'all_pass': bool(all_pass),
        'n_algorithms': len(algorithms),
        'n_pairs': n_pairs
    }

    # Save results
    output_dir = Path("results/compression_agreement")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "compression_agreement_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/compression_agreement_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
