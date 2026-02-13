"""
RES-168: Row-wise vs column-wise compression in CPPN images

HYPOTHESIS: Row-wise compression ratio predicts order better than column-wise in CPPN images

RATIONALE:
CPPNs generate images by evaluating smooth coordinate functions. The rendering typically
iterates row-by-row (fixed y, varying x). If CPPN smoothness is axis-aligned, we'd expect
row-wise compression to be more predictive of order than column-wise.

This tests whether the compression benefit comes from horizontal coherence (adjacent pixels
in same row) vs vertical coherence (adjacent pixels in same column).

METHODOLOGY:
1. Generate N CPPN images with varying order
2. For each image, compute:
   - Row-wise compression: compress each row separately, average ratio
   - Column-wise compression: compress each column separately, average ratio
   - Full image compression (baseline)
3. Correlate each compression type with order_multiplicative
4. Compare correlations using Fisher z-test
"""

import numpy as np
import zlib
from scipy import stats
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, compute_compressibility, set_global_seed


def compress_ratio(data: bytes) -> float:
    """Compute compression ratio (1 - compressed/original)."""
    if len(data) == 0:
        return 0.0
    compressed = zlib.compress(data, level=9)
    return 1 - len(compressed) / len(data)


def row_major_compression(img: np.ndarray) -> float:
    """Compression ratio when scanning image row-by-row (default C order).

    This captures horizontal coherence - adjacent pixels in same row are
    adjacent in the byte stream.
    """
    # Tile 2x2 to overcome zlib header overhead (same as compute_compressibility)
    tiled = np.tile(img, (2, 2))
    packed = np.packbits(tiled.flatten(order='C'))  # Row-major
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


def col_major_compression(img: np.ndarray) -> float:
    """Compression ratio when scanning image column-by-column (Fortran order).

    This captures vertical coherence - adjacent pixels in same column are
    adjacent in the byte stream.
    """
    tiled = np.tile(img, (2, 2))
    packed = np.packbits(tiled.flatten(order='F'))  # Column-major
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


def hilbert_order(n: int) -> list[tuple[int, int]]:
    """Generate Hilbert curve coordinates for nxn grid (n must be power of 2)."""
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y

    coords = []
    for i in range(n * n):
        x, y = 0, 0
        s = 1
        t = i
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        coords.append((y, x))
    return coords


def hilbert_compression(img: np.ndarray) -> float:
    """Compression ratio when scanning image in Hilbert curve order.

    This captures 2D locality - nearby pixels in 2D space tend to be
    adjacent in the byte stream.
    """
    tiled = np.tile(img, (2, 2))
    n = tiled.shape[0]
    coords = hilbert_order(n)
    hilbert_flat = np.array([tiled[i, j] for i, j in coords], dtype=np.uint8)
    packed = np.packbits(hilbert_flat)
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


def diagonal_compression(img: np.ndarray) -> float:
    """Compression ratio when scanning image along diagonals.

    This captures diagonal coherence patterns.
    """
    tiled = np.tile(img, (2, 2))
    n = tiled.shape[0]
    diag_flat = []
    # Scan anti-diagonals
    for d in range(2 * n - 1):
        for i in range(max(0, d - n + 1), min(d + 1, n)):
            j = d - i
            diag_flat.append(tiled[i, j])
    diag_flat = np.array(diag_flat, dtype=np.uint8)
    packed = np.packbits(diag_flat)
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


def generate_samples(n_samples: int = 500, seed: int = 42) -> list[dict]:
    """Generate CPPN samples with compression metrics."""
    set_global_seed(seed)
    results = []

    for i in range(n_samples):
        # Random CPPN
        cppn = CPPN()
        img = cppn.render(size=32)

        order = order_multiplicative(img)
        row_comp = row_major_compression(img)
        col_comp = col_major_compression(img)
        hilbert_comp = hilbert_compression(img)
        diag_comp = diagonal_compression(img)
        tiled_comp = compute_compressibility(img)  # Uses 2x2 tiling (row-major)

        results.append({
            'order': order,
            'row_compression': row_comp,
            'col_compression': col_comp,
            'hilbert_compression': hilbert_comp,
            'diagonal_compression': diag_comp,
            'tiled_compression': tiled_comp,
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")

    return results


def fisher_z_test(r1: float, r2: float, n: int) -> tuple[float, float]:
    """
    Fisher z-test for comparing two correlation coefficients from same sample.

    Returns (z_statistic, p_value).
    """
    # Fisher z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1 + 1e-10))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2 + 1e-10))

    # Standard error for dependent correlations (conservative estimate)
    se = np.sqrt(2 / (n - 3))

    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def run_experiment():
    print("RES-168: Row-major vs column-major compression scan orders")
    print("=" * 60)

    # Generate samples
    print("\nGenerating CPPN samples...")
    samples = generate_samples(n_samples=500, seed=42)

    # Extract arrays
    orders = np.array([s['order'] for s in samples])
    row_comps = np.array([s['row_compression'] for s in samples])
    col_comps = np.array([s['col_compression'] for s in samples])
    hilbert_comps = np.array([s['hilbert_compression'] for s in samples])
    diag_comps = np.array([s['diagonal_compression'] for s in samples])
    tiled_comps = np.array([s['tiled_compression'] for s in samples])

    # Compute correlations with order
    r_row, p_row = stats.pearsonr(row_comps, orders)
    r_col, p_col = stats.pearsonr(col_comps, orders)
    r_hilbert, p_hilbert = stats.pearsonr(hilbert_comps, orders)
    r_diag, p_diag = stats.pearsonr(diag_comps, orders)
    r_tiled, p_tiled = stats.pearsonr(tiled_comps, orders)

    print(f"\nCorrelations with order_multiplicative:")
    print(f"  Row-major scan:     r = {r_row:.4f}, p = {p_row:.2e}")
    print(f"  Col-major scan:     r = {r_col:.4f}, p = {p_col:.2e}")
    print(f"  Hilbert scan:       r = {r_hilbert:.4f}, p = {p_hilbert:.2e}")
    print(f"  Diagonal scan:      r = {r_diag:.4f}, p = {p_diag:.2e}")
    print(f"  Tiled (baseline):   r = {r_tiled:.4f}, p = {p_tiled:.2e}")

    # Compare row vs column
    z_stat, p_diff = fisher_z_test(r_row, r_col, len(samples))
    print(f"\nRow vs Column comparison:")
    print(f"  Fisher z-stat = {z_stat:.4f}, p = {p_diff:.4e}")

    # Compare row vs Hilbert
    z_row_hilbert, p_row_hilbert = fisher_z_test(r_row, r_hilbert, len(samples))
    print(f"\nRow vs Hilbert comparison:")
    print(f"  Fisher z-stat = {z_row_hilbert:.4f}, p = {p_row_hilbert:.4e}")

    # Compute row-column correlation (how similar are they?)
    r_row_col, p_row_col = stats.pearsonr(row_comps, col_comps)
    print(f"\nRow-Column compression correlation: r = {r_row_col:.4f}, p = {p_row_col:.2e}")

    # Summary statistics
    print(f"\nSummary statistics:")
    print(f"  Row-major:   mean = {row_comps.mean():.4f}, std = {row_comps.std():.4f}")
    print(f"  Col-major:   mean = {col_comps.mean():.4f}, std = {col_comps.std():.4f}")
    print(f"  Hilbert:     mean = {hilbert_comps.mean():.4f}, std = {hilbert_comps.std():.4f}")
    print(f"  Diagonal:    mean = {diag_comps.mean():.4f}, std = {diag_comps.std():.4f}")

    # Effect size: correlation difference
    effect_size_row_col = abs(r_row - r_col)

    # Best scan order
    scan_results = [
        ('row', r_row),
        ('col', r_col),
        ('hilbert', r_hilbert),
        ('diagonal', r_diag),
    ]
    best_scan = max(scan_results, key=lambda x: abs(x[1]))
    worst_scan = min(scan_results, key=lambda x: abs(x[1]))

    # Determine outcome
    if p_diff < 0.01 and effect_size_row_col > 0.05:
        if abs(r_row) > abs(r_col):
            status = "validated"
            summary = f"Row-major compression correlates better with order than column-major (r={r_row:.3f} vs r={r_col:.3f}, z={z_stat:.2f}, p={p_diff:.2e}). Horizontal coherence dominates CPPN structure."
        else:
            status = "refuted"
            summary = f"Column-major compression correlates BETTER with order (r={r_col:.3f} vs r={r_row:.3f}, z={z_stat:.2f}, p={p_diff:.2e}). Vertical coherence dominates."
    else:
        status = "refuted"
        summary = f"Row and column scans correlate equally with order (r={r_row:.3f} vs r={r_col:.3f}, p={p_diff:.2e}). CPPN compression is axis-symmetric - no preferred scan direction. Best: {best_scan[0]} (r={best_scan[1]:.3f})."

    print(f"\n{'='*60}")
    print(f"STATUS: {status.upper()}")
    print(f"SUMMARY: {summary}")

    # Save results
    os.makedirs("results/row_vs_col_compression", exist_ok=True)
    results = {
        'experiment_id': 'RES-168',
        'hypothesis': 'Row-wise compression ratio predicts order better than column-wise in CPPN images',
        'domain': 'compression_mechanics',
        'status': status,
        'metrics': {
            'r_row_order': float(r_row),
            'r_col_order': float(r_col),
            'r_hilbert_order': float(r_hilbert),
            'r_diagonal_order': float(r_diag),
            'r_tiled_order': float(r_tiled),
            'z_stat_row_vs_col': float(z_stat),
            'p_value_row_vs_col': float(p_diff),
            'effect_size_row_col': float(effect_size_row_col),
            'r_row_col': float(r_row_col),
            'n_samples': len(samples),
            'best_scan_order': best_scan[0],
            'worst_scan_order': worst_scan[0],
        },
        'summary': summary,
    }

    with open("results/row_vs_col_compression/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/row_vs_col_compression/results.json")

    return results


if __name__ == "__main__":
    run_experiment()
