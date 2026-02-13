"""
RES-166: Power spectral exponent correlates with Euler characteristic in CPPN images

Hypothesis: The power spectral decay exponent (beta from P(f) ~ f^beta) correlates
with the Euler characteristic (chi = #connected_components - #holes) in CPPN images.

Rationale: Spectral decay characterizes spatial smoothness/roughness. Euler characteristic
captures topological structure. If structure is fundamentally linked across scales,
these should correlate. RES-017 found spectral decay predicts order (beta=-2.65 for high-order).
RES-113 found high-order images have ~1 connected component. This experiment tests
if spectral and topological descriptors are linked.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats

def compute_power_spectral_exponent(img: np.ndarray) -> float:
    """
    Compute power spectral decay exponent via radial averaging.
    Returns beta where P(f) ~ f^beta (typically negative for natural images).
    """
    # FFT and shift
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    # Radial averaging (exclude DC)
    max_r = min(cx, cy)
    freqs = []
    powers = []
    for ri in range(1, max_r):
        mask = (r == ri)
        if np.any(mask):
            freqs.append(ri)
            powers.append(np.mean(power[mask]))

    if len(freqs) < 3:
        return 0.0

    # Log-log linear fit
    log_f = np.log(freqs)
    log_p = np.log(np.array(powers) + 1e-10)

    slope, _, r_value, p_value, _ = stats.linregress(log_f, log_p)
    return slope


def compute_euler_characteristic(img: np.ndarray) -> int:
    """
    Compute Euler characteristic using the formula:
    chi = V - E + F  (vertices - edges + faces)

    For binary image: chi = #connected_components - #holes (in foreground)

    Using pixel-based formula for 4-connectivity:
    chi = n1 - n2 + n4
    where:
    - n1 = number of foreground pixels
    - n2 = number of adjacent foreground pixel pairs (horizontal + vertical)
    - n4 = number of 2x2 foreground blocks
    """
    fg = (img == 1)

    # Count foreground pixels
    n1 = np.sum(fg)

    # Count adjacent pairs (4-connectivity)
    horiz_pairs = np.sum(fg[:, :-1] & fg[:, 1:])
    vert_pairs = np.sum(fg[:-1, :] & fg[1:, :])
    n2 = horiz_pairs + vert_pairs

    # Count 2x2 blocks
    blocks = fg[:-1, :-1] & fg[:-1, 1:] & fg[1:, :-1] & fg[1:, 1:]
    n4 = np.sum(blocks)

    # Euler characteristic
    chi = n1 - n2 + n4
    return int(chi)


def compute_connected_components(img: np.ndarray) -> int:
    """Count foreground connected components using flood fill."""
    visited = np.zeros_like(img, dtype=bool)
    count = 0

    def flood_fill(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= img.shape[0] or cj < 0 or cj >= img.shape[1]:
                continue
            if visited[ci, cj] or img[ci, cj] != 1:
                continue
            visited[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not visited[i, j] and img[i, j] == 1:
                flood_fill(i, j)
                count += 1
    return count


def count_holes(img: np.ndarray) -> int:
    """
    Count holes in foreground by computing background components
    that don't touch border, using flood fill.
    """
    h, w = img.shape
    bg = (img == 0)
    visited = np.zeros_like(bg, dtype=bool)

    # First flood fill from border to find outer background
    def flood_fill(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= h or cj < 0 or cj >= w:
                continue
            if visited[ci, cj] or not bg[ci, cj]:
                continue
            visited[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    # Flood from all border pixels
    for i in range(h):
        if bg[i, 0]:
            flood_fill(i, 0)
        if bg[i, w-1]:
            flood_fill(i, w-1)
    for j in range(w):
        if bg[0, j]:
            flood_fill(0, j)
        if bg[h-1, j]:
            flood_fill(h-1, j)

    # Count remaining unvisited background components (holes)
    holes = 0
    for i in range(h):
        for j in range(w):
            if bg[i, j] and not visited[i, j]:
                holes += 1
                # Flood fill this hole
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= h or cj < 0 or cj >= w:
                        continue
                    if visited[ci, cj] or not bg[ci, cj]:
                        continue
                    visited[ci, cj] = True
                    stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    return holes


def run_experiment(n_samples=500, seed=42):
    """Run the spectral-topology correlation experiment."""
    set_global_seed(seed)

    results = {
        'spectral_exponents': [],
        'euler_chars': [],
        'connected_components': [],
        'holes': [],
        'orders': [],
    }

    print(f"Generating {n_samples} CPPN images...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=64)

        order = order_multiplicative(img)
        beta = compute_power_spectral_exponent(img)
        chi = compute_euler_characteristic(img)
        cc = compute_connected_components(img)
        holes = count_holes(img)

        results['spectral_exponents'].append(beta)
        results['euler_chars'].append(chi)
        results['connected_components'].append(cc)
        results['holes'].append(holes)
        results['orders'].append(order)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    # Convert to numpy arrays
    betas = np.array(results['spectral_exponents'])
    chis = np.array(results['euler_chars'])
    ccs = np.array(results['connected_components'])
    holes = np.array(results['holes'])
    orders = np.array(results['orders'])

    # Primary hypothesis: spectral exponent vs Euler characteristic
    r_beta_chi, p_beta_chi = stats.spearmanr(betas, chis)

    # Secondary correlations
    r_beta_cc, p_beta_cc = stats.spearmanr(betas, ccs)
    r_beta_holes, p_beta_holes = stats.spearmanr(betas, holes)
    r_chi_order, p_chi_order = stats.spearmanr(chis, orders)
    r_beta_order, p_beta_order = stats.spearmanr(betas, orders)

    # Effect size (Cohen's d approximation from correlation)
    # d = 2r / sqrt(1 - r^2)
    def cohens_d_from_r(r):
        if abs(r) >= 0.999:
            return np.sign(r) * 10.0
        return 2 * r / np.sqrt(1 - r**2)

    d_beta_chi = cohens_d_from_r(r_beta_chi)

    # Statistics
    stats_results = {
        'n_samples': n_samples,
        'seed': seed,
        'primary_correlation': {
            'variables': 'spectral_exponent vs euler_characteristic',
            'spearman_r': float(r_beta_chi),
            'p_value': float(p_beta_chi),
            'cohens_d': float(d_beta_chi),
        },
        'secondary_correlations': {
            'beta_vs_components': {'r': float(r_beta_cc), 'p': float(p_beta_cc)},
            'beta_vs_holes': {'r': float(r_beta_holes), 'p': float(p_beta_holes)},
            'euler_vs_order': {'r': float(r_chi_order), 'p': float(p_chi_order)},
            'beta_vs_order': {'r': float(r_beta_order), 'p': float(p_beta_order)},
        },
        'descriptive_stats': {
            'beta_mean': float(np.mean(betas)),
            'beta_std': float(np.std(betas)),
            'chi_mean': float(np.mean(chis)),
            'chi_std': float(np.std(chis)),
            'cc_mean': float(np.mean(ccs)),
            'holes_mean': float(np.mean(holes)),
            'order_mean': float(np.mean(orders)),
        },
        'raw_data': {
            'spectral_exponents': [float(x) for x in betas],
            'euler_chars': [int(x) for x in chis],
            'orders': [float(x) for x in orders],
        }
    }

    # Determine outcome
    validated = abs(d_beta_chi) > 0.5 and p_beta_chi < 0.01

    print("\n" + "="*60)
    print("RESULTS: RES-166 Spectral-Topology Correlation")
    print("="*60)
    print(f"\nPrimary Hypothesis: Spectral exponent correlates with Euler characteristic")
    print(f"  Spearman r = {r_beta_chi:.4f}")
    print(f"  p-value    = {p_beta_chi:.2e}")
    print(f"  Cohen's d  = {d_beta_chi:.4f}")
    print(f"  Validated  = {validated}")

    print(f"\nSecondary Correlations:")
    print(f"  Beta vs Components: r={r_beta_cc:.3f}, p={p_beta_cc:.2e}")
    print(f"  Beta vs Holes:      r={r_beta_holes:.3f}, p={p_beta_holes:.2e}")
    print(f"  Euler vs Order:     r={r_chi_order:.3f}, p={p_chi_order:.2e}")
    print(f"  Beta vs Order:      r={r_beta_order:.3f}, p={p_beta_order:.2e}")

    print(f"\nDescriptive Statistics:")
    print(f"  Spectral exponent (beta): mean={np.mean(betas):.3f}, std={np.std(betas):.3f}")
    print(f"  Euler characteristic:     mean={np.mean(chis):.1f}, std={np.std(chis):.1f}")
    print(f"  Connected components:     mean={np.mean(ccs):.2f}")
    print(f"  Holes:                    mean={np.mean(holes):.2f}")
    print(f"  Order:                    mean={np.mean(orders):.4f}")

    return stats_results, validated


if __name__ == '__main__':
    results, validated = run_experiment(n_samples=500, seed=42)

    # Save results
    output_path = 'results/spectral_topology/spectral_euler_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Status: {'VALIDATED' if validated else 'REFUTED/INCONCLUSIVE'}")
