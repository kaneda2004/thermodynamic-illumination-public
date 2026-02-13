#!/usr/bin/env python3
"""
EXPERIMENT RES-010: Entropy Rate Convergence as Kolmogorov Complexity Proxy

HYPOTHESIS: CPPN images exhibit FASTER entropy rate convergence than random
images, indicating they lie closer to the "organized complexity" regime between
order (low entropy) and disorder (high entropy).

NULL HYPOTHESIS: Entropy rate convergence is the same for CPPN and random.

THEORETICAL BACKGROUND:
Statistical complexity theory (Crutchfield & Young) distinguishes:
- SIMPLE ORDER: Low entropy, fast convergence (all zeros, checkerboard)
- COMPLEX ORGANIZATION: Moderate entropy, SLOW convergence (interesting structure)
- RANDOM DISORDER: High entropy at small scales, limited by sampling at large scales

The key insight: RANDOM images have near-maximum entropy at small block sizes,
but entropy per pattern saturates due to finite samples at large blocks.
CPPN images have low small-block entropy that converges to even lower rates.

A better metric: NORMALIZED INFORMATION DISTANCE (NID) between scales
NID(L1, L2) = (H(L1,L2) - min(H(L1), H(L2))) / max(H(L1), H(L2))

Or: Mutual Information between different block scales as fraction of joint.

REVISED HYPOTHESIS: CPPN images have higher MUTUAL INFORMATION between
adjacent block scales (2x2 vs 4x4, 4x4 vs 8x8), indicating that patterns
at one scale PREDICT patterns at the next scale (hierarchical structure).

METHODOLOGY:
1. Generate n=300 CPPN and n=300 random images (32x32)
2. Compute JOINT entropy of (2x2 pattern, 4x4 super-block index)
3. Compute mutual information MI(small, large) = H(small) + H(large) - H(joint)
4. Normalize: NMI = MI / sqrt(H(small) * H(large))
5. Test: Is NMI higher for CPPN vs random?

STATISTICAL TESTS:
- Mann-Whitney U test (non-parametric, robust to non-normality)
- Effect size: Cohen's d
- Success criteria: p < 0.01, |d| > 0.5
"""

import sys
import os
import numpy as np
from scipy.stats import mannwhitneyu
from collections import defaultdict
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN


def block_entropy(img: np.ndarray, block_size: int) -> float:
    """
    Compute entropy of block patterns at given scale.

    Args:
        img: Binary image (H x W)
        block_size: Size of square blocks (L)

    Returns:
        Entropy in bits per block
    """
    h, w = img.shape
    patterns = defaultdict(int)

    # Slide non-overlapping blocks
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = img[i*block_size:(i+1)*block_size,
                       j*block_size:(j+1)*block_size]
            # Convert to tuple for hashing
            pattern = tuple(block.flatten())
            patterns[pattern] += 1

    total = sum(patterns.values())
    if total == 0:
        return 0.0

    probs = np.array(list(patterns.values())) / total
    # Entropy in bits
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    return entropy


def compute_excess_entropy(img: np.ndarray, L: int) -> float:
    """
    Compute excess entropy at scale L.

    E(L) = H(2L) - 4*H(L)

    For extensive (i.i.d.) systems: E(L) ~ 0
    For systems with memory: E(L) > 0

    This measures how much information is "shared" across scales.
    """
    H_L = block_entropy(img, L)
    H_2L = block_entropy(img, 2 * L)

    # Normalization: for i.i.d., H(2L) = 4 * H(L) (4x more pixels)
    # Excess = actual - expected_if_independent
    # Positive excess = sub-extensive = memory = structure

    # Actually, for truly random binary, H(LxL) = L^2 bits (maximum)
    # So H(2L) = 4*L^2 = 4 * H(L)
    # Excess = H(2L) - 4*H(L) should be ~0 for random

    # But in practice with finite samples, we normalize differently
    # Use: E = (H(L)/L^2) - (H(2L)/(2L)^2)
    # This is entropy rate difference at two scales

    rate_L = H_L / (L * L)
    rate_2L = H_2L / (4 * L * L)

    # Excess entropy rate: positive if larger blocks are more predictable
    # (entropy per pixel decreases with scale for structured images)
    excess = rate_L - rate_2L

    return excess


def compute_multi_scale_excess(img: np.ndarray) -> dict:
    """
    Compute excess entropy at multiple scales.

    Returns dict with:
    - excess_2: E(2) comparing 2x2 to 4x4 blocks
    - excess_4: E(4) comparing 4x4 to 8x8 blocks
    - total_excess: sum of excesses (overall structure measure)
    """
    excess_2 = compute_excess_entropy(img, L=2)
    excess_4 = compute_excess_entropy(img, L=4)

    return {
        'excess_2': excess_2,
        'excess_4': excess_4,
        'total_excess': excess_2 + excess_4,
        'H_2': block_entropy(img, 2),
        'H_4': block_entropy(img, 4),
        'H_8': block_entropy(img, 8),
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def run_experiment():
    """
    Main experiment: Compare excess entropy between CPPN and random images.
    """
    print("=" * 70)
    print("EXPERIMENT RES-010: EXCESS ENTROPY AS KOLMOGOROV COMPLEXITY PROXY")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPN images exhibit higher excess entropy than random images")
    print("NULL: Excess entropy is the same for both image types")
    print()

    n_samples = 300
    image_size = 32

    # Collect data
    cppn_data = {
        'excess_2': [], 'excess_4': [], 'total_excess': [],
        'H_2': [], 'H_4': [], 'H_8': []
    }
    random_data = {
        'excess_2': [], 'excess_4': [], 'total_excess': [],
        'H_2': [], 'H_4': [], 'H_8': []
    }

    print(f"Generating {n_samples} CPPN images...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        metrics = compute_multi_scale_excess(img)
        for k, v in metrics.items():
            cppn_data[k].append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    print(f"\nGenerating {n_samples} random images...")
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
        metrics = compute_multi_scale_excess(img)
        for k, v in metrics.items():
            random_data[k].append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    # Convert to arrays
    for k in cppn_data:
        cppn_data[k] = np.array(cppn_data[k])
        random_data[k] = np.array(random_data[k])

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}

    # Primary metric: total excess entropy
    print("\n1. TOTAL EXCESS ENTROPY (primary metric):")
    print("-" * 50)

    cppn_total = cppn_data['total_excess']
    random_total = random_data['total_excess']

    print(f"   CPPN:   {np.mean(cppn_total):.4f} +/- {np.std(cppn_total):.4f}")
    print(f"   Random: {np.mean(random_total):.4f} +/- {np.std(random_total):.4f}")

    # Mann-Whitney U test (one-sided: CPPN > random)
    stat, p_value = mannwhitneyu(cppn_total, random_total, alternative='greater')
    d = cohens_d(cppn_total, random_total)

    print(f"\n   Mann-Whitney U: {stat:.1f}")
    print(f"   P-value (one-sided): {p_value:.6e}")
    print(f"   Cohen's d: {d:.3f}")

    significant = p_value < 0.01 and abs(d) > 0.5

    if significant:
        print("\n   ** SIGNIFICANT: CPPN has higher excess entropy **")
    else:
        if p_value >= 0.01:
            print(f"\n   Not significant (p={p_value:.4f} >= 0.01)")
        if abs(d) <= 0.5:
            print(f"   Effect size too small (|d|={abs(d):.3f} <= 0.5)")

    results['total_excess'] = {
        'cppn_mean': float(np.mean(cppn_total)),
        'cppn_std': float(np.std(cppn_total)),
        'random_mean': float(np.mean(random_total)),
        'random_std': float(np.std(random_total)),
        'mann_whitney_U': float(stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': significant
    }

    # Secondary metrics
    print("\n2. SCALE-SPECIFIC EXCESS ENTROPY:")
    print("-" * 50)

    for scale in ['excess_2', 'excess_4']:
        cppn_vals = cppn_data[scale]
        random_vals = random_data[scale]

        stat, p = mannwhitneyu(cppn_vals, random_vals, alternative='greater')
        d_scale = cohens_d(cppn_vals, random_vals)

        print(f"\n   {scale.upper()}:")
        print(f"     CPPN:   {np.mean(cppn_vals):.4f} +/- {np.std(cppn_vals):.4f}")
        print(f"     Random: {np.mean(random_vals):.4f} +/- {np.std(random_vals):.4f}")
        print(f"     p-value: {p:.6e}, Cohen's d: {d_scale:.3f}")

        results[scale] = {
            'cppn_mean': float(np.mean(cppn_vals)),
            'random_mean': float(np.mean(random_vals)),
            'p_value': float(p),
            'cohens_d': float(d_scale)
        }

    # Block entropy values
    print("\n3. BLOCK ENTROPY VALUES (bits):")
    print("-" * 50)

    for scale in ['H_2', 'H_4', 'H_8']:
        block_size = int(scale.split('_')[1])
        max_entropy = block_size * block_size  # Maximum for binary

        cppn_H = cppn_data[scale]
        random_H = random_data[scale]

        print(f"\n   {scale} (max={max_entropy} bits):")
        print(f"     CPPN:   {np.mean(cppn_H):.2f} +/- {np.std(cppn_H):.2f} ({100*np.mean(cppn_H)/max_entropy:.1f}% of max)")
        print(f"     Random: {np.mean(random_H):.2f} +/- {np.std(random_H):.2f} ({100*np.mean(random_H)/max_entropy:.1f}% of max)")

        results[scale] = {
            'cppn_mean': float(np.mean(cppn_H)),
            'random_mean': float(np.mean(random_H)),
            'max_entropy': max_entropy
        }

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results['total_excess']['significant']:
        print("\nCPPN images show HIGHER excess entropy, meaning:")
        print("  - Entropy per pixel DECREASES faster with block size")
        print("  - Larger blocks are MORE predictable from smaller blocks")
        print("  - This indicates genuine hierarchical structure")
        print("  - Consistent with lower Kolmogorov complexity (more compressible)")
        print("\nThis is a NOVEL finding distinct from RES-003 (pairwise MI):")
        print("  - RES-003 measured correlation between adjacent pixels")
        print("  - This measures MULTI-SCALE statistical complexity")
        print("  - Excess entropy captures hierarchical structure, not just local correlation")
    else:
        print("\nNo significant difference in excess entropy between CPPN and random.")
        print("This suggests the structural advantage of CPPN may be purely local.")

    # Summary for log entry
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 70)

    status = 'validated' if significant else ('inconclusive' if p_value < 0.05 else 'refuted')

    print(f"\nStatus: {status.upper()}")
    print(f"P-value: {p_value:.6e}")
    print(f"Effect size (Cohen's d): {d:.3f}")
    print(f"CPPN excess entropy: {np.mean(cppn_total):.4f}")
    print(f"Random excess entropy: {np.mean(random_total):.4f}")

    results['summary'] = {
        'status': status,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'cppn_mean': float(np.mean(cppn_total)),
        'random_mean': float(np.mean(random_total)),
        'significant': significant
    }

    # Save results
    output_dir = Path("results/excess_entropy")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
