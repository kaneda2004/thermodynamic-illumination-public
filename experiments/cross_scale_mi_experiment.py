#!/usr/bin/env python3
"""
EXPERIMENT RES-010: Cross-Scale Mutual Information as Kolmogorov Complexity Proxy

HYPOTHESIS: CPPN images exhibit higher cross-scale mutual information than
random images, indicating hierarchical predictability across spatial scales.

NULL HYPOTHESIS: Cross-scale MI is the same for CPPN and random images.

THEORETICAL BACKGROUND:
Kolmogorov complexity K(x) measures the length of the shortest program that
generates x. For images, K is approximated by compression. However, compression
only captures overall redundancy, not HIERARCHICAL structure.

Cross-scale mutual information (CSMI) captures a different aspect:
How well do SMALL patterns predict LARGE patterns?

For i.i.d. random images: CSMI ~ 0 (small blocks don't predict large blocks)
For structured images: CSMI > 0 (hierarchical dependencies exist)

This is particularly relevant for CPPN images because:
1. CPPNs are compositional (functions of functions)
2. This creates multi-scale structure
3. Small-scale features should predict large-scale features

METHODOLOGY:
1. Generate n=300 CPPN and n=300 random images (32x32)
2. For each image, extract (position, 2x2 pattern, 4x4 super-pattern) tuples
3. Compute MI between 2x2 patterns and their 4x4 super-block context
4. Normalize: NMI = 2*MI / (H(2x2) + H(4x4))
5. Test: Is NMI higher for CPPN vs random?

STATISTICAL TESTS:
- Mann-Whitney U test (non-parametric)
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


def entropy_from_counts(counts: dict) -> float:
    """Compute entropy from a dictionary of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-12))


def compute_cross_scale_mi(img: np.ndarray, small_size: int = 2, large_size: int = 4) -> dict:
    """
    Compute mutual information between small-scale and large-scale block patterns.

    For each large block, we look at the small blocks within it.
    MI measures how much knowing the large block tells us about small blocks.

    Args:
        img: Binary image (H x W), must be divisible by large_size
        small_size: Size of small blocks (default 2x2)
        large_size: Size of large blocks (default 4x4), must be multiple of small_size

    Returns:
        dict with MI, normalized MI, and component entropies
    """
    h, w = img.shape
    assert large_size % small_size == 0, "large_size must be multiple of small_size"

    # Count joint occurrences of (large_pattern, small_pattern_within_position)
    # Position within large block matters for capturing spatial structure
    small_per_large = large_size // small_size  # e.g., 2 for 4x4/2x2

    # Joint counts: (large_pattern, position_in_large, small_pattern)
    joint_counts = defaultdict(int)
    large_counts = defaultdict(int)
    small_counts = defaultdict(int)  # marginal over all positions
    small_pos_counts = defaultdict(lambda: defaultdict(int))  # per position

    n_large_h = h // large_size
    n_large_w = w // large_size

    for li in range(n_large_h):
        for lj in range(n_large_w):
            # Extract large block
            large_block = img[li*large_size:(li+1)*large_size,
                             lj*large_size:(lj+1)*large_size]
            large_pattern = tuple(large_block.flatten())
            large_counts[large_pattern] += 1

            # Extract small blocks within
            for si in range(small_per_large):
                for sj in range(small_per_large):
                    small_block = large_block[si*small_size:(si+1)*small_size,
                                             sj*small_size:(sj+1)*small_size]
                    small_pattern = tuple(small_block.flatten())
                    position = si * small_per_large + sj

                    joint_key = (large_pattern, position, small_pattern)
                    joint_counts[joint_key] += 1
                    small_counts[small_pattern] += 1
                    small_pos_counts[position][small_pattern] += 1

    # Compute entropies
    H_large = entropy_from_counts(large_counts)

    # Average small block entropy (over positions)
    H_small_avg = 0
    for pos in range(small_per_large * small_per_large):
        H_small_avg += entropy_from_counts(small_pos_counts[pos])
    H_small_avg /= (small_per_large * small_per_large)

    H_small_total = entropy_from_counts(small_counts)

    # Joint entropy H(large, position, small)
    H_joint = entropy_from_counts(joint_counts)

    # Mutual information: I(large; small|position) = H(large) + H(small|pos) - H(large, small, pos)
    # But since position is determined by enumeration, we use:
    # I(large; small) = H(large) + H(small) - H(large, small)
    # where H(small) is average per-position entropy

    # Simpler: just compute I(large_pattern; small_patterns_in_it)
    # For a specific position p: I = H(large) + H(small_at_p) - H(large, small_at_p)

    # Let's compute per-position MI and average
    mi_per_position = []
    for pos in range(small_per_large * small_per_large):
        # Joint counts for this position
        joint_pos = defaultdict(int)
        for (lp, p, sp), count in joint_counts.items():
            if p == pos:
                joint_pos[(lp, sp)] += count

        H_joint_pos = entropy_from_counts(joint_pos)
        H_small_pos = entropy_from_counts(small_pos_counts[pos])

        # MI(large; small_at_pos) = H(large) + H(small_at_pos) - H(large, small_at_pos)
        mi = H_large + H_small_pos - H_joint_pos
        mi_per_position.append(mi)

    avg_mi = np.mean(mi_per_position)

    # Normalized MI (0 to 1 scale)
    # NMI = 2*MI / (H(X) + H(Y))
    nmi = 2 * avg_mi / (H_large + H_small_avg + 1e-12) if (H_large + H_small_avg) > 0 else 0

    return {
        'mi': avg_mi,
        'nmi': nmi,
        'H_large': H_large,
        'H_small_avg': H_small_avg,
        'mi_per_position': mi_per_position,
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
    Main experiment: Compare cross-scale MI between CPPN and random images.
    """
    print("=" * 70)
    print("EXPERIMENT RES-010: CROSS-SCALE MUTUAL INFORMATION")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPN images exhibit higher cross-scale MI than random")
    print("            (small patterns predict large patterns better)")
    print("NULL: Cross-scale MI is the same for both image types")
    print()

    n_samples = 300
    image_size = 32

    # Collect data
    cppn_data = {'mi': [], 'nmi': [], 'H_large': [], 'H_small': []}
    random_data = {'mi': [], 'nmi': [], 'H_large': [], 'H_small': []}

    print(f"Generating {n_samples} CPPN images...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        metrics = compute_cross_scale_mi(img, small_size=2, large_size=4)
        cppn_data['mi'].append(metrics['mi'])
        cppn_data['nmi'].append(metrics['nmi'])
        cppn_data['H_large'].append(metrics['H_large'])
        cppn_data['H_small'].append(metrics['H_small_avg'])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    print(f"\nGenerating {n_samples} random images...")
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
        metrics = compute_cross_scale_mi(img, small_size=2, large_size=4)
        random_data['mi'].append(metrics['mi'])
        random_data['nmi'].append(metrics['nmi'])
        random_data['H_large'].append(metrics['H_large'])
        random_data['H_small'].append(metrics['H_small_avg'])
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

    # Primary metric: normalized mutual information
    print("\n1. NORMALIZED CROSS-SCALE MI (primary metric):")
    print("-" * 50)

    cppn_nmi = cppn_data['nmi']
    random_nmi = random_data['nmi']

    print(f"   CPPN:   {np.mean(cppn_nmi):.4f} +/- {np.std(cppn_nmi):.4f}")
    print(f"   Random: {np.mean(random_nmi):.4f} +/- {np.std(random_nmi):.4f}")

    # Mann-Whitney U test (one-sided: CPPN > random)
    stat, p_value = mannwhitneyu(cppn_nmi, random_nmi, alternative='greater')
    d = cohens_d(cppn_nmi, random_nmi)

    print(f"\n   Mann-Whitney U: {stat:.1f}")
    print(f"   P-value (one-sided): {p_value:.6e}")
    print(f"   Cohen's d: {d:.3f}")

    significant = p_value < 0.01 and abs(d) > 0.5

    if significant:
        print("\n   ** SIGNIFICANT: CPPN has higher cross-scale MI **")
    else:
        if p_value >= 0.01:
            print(f"\n   Not significant (p={p_value:.4f} >= 0.01)")
        if abs(d) <= 0.5:
            print(f"   Small effect size (|d|={abs(d):.3f} <= 0.5)")

    results['nmi'] = {
        'cppn_mean': float(np.mean(cppn_nmi)),
        'cppn_std': float(np.std(cppn_nmi)),
        'random_mean': float(np.mean(random_nmi)),
        'random_std': float(np.std(random_nmi)),
        'mann_whitney_U': float(stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': bool(significant)
    }

    # Secondary: raw MI
    print("\n2. RAW CROSS-SCALE MI (bits):")
    print("-" * 50)

    cppn_mi = cppn_data['mi']
    random_mi = random_data['mi']

    print(f"   CPPN:   {np.mean(cppn_mi):.4f} +/- {np.std(cppn_mi):.4f}")
    print(f"   Random: {np.mean(random_mi):.4f} +/- {np.std(random_mi):.4f}")

    stat_mi, p_mi = mannwhitneyu(cppn_mi, random_mi, alternative='greater')
    d_mi = cohens_d(cppn_mi, random_mi)

    print(f"   p-value: {p_mi:.6e}, Cohen's d: {d_mi:.3f}")

    results['mi'] = {
        'cppn_mean': float(np.mean(cppn_mi)),
        'random_mean': float(np.mean(random_mi)),
        'p_value': float(p_mi),
        'cohens_d': float(d_mi)
    }

    # Component entropies
    print("\n3. COMPONENT ENTROPIES (bits):")
    print("-" * 50)

    print(f"   H(4x4 block):")
    print(f"     CPPN:   {np.mean(cppn_data['H_large']):.2f} +/- {np.std(cppn_data['H_large']):.2f}")
    print(f"     Random: {np.mean(random_data['H_large']):.2f} +/- {np.std(random_data['H_large']):.2f}")

    print(f"\n   H(2x2 block):")
    print(f"     CPPN:   {np.mean(cppn_data['H_small']):.2f} +/- {np.std(cppn_data['H_small']):.2f}")
    print(f"     Random: {np.mean(random_data['H_small']):.2f} +/- {np.std(random_data['H_small']):.2f}")

    results['H_large'] = {
        'cppn_mean': float(np.mean(cppn_data['H_large'])),
        'random_mean': float(np.mean(random_data['H_large']))
    }
    results['H_small'] = {
        'cppn_mean': float(np.mean(cppn_data['H_small'])),
        'random_mean': float(np.mean(random_data['H_small']))
    }

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results['nmi']['significant']:
        print("\nCPPN images show HIGHER cross-scale mutual information:")
        print("  - Small (2x2) patterns PREDICT large (4x4) patterns better")
        print("  - This indicates genuine HIERARCHICAL structure")
        print("  - Consistent with CPPN's compositional architecture")
        print("\nThis is NOVEL compared to existing entries:")
        print("  - RES-003 measured pairwise (adjacent pixel) MI")
        print("  - This measures CROSS-SCALE MI (2x2 to 4x4)")
        print("  - Captures hierarchical structure, not just local correlation")
    else:
        print("\nNo significant difference in cross-scale MI.")
        print("This could mean:")
        print("  - CPPN structure is primarily LOCAL (not hierarchical)")
        print("  - The 2x2 to 4x4 scale range doesn't capture the relevant structure")

    # Summary for log
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 70)

    if significant:
        status = 'validated'
    elif p_value < 0.05:
        status = 'inconclusive'
    else:
        status = 'refuted'

    print(f"\nStatus: {status.upper()}")
    print(f"P-value: {p_value:.6e}")
    print(f"Effect size (Cohen's d): {d:.3f}")
    print(f"CPPN NMI: {np.mean(cppn_nmi):.4f}")
    print(f"Random NMI: {np.mean(random_nmi):.4f}")

    results['summary'] = {
        'status': status,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'cppn_mean': float(np.mean(cppn_nmi)),
        'random_mean': float(np.mean(random_nmi)),
        'significant': bool(significant)
    }

    # Save results
    output_dir = Path("results/cross_scale_mi")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
