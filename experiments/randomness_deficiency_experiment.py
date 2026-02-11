#!/usr/bin/env python3
"""
EXPERIMENT RES-010: Randomness Deficiency as Kolmogorov Complexity Measure

HYPOTHESIS: CPPN images have higher randomness deficiency than truly random
images, indicating they are "far from random" in the Kolmogorov sense.

NULL HYPOTHESIS: Randomness deficiency is the same for CPPN and random images.

THEORETICAL BACKGROUND:
Randomness deficiency d(x|P) measures how far a string x is from being a
"typical" sample of distribution P. For a uniform prior on binary images:

  d(x) = n - K(x)

where n = image size in bits (expected code length for random)
and K(x) = Kolmogorov complexity (shortest description length)

Since K(x) is uncomputable, we approximate with:
  K_approx(x) = len(zlib.compress(x))

This gives us:
  d_approx(x) = n - compress_len(x)

For RANDOM images: K ~ n, so d ~ 0 (no deficiency)
For STRUCTURED images: K << n, so d >> 0 (high deficiency)

This is a DIRECT Kolmogorov complexity comparison that doesn't suffer from
the normalization issues of mutual information.

NOVELTY vs EXISTING ENTRIES:
- RES-003 measured spatial MI (pairwise correlation)
- RES-007 measured compressibility correlation with order
- THIS measures ABSOLUTE randomness deficiency as a Kolmogorov proxy

METHODOLOGY:
1. Generate n=300 CPPN and n=300 random images (32x32)
2. Compute raw bit count: n = 32*32 = 1024 bits
3. Compute compressed size using zlib
4. Randomness deficiency = n - 8*len(compressed)
5. Test: Is deficiency higher for CPPN vs random?

STATISTICAL TESTS:
- Mann-Whitney U test (non-parametric)
- Effect size: Cohen's d
- Success criteria: p < 0.01, |d| > 0.5
"""

import sys
import os
import numpy as np
from scipy.stats import mannwhitneyu
import zlib
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN


def compute_randomness_deficiency(img: np.ndarray) -> dict:
    """
    Compute randomness deficiency as a Kolmogorov complexity proxy.

    Args:
        img: Binary image (H x W)

    Returns:
        dict with deficiency metrics
    """
    h, w = img.shape
    n_bits = h * w  # Maximum entropy for binary image

    # Pack bits and compress
    packed = np.packbits(img.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    compressed_bits = len(compressed) * 8

    # Randomness deficiency: how many bits "saved" vs random
    deficiency = n_bits - compressed_bits

    # Normalized deficiency (fraction of bits saved)
    normalized_deficiency = deficiency / n_bits

    # Also compute the compression ratio (for comparison)
    compression_ratio = compressed_bits / n_bits

    return {
        'deficiency': deficiency,
        'normalized_deficiency': normalized_deficiency,
        'compressed_bits': compressed_bits,
        'raw_bits': n_bits,
        'compression_ratio': compression_ratio
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
    Main experiment: Compare randomness deficiency between CPPN and random.
    """
    print("=" * 70)
    print("EXPERIMENT RES-010: RANDOMNESS DEFICIENCY (Kolmogorov Proxy)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPN images have higher randomness deficiency than random")
    print("            (they are 'far from random' in the Kolmogorov sense)")
    print("NULL: Randomness deficiency is the same for both image types")
    print()

    n_samples = 300
    image_size = 32

    # Collect data
    cppn_data = {'deficiency': [], 'normalized': [], 'compressed': []}
    random_data = {'deficiency': [], 'normalized': [], 'compressed': []}

    print(f"Generating {n_samples} CPPN images...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        metrics = compute_randomness_deficiency(img)
        cppn_data['deficiency'].append(metrics['deficiency'])
        cppn_data['normalized'].append(metrics['normalized_deficiency'])
        cppn_data['compressed'].append(metrics['compressed_bits'])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    print(f"\nGenerating {n_samples} random images...")
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
        metrics = compute_randomness_deficiency(img)
        random_data['deficiency'].append(metrics['deficiency'])
        random_data['normalized'].append(metrics['normalized_deficiency'])
        random_data['compressed'].append(metrics['compressed_bits'])
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
    n_bits = image_size * image_size

    # Primary metric: normalized randomness deficiency
    print("\n1. NORMALIZED RANDOMNESS DEFICIENCY (primary metric):")
    print("-" * 50)
    print(f"   (= fraction of bits 'saved' vs incompressible random)")

    cppn_norm = cppn_data['normalized']
    random_norm = random_data['normalized']

    print(f"\n   CPPN:   {np.mean(cppn_norm):.4f} +/- {np.std(cppn_norm):.4f}")
    print(f"   Random: {np.mean(random_norm):.4f} +/- {np.std(random_norm):.4f}")

    # Mann-Whitney U test (one-sided: CPPN > random)
    stat, p_value = mannwhitneyu(cppn_norm, random_norm, alternative='greater')
    d = cohens_d(cppn_norm, random_norm)

    print(f"\n   Mann-Whitney U: {stat:.1f}")
    print(f"   P-value (one-sided): {p_value:.6e}")
    print(f"   Cohen's d: {d:.3f}")

    significant = p_value < 0.01 and abs(d) > 0.5

    if significant:
        print("\n   ** SIGNIFICANT: CPPN has higher randomness deficiency **")
    else:
        if p_value >= 0.01:
            print(f"\n   Not significant (p={p_value:.4f} >= 0.01)")
        if abs(d) <= 0.5:
            print(f"   Small effect size (|d|={abs(d):.3f} <= 0.5)")

    results['normalized_deficiency'] = {
        'cppn_mean': float(np.mean(cppn_norm)),
        'cppn_std': float(np.std(cppn_norm)),
        'random_mean': float(np.mean(random_norm)),
        'random_std': float(np.std(random_norm)),
        'mann_whitney_U': float(stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': bool(significant)
    }

    # Raw deficiency
    print("\n2. RAW RANDOMNESS DEFICIENCY (bits):")
    print("-" * 50)

    cppn_def = cppn_data['deficiency']
    random_def = random_data['deficiency']

    print(f"   CPPN:   {np.mean(cppn_def):.1f} +/- {np.std(cppn_def):.1f} bits")
    print(f"   Random: {np.mean(random_def):.1f} +/- {np.std(random_def):.1f} bits")
    print(f"   (out of {n_bits} total bits)")

    stat_def, p_def = mannwhitneyu(cppn_def, random_def, alternative='greater')
    d_def = cohens_d(cppn_def, random_def)

    print(f"   p-value: {p_def:.6e}, Cohen's d: {d_def:.3f}")

    results['raw_deficiency'] = {
        'cppn_mean': float(np.mean(cppn_def)),
        'random_mean': float(np.mean(random_def)),
        'p_value': float(p_def),
        'cohens_d': float(d_def)
    }

    # Compressed sizes
    print("\n3. COMPRESSED SIZES (bits):")
    print("-" * 50)

    print(f"   CPPN:   {np.mean(cppn_data['compressed']):.1f} +/- {np.std(cppn_data['compressed']):.1f}")
    print(f"   Random: {np.mean(random_data['compressed']):.1f} +/- {np.std(random_data['compressed']):.1f}")
    print(f"   (raw size: {n_bits} bits)")

    results['compressed_bits'] = {
        'cppn_mean': float(np.mean(cppn_data['compressed'])),
        'random_mean': float(np.mean(random_data['compressed']))
    }

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results['normalized_deficiency']['significant']:
        improvement = np.mean(cppn_norm) - np.mean(random_norm)
        print(f"\nCPPN images have {improvement*100:.1f}% HIGHER randomness deficiency:")
        print("  - CPPN images are 'far from random' in the Kolmogorov sense")
        print("  - They have measurably lower algorithmic complexity")
        print("  - This validates the CPPN prior as 'structured' vs 'random'")
        print("\nThis finding is NOVEL compared to existing entries:")
        print("  - RES-003: Measured spatial MI (local correlation)")
        print("  - RES-007: Correlated compressibility with order metric")
        print("  - THIS: Directly measures Kolmogorov complexity proxy")
        print("    via absolute randomness deficiency")
    else:
        print("\nNo significant difference in randomness deficiency.")
        print("This would be surprising - zlib should compress CPPN better.")

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
    print(f"CPPN normalized deficiency: {np.mean(cppn_norm):.4f}")
    print(f"Random normalized deficiency: {np.mean(random_norm):.4f}")

    results['summary'] = {
        'status': status,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'cppn_mean': float(np.mean(cppn_norm)),
        'random_mean': float(np.mean(random_norm)),
        'significant': bool(significant)
    }

    # Save results
    output_dir = Path("results/randomness_deficiency")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
