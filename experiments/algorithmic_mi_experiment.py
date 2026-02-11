#!/usr/bin/env python3
"""
EXPERIMENT RES-010: Algorithmic Mutual Information (Non-Local Structure)

HYPOTHESIS: CPPN images exhibit higher algorithmic mutual information between
distant spatial regions than random images, indicating NON-LOCAL structural
dependencies captured by the compositional CPPN architecture.

NULL HYPOTHESIS: Algorithmic MI between distant regions is the same for both.

THEORETICAL BACKGROUND:
Algorithmic mutual information between strings x and y is:
  I(x:y) = K(x) + K(y) - K(x,y)

This measures how much knowing x helps compress y (and vice versa).

For random images: I(x:y) ~ 0 (independent regions)
For locally-correlated images: I(adjacent) > 0, I(distant) ~ 0
For globally-structured images: I(distant) > 0 (CPPN should show this!)

The key insight: RES-003 measured LOCAL (adjacent pixel) mutual information.
This experiment measures LONG-RANGE algorithmic MI between DISTANT quadrants.

CPPN images should show higher long-range I because:
1. CPPNs are smooth functions over (x,y) coordinates
2. Distant regions share the same underlying function
3. This creates correlations even across large distances

METHODOLOGY:
1. Generate n=300 CPPN and n=300 random images (32x32)
2. Split each image into 4 quadrants (16x16 each)
3. Compute K(Q1), K(Q3), K(Q1,Q3) for DIAGONAL quadrants (maximally distant)
4. Algorithmic MI: I = K(Q1) + K(Q3) - K(Q1,Q3)
5. Normalize: NI = I / sqrt(K(Q1) * K(Q3))
6. Test: Is I higher for CPPN vs random for DIAGONAL quadrants?

CONTROLS:
- Also measure I for ADJACENT quadrants (should be higher for both)
- Compute ADJACENT/DIAGONAL ratio: measures "locality" of structure
- For CPPN: ratio should be LOWER (more non-local)

STATISTICAL TESTS:
- Mann-Whitney U test for diagonal I
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


def compress_length(data: np.ndarray) -> int:
    """Get compressed length in bits as K(x) proxy."""
    packed = np.packbits(data.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    return len(compressed) * 8


def compute_algorithmic_mi(img: np.ndarray) -> dict:
    """
    Compute algorithmic mutual information between image quadrants.

    Quadrant layout:
    Q0 | Q1
    -------
    Q2 | Q3

    Diagonal pairs: (Q0, Q3) and (Q1, Q2) - maximally distant
    Adjacent pairs: (Q0, Q1), (Q0, Q2), etc.

    Returns:
        dict with MI for diagonal and adjacent pairs
    """
    h, w = img.shape
    half_h, half_w = h // 2, w // 2

    # Extract quadrants
    Q0 = img[:half_h, :half_w]
    Q1 = img[:half_h, half_w:]
    Q2 = img[half_h:, :half_w]
    Q3 = img[half_h:, half_w:]

    # Compress individual quadrants
    K_Q0 = compress_length(Q0)
    K_Q1 = compress_length(Q1)
    K_Q2 = compress_length(Q2)
    K_Q3 = compress_length(Q3)

    # Compress pairs (concatenated)
    # Diagonal pairs
    K_Q0_Q3 = compress_length(np.concatenate([Q0.flatten(), Q3.flatten()]))
    K_Q1_Q2 = compress_length(np.concatenate([Q1.flatten(), Q2.flatten()]))

    # Adjacent pairs (horizontal)
    K_Q0_Q1 = compress_length(np.concatenate([Q0.flatten(), Q1.flatten()]))
    K_Q2_Q3 = compress_length(np.concatenate([Q2.flatten(), Q3.flatten()]))

    # Adjacent pairs (vertical)
    K_Q0_Q2 = compress_length(np.concatenate([Q0.flatten(), Q2.flatten()]))
    K_Q1_Q3 = compress_length(np.concatenate([Q1.flatten(), Q3.flatten()]))

    # Algorithmic MI: I(x:y) = K(x) + K(y) - K(x,y)
    # Diagonal
    I_diag_03 = K_Q0 + K_Q3 - K_Q0_Q3
    I_diag_12 = K_Q1 + K_Q2 - K_Q1_Q2
    I_diagonal = (I_diag_03 + I_diag_12) / 2

    # Adjacent horizontal
    I_adj_01 = K_Q0 + K_Q1 - K_Q0_Q1
    I_adj_23 = K_Q2 + K_Q3 - K_Q2_Q3
    I_adjacent_h = (I_adj_01 + I_adj_23) / 2

    # Adjacent vertical
    I_adj_02 = K_Q0 + K_Q2 - K_Q0_Q2
    I_adj_13 = K_Q1 + K_Q3 - K_Q1_Q3
    I_adjacent_v = (I_adj_02 + I_adj_13) / 2

    I_adjacent = (I_adjacent_h + I_adjacent_v) / 2

    # Normalize by geometric mean of individual complexities
    K_mean = (K_Q0 + K_Q1 + K_Q2 + K_Q3) / 4

    # Normalized MI (fraction of possible mutual information)
    NI_diagonal = I_diagonal / (K_mean + 1e-10)
    NI_adjacent = I_adjacent / (K_mean + 1e-10)

    # Locality ratio: how much MORE information is shared by adjacent vs diagonal
    # Lower ratio = more non-local structure
    if I_diagonal > 0:
        locality_ratio = I_adjacent / (I_diagonal + 1e-10)
    else:
        locality_ratio = float('inf') if I_adjacent > 0 else 1.0

    return {
        'I_diagonal': I_diagonal,
        'I_adjacent': I_adjacent,
        'NI_diagonal': NI_diagonal,
        'NI_adjacent': NI_adjacent,
        'locality_ratio': locality_ratio,
        'K_mean': K_mean,
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
    Main experiment: Compare algorithmic MI between CPPN and random.
    """
    print("=" * 70)
    print("EXPERIMENT RES-010: ALGORITHMIC MUTUAL INFORMATION (Non-Local)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPN images have higher algorithmic MI between")
    print("            DIAGONAL quadrants (long-range structure)")
    print("NULL: Algorithmic MI is the same for both image types")
    print()

    n_samples = 300
    image_size = 32

    # Collect data
    cppn_data = {
        'I_diagonal': [], 'I_adjacent': [],
        'NI_diagonal': [], 'NI_adjacent': [],
        'locality_ratio': [], 'K_mean': []
    }
    random_data = {
        'I_diagonal': [], 'I_adjacent': [],
        'NI_diagonal': [], 'NI_adjacent': [],
        'locality_ratio': [], 'K_mean': []
    }

    print(f"Generating {n_samples} CPPN images...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        metrics = compute_algorithmic_mi(img)
        for k, v in metrics.items():
            cppn_data[k].append(v)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    print(f"\nGenerating {n_samples} random images...")
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
        metrics = compute_algorithmic_mi(img)
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

    # Primary metric: normalized diagonal MI
    print("\n1. NORMALIZED DIAGONAL MI (primary metric):")
    print("-" * 50)
    print("   (mutual information between MAXIMALLY DISTANT quadrants)")

    cppn_ni_diag = cppn_data['NI_diagonal']
    random_ni_diag = random_data['NI_diagonal']

    print(f"\n   CPPN:   {np.mean(cppn_ni_diag):.4f} +/- {np.std(cppn_ni_diag):.4f}")
    print(f"   Random: {np.mean(random_ni_diag):.4f} +/- {np.std(random_ni_diag):.4f}")

    # Mann-Whitney U test (one-sided: CPPN > random)
    stat, p_value = mannwhitneyu(cppn_ni_diag, random_ni_diag, alternative='greater')
    d = cohens_d(cppn_ni_diag, random_ni_diag)

    print(f"\n   Mann-Whitney U: {stat:.1f}")
    print(f"   P-value (one-sided): {p_value:.6e}")
    print(f"   Cohen's d: {d:.3f}")

    significant = p_value < 0.01 and abs(d) > 0.5

    if significant:
        print("\n   ** SIGNIFICANT: CPPN has higher long-range algorithmic MI **")
    else:
        if p_value >= 0.01:
            print(f"\n   Not significant (p={p_value:.4f} >= 0.01)")
        if abs(d) <= 0.5:
            print(f"   Small effect size (|d|={abs(d):.3f} <= 0.5)")

    results['NI_diagonal'] = {
        'cppn_mean': float(np.mean(cppn_ni_diag)),
        'cppn_std': float(np.std(cppn_ni_diag)),
        'random_mean': float(np.mean(random_ni_diag)),
        'random_std': float(np.std(random_ni_diag)),
        'mann_whitney_U': float(stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': bool(significant)
    }

    # Adjacent MI for comparison
    print("\n2. NORMALIZED ADJACENT MI (control):")
    print("-" * 50)

    cppn_ni_adj = cppn_data['NI_adjacent']
    random_ni_adj = random_data['NI_adjacent']

    print(f"   CPPN:   {np.mean(cppn_ni_adj):.4f} +/- {np.std(cppn_ni_adj):.4f}")
    print(f"   Random: {np.mean(random_ni_adj):.4f} +/- {np.std(random_ni_adj):.4f}")

    stat_adj, p_adj = mannwhitneyu(cppn_ni_adj, random_ni_adj, alternative='greater')
    d_adj = cohens_d(cppn_ni_adj, random_ni_adj)
    print(f"   p-value: {p_adj:.6e}, Cohen's d: {d_adj:.3f}")

    results['NI_adjacent'] = {
        'cppn_mean': float(np.mean(cppn_ni_adj)),
        'random_mean': float(np.mean(random_ni_adj)),
        'p_value': float(p_adj),
        'cohens_d': float(d_adj)
    }

    # Locality ratio
    print("\n3. LOCALITY RATIO (adjacent/diagonal MI):")
    print("-" * 50)
    print("   (LOWER = more non-local structure)")

    cppn_locality = cppn_data['locality_ratio']
    random_locality = random_data['locality_ratio']

    # Filter out infinities for statistics
    cppn_loc_finite = cppn_locality[np.isfinite(cppn_locality)]
    random_loc_finite = random_locality[np.isfinite(random_locality)]

    print(f"\n   CPPN:   {np.mean(cppn_loc_finite):.2f} +/- {np.std(cppn_loc_finite):.2f}")
    print(f"   Random: {np.mean(random_loc_finite):.2f} +/- {np.std(random_loc_finite):.2f}")

    # Test: CPPN should have LOWER locality ratio (more non-local)
    stat_loc, p_loc = mannwhitneyu(random_loc_finite, cppn_loc_finite, alternative='greater')
    d_loc = cohens_d(random_loc_finite, cppn_loc_finite)  # Note: reversed for direction

    print(f"   p-value (random > CPPN): {p_loc:.6e}")
    print(f"   Cohen's d: {d_loc:.3f}")

    results['locality_ratio'] = {
        'cppn_mean': float(np.mean(cppn_loc_finite)),
        'random_mean': float(np.mean(random_loc_finite)),
        'p_value': float(p_loc),
        'cohens_d': float(d_loc)
    }

    # Raw MI values
    print("\n4. RAW ALGORITHMIC MI (bits):")
    print("-" * 50)

    print(f"   Diagonal:")
    print(f"     CPPN:   {np.mean(cppn_data['I_diagonal']):.1f} +/- {np.std(cppn_data['I_diagonal']):.1f}")
    print(f"     Random: {np.mean(random_data['I_diagonal']):.1f} +/- {np.std(random_data['I_diagonal']):.1f}")

    print(f"\n   Adjacent:")
    print(f"     CPPN:   {np.mean(cppn_data['I_adjacent']):.1f} +/- {np.std(cppn_data['I_adjacent']):.1f}")
    print(f"     Random: {np.mean(random_data['I_adjacent']):.1f} +/- {np.std(random_data['I_adjacent']):.1f}")

    results['I_diagonal_raw'] = {
        'cppn_mean': float(np.mean(cppn_data['I_diagonal'])),
        'random_mean': float(np.mean(random_data['I_diagonal']))
    }
    results['I_adjacent_raw'] = {
        'cppn_mean': float(np.mean(cppn_data['I_adjacent'])),
        'random_mean': float(np.mean(random_data['I_adjacent']))
    }

    # Mean complexity
    print(f"\n   Mean K(quadrant):")
    print(f"     CPPN:   {np.mean(cppn_data['K_mean']):.1f} bits")
    print(f"     Random: {np.mean(random_data['K_mean']):.1f} bits")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if results['NI_diagonal']['significant']:
        print("\nCPPN images show HIGHER algorithmic MI between distant regions:")
        print("  - Diagonal quadrants share more 'algorithmic information'")
        print("  - This indicates NON-LOCAL structure (not just local smoothness)")
        print("  - Consistent with CPPN's compositional function architecture")
        print("\nThis is NOVEL compared to existing entries:")
        print("  - RES-003: Measured ADJACENT pixel MI (local)")
        print("  - THIS: Measures DIAGONAL quadrant MI (non-local)")
        print("  - Captures different aspect of structure: global coherence")
    else:
        print("\nNo significant difference in long-range algorithmic MI.")
        print("This suggests CPPN structure may be primarily LOCAL.")

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
    print(f"CPPN normalized diagonal MI: {np.mean(cppn_ni_diag):.4f}")
    print(f"Random normalized diagonal MI: {np.mean(random_ni_diag):.4f}")

    results['summary'] = {
        'status': status,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'cppn_mean': float(np.mean(cppn_ni_diag)),
        'random_mean': float(np.mean(random_ni_diag)),
        'significant': bool(significant)
    }

    # Save results
    output_dir = Path("results/algorithmic_mi")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
