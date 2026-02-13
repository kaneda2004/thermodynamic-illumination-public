#!/usr/bin/env python3
"""
EXPERIMENT RES-053: Mutual Information Decomposition (PID)

HYPOTHESIS: High-order CPPN images exhibit synergy-dominated MI structure
compared to redundancy-dominated random images.

THEORETICAL BACKGROUND:
Partial Information Decomposition (PID) decomposes MI into:
- Unique(X1->Y): info only X1 provides about Y
- Unique(X2->Y): info only X2 provides about Y
- Redundancy(X1,X2->Y): info both X1 AND X2 provide (overlapping)
- Synergy(X1,X2->Y): info only X1 AND X2 together provide

For CPPN images with global compositional structure:
- Synergy should dominate: quadrants combine to reveal whole-image properties
- Compositional functions create emergent information at combination

For random images:
- Redundancy should dominate: information is local, repeated
- No new information emerges from combining regions

METHODOLOGY:
1. Generate CPPN and random images (32x32)
2. Split into quadrants: X1=top-half, X2=left-half, Y=whole image features
3. Compute PID using Williams-Beer lattice / Ibroja approach
4. Compare synergy/redundancy ratio between CPPN and random

STATISTICAL TESTS:
- Mann-Whitney U for synergy-redundancy ratio
- Cohen's d effect size
- p < 0.01, |d| > 0.5 for validation
"""

import sys
import os
import numpy as np
from scipy.stats import mannwhitneyu, entropy
import zlib
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN


def compress_length(data: np.ndarray) -> float:
    """Get compressed length in bits as complexity proxy."""
    packed = np.packbits(data.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    return len(compressed) * 8


def compute_entropy_proxy(data: np.ndarray) -> float:
    """Compute entropy of a binary region using compression."""
    # Raw bits
    raw_bits = data.size
    # Compressed bits
    compressed_bits = compress_length(data)
    # Normalize to [0, 1]
    return compressed_bits / raw_bits


def algorithmic_mi(x: np.ndarray, y: np.ndarray) -> float:
    """I(X:Y) = K(X) + K(Y) - K(X,Y)"""
    k_x = compress_length(x)
    k_y = compress_length(y)
    k_xy = compress_length(np.concatenate([x.flatten(), y.flatten()]))
    return max(0, k_x + k_y - k_xy)


def compute_pid_compression(img: np.ndarray) -> dict:
    """
    Compute PID-like decomposition using compression-based MI.

    We use three "views" of the image:
    - X1: Top half
    - X2: Left half
    - Y: Global features (compression, order metric)

    The decomposition follows Williams-Beer intuition:
    - I(X1:Y), I(X2:Y) = individual MI
    - I(X1,X2:Y) = joint MI (upper bound)
    - I(X1:Y|X2), I(X2:Y|X1) = conditional MI

    PID decomposition:
    - Redundancy = min(I(X1:Y), I(X2:Y))  [minimum specific info]
    - Unique(X1) = I(X1:Y) - Redundancy
    - Unique(X2) = I(X2:Y) - Redundancy
    - Synergy = I(X1,X2:Y) - I(X1:Y) - I(X2:Y) + Redundancy
    """
    h, w = img.shape
    half_h, half_w = h // 2, w // 2

    # Define regions
    X1 = img[:half_h, :]  # Top half
    X2 = img[:, :half_w]  # Left half
    X1_X2 = img  # Full image (represents joint)

    # Y = "target" = global compression (whole image structure)
    # We use different feature extractors

    # Feature 1: Compression-based complexity
    K_X1 = compress_length(X1)
    K_X2 = compress_length(X2)
    K_Y = compress_length(img)

    # Joint compressions
    K_X1_Y = compress_length(np.concatenate([X1.flatten(), img.flatten()]))
    K_X2_Y = compress_length(np.concatenate([X2.flatten(), img.flatten()]))
    K_X1_X2_Y = compress_length(np.concatenate([X1.flatten(), X2.flatten(), img.flatten()]))
    K_X1_X2 = compress_length(np.concatenate([X1.flatten(), X2.flatten()]))

    # Mutual informations (algorithmic)
    I_X1_Y = max(0, K_X1 + K_Y - K_X1_Y)
    I_X2_Y = max(0, K_X2 + K_Y - K_X2_Y)

    # Joint MI: I(X1,X2:Y) = K(X1,X2) + K(Y) - K(X1,X2,Y)
    I_X1X2_Y = max(0, K_X1_X2 + K_Y - K_X1_X2_Y)

    # PID decomposition (Williams-Beer minimum)
    redundancy = min(I_X1_Y, I_X2_Y)
    unique_X1 = I_X1_Y - redundancy
    unique_X2 = I_X2_Y - redundancy

    # Synergy: I(X1,X2:Y) - I(X1:Y) - I(X2:Y) + Redundancy
    # This is the extra info from COMBINING X1 and X2
    synergy = I_X1X2_Y - I_X1_Y - I_X2_Y + redundancy
    synergy = max(0, synergy)  # Can be negative due to compression noise

    # Total MI
    total_mi = redundancy + unique_X1 + unique_X2 + synergy

    # Ratios (normalized)
    eps = 1e-10
    synergy_ratio = synergy / (total_mi + eps)
    redundancy_ratio = redundancy / (total_mi + eps)
    synergy_over_redundancy = synergy / (redundancy + eps)

    return {
        'synergy': synergy,
        'redundancy': redundancy,
        'unique_X1': unique_X1,
        'unique_X2': unique_X2,
        'total_mi': total_mi,
        'synergy_ratio': synergy_ratio,
        'redundancy_ratio': redundancy_ratio,
        'synergy_over_redundancy': synergy_over_redundancy,
        'I_X1_Y': I_X1_Y,
        'I_X2_Y': I_X2_Y,
        'I_X1X2_Y': I_X1X2_Y,
    }


def compute_pid_quadrants(img: np.ndarray) -> dict:
    """
    Alternative PID using 4 quadrants to predict center region.

    X1 = Q0 (top-left), X2 = Q3 (bottom-right)
    Y = center 16x16 region

    This tests if DISTANT regions synergistically predict CENTER.
    """
    h, w = img.shape
    q = h // 4  # Quarter size

    # Quadrants
    Q0 = img[:h//2, :w//2]  # Top-left
    Q3 = img[h//2:, w//2:]  # Bottom-right (diagonal)

    # Center region
    center = img[q:3*q, q:3*q]  # Middle 16x16

    # Compressions
    K_Q0 = compress_length(Q0)
    K_Q3 = compress_length(Q3)
    K_center = compress_length(center)

    K_Q0_center = compress_length(np.concatenate([Q0.flatten(), center.flatten()]))
    K_Q3_center = compress_length(np.concatenate([Q3.flatten(), center.flatten()]))
    K_Q0_Q3 = compress_length(np.concatenate([Q0.flatten(), Q3.flatten()]))
    K_Q0_Q3_center = compress_length(np.concatenate([Q0.flatten(), Q3.flatten(), center.flatten()]))

    # MIs
    I_Q0_C = max(0, K_Q0 + K_center - K_Q0_center)
    I_Q3_C = max(0, K_Q3 + K_center - K_Q3_center)
    I_Q0Q3_C = max(0, K_Q0_Q3 + K_center - K_Q0_Q3_center)

    # PID
    redundancy = min(I_Q0_C, I_Q3_C)
    unique_Q0 = I_Q0_C - redundancy
    unique_Q3 = I_Q3_C - redundancy
    synergy = max(0, I_Q0Q3_C - I_Q0_C - I_Q3_C + redundancy)

    total_mi = redundancy + unique_Q0 + unique_Q3 + synergy
    eps = 1e-10

    return {
        'synergy': synergy,
        'redundancy': redundancy,
        'unique_Q0': unique_Q0,
        'unique_Q3': unique_Q3,
        'total_mi': total_mi,
        'synergy_ratio': synergy / (total_mi + eps),
        'redundancy_ratio': redundancy / (total_mi + eps),
        'synergy_over_redundancy': synergy / (redundancy + eps),
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
    Main experiment: Compare PID decomposition between CPPN and random.
    """
    print("=" * 70)
    print("EXPERIMENT RES-053: MI DECOMPOSITION (PID)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPN images exhibit synergy-dominated MI structure")
    print("            compared to redundancy-dominated random images")
    print()

    n_samples = 300
    image_size = 32
    np.random.seed(42)

    # Collect data
    cppn_pid = []
    random_pid = []
    cppn_quadrant_pid = []
    random_quadrant_pid = []

    print(f"Generating {n_samples} CPPN images...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        cppn_pid.append(compute_pid_compression(img))
        cppn_quadrant_pid.append(compute_pid_quadrants(img))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    print(f"\nGenerating {n_samples} random images...")
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size)).astype(np.uint8)
        random_pid.append(compute_pid_compression(img))
        random_quadrant_pid.append(compute_pid_quadrants(img))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} done")

    # Extract metrics
    def extract_metric(data_list, key):
        return np.array([d[key] for d in data_list])

    results = {}

    # ==========================================
    # Primary Analysis: Half-image PID
    # ==========================================
    print("\n" + "=" * 70)
    print("RESULTS: HALF-IMAGE PID DECOMPOSITION")
    print("=" * 70)

    # Synergy ratio
    cppn_synergy = extract_metric(cppn_pid, 'synergy_ratio')
    random_synergy = extract_metric(random_pid, 'synergy_ratio')

    print("\n1. SYNERGY RATIO (primary metric):")
    print("-" * 50)
    print(f"   CPPN:   {np.mean(cppn_synergy):.4f} +/- {np.std(cppn_synergy):.4f}")
    print(f"   Random: {np.mean(random_synergy):.4f} +/- {np.std(random_synergy):.4f}")

    stat, p_value = mannwhitneyu(cppn_synergy, random_synergy, alternative='greater')
    d = cohens_d(cppn_synergy, random_synergy)

    print(f"\n   Mann-Whitney U: {stat:.1f}")
    print(f"   P-value (CPPN > random): {p_value:.6e}")
    print(f"   Cohen's d: {d:.3f}")

    significant_synergy = p_value < 0.01 and abs(d) > 0.5

    results['synergy_ratio'] = {
        'cppn_mean': float(np.mean(cppn_synergy)),
        'cppn_std': float(np.std(cppn_synergy)),
        'random_mean': float(np.mean(random_synergy)),
        'random_std': float(np.std(random_synergy)),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': bool(significant_synergy)
    }

    # Redundancy ratio
    cppn_redundancy = extract_metric(cppn_pid, 'redundancy_ratio')
    random_redundancy = extract_metric(random_pid, 'redundancy_ratio')

    print("\n2. REDUNDANCY RATIO:")
    print("-" * 50)
    print(f"   CPPN:   {np.mean(cppn_redundancy):.4f} +/- {np.std(cppn_redundancy):.4f}")
    print(f"   Random: {np.mean(random_redundancy):.4f} +/- {np.std(random_redundancy):.4f}")

    stat_r, p_r = mannwhitneyu(random_redundancy, cppn_redundancy, alternative='greater')
    d_r = cohens_d(random_redundancy, cppn_redundancy)

    print(f"   P-value (random > CPPN): {p_r:.6e}")
    print(f"   Cohen's d: {d_r:.3f}")

    results['redundancy_ratio'] = {
        'cppn_mean': float(np.mean(cppn_redundancy)),
        'random_mean': float(np.mean(random_redundancy)),
        'p_value': float(p_r),
        'cohens_d': float(d_r)
    }

    # Synergy/Redundancy ratio
    cppn_syn_red = extract_metric(cppn_pid, 'synergy_over_redundancy')
    random_syn_red = extract_metric(random_pid, 'synergy_over_redundancy')

    # Filter infinities
    cppn_sr_finite = cppn_syn_red[np.isfinite(cppn_syn_red)]
    random_sr_finite = random_syn_red[np.isfinite(random_syn_red)]

    print("\n3. SYNERGY/REDUNDANCY RATIO:")
    print("-" * 50)
    print(f"   CPPN:   {np.mean(cppn_sr_finite):.4f} +/- {np.std(cppn_sr_finite):.4f}")
    print(f"   Random: {np.mean(random_sr_finite):.4f} +/- {np.std(random_sr_finite):.4f}")

    if len(cppn_sr_finite) > 10 and len(random_sr_finite) > 10:
        stat_sr, p_sr = mannwhitneyu(cppn_sr_finite, random_sr_finite, alternative='greater')
        d_sr = cohens_d(cppn_sr_finite, random_sr_finite)
        print(f"   P-value (CPPN > random): {p_sr:.6e}")
        print(f"   Cohen's d: {d_sr:.3f}")
        results['synergy_over_redundancy'] = {
            'cppn_mean': float(np.mean(cppn_sr_finite)),
            'random_mean': float(np.mean(random_sr_finite)),
            'p_value': float(p_sr),
            'cohens_d': float(d_sr)
        }

    # Raw values
    print("\n4. RAW PID COMPONENTS (bits):")
    print("-" * 50)

    for component in ['synergy', 'redundancy', 'unique_X1', 'unique_X2', 'total_mi']:
        cppn_vals = extract_metric(cppn_pid, component)
        random_vals = extract_metric(random_pid, component)
        print(f"   {component}:")
        print(f"     CPPN:   {np.mean(cppn_vals):.1f} +/- {np.std(cppn_vals):.1f}")
        print(f"     Random: {np.mean(random_vals):.1f} +/- {np.std(random_vals):.1f}")

    # ==========================================
    # Secondary Analysis: Quadrant PID (distant regions)
    # ==========================================
    print("\n" + "=" * 70)
    print("RESULTS: QUADRANT PID (DISTANT -> CENTER)")
    print("=" * 70)

    cppn_q_syn = extract_metric(cppn_quadrant_pid, 'synergy_ratio')
    random_q_syn = extract_metric(random_quadrant_pid, 'synergy_ratio')

    print("\n1. QUADRANT SYNERGY RATIO:")
    print("-" * 50)
    print(f"   CPPN:   {np.mean(cppn_q_syn):.4f} +/- {np.std(cppn_q_syn):.4f}")
    print(f"   Random: {np.mean(random_q_syn):.4f} +/- {np.std(random_q_syn):.4f}")

    stat_q, p_q = mannwhitneyu(cppn_q_syn, random_q_syn, alternative='greater')
    d_q = cohens_d(cppn_q_syn, random_q_syn)

    print(f"   P-value (CPPN > random): {p_q:.6e}")
    print(f"   Cohen's d: {d_q:.3f}")

    significant_quadrant = p_q < 0.01 and abs(d_q) > 0.5

    results['quadrant_synergy'] = {
        'cppn_mean': float(np.mean(cppn_q_syn)),
        'random_mean': float(np.mean(random_q_syn)),
        'p_value': float(p_q),
        'cohens_d': float(d_q),
        'significant': bool(significant_quadrant)
    }

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Determine overall status
    if significant_synergy or significant_quadrant:
        status = 'validated'
        summary = (f"CPPN images show significantly higher synergy ratio "
                  f"(d={d:.2f}, p={p_value:.2e}) supporting synergy-dominated structure")
    elif p_value < 0.05 or p_q < 0.05:
        status = 'inconclusive'
        summary = f"Weak evidence for synergy dominance (p={min(p_value, p_q):.3f})"
    else:
        status = 'refuted'
        summary = "No significant difference in synergy/redundancy structure"

    print(f"\nStatus: {status.upper()}")
    print(f"Primary p-value: {p_value:.6e}")
    print(f"Primary effect size (Cohen's d): {d:.3f}")
    print(f"Summary: {summary}")

    results['summary'] = {
        'status': status,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'summary': summary
    }

    # Save results
    output_dir = Path("results/mi_decomposition")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
