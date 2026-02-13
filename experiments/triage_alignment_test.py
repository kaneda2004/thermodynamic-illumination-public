#!/usr/bin/env python3
"""
TRIAGE-3: Alignment Principle Critical Test

Question: Does thermodynamic alignment hold at 64×64 with normalized metric?

Simplified approach:
- Generate 5 CPPN samples with varying orders
- Measure their order using normalized metric (v2)
- For each, measure how well it can be reconstructed (as proxy for DIP)
- Test: Does delta = |order - baseline_order| predict reconstruction difficulty?

GO criteria: r(reconstruction_quality, -|delta|) < -0.6 (negative correlation)
NO-GO criteria: r > -0.4 (alignment principle fails)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative_v2, set_global_seed
)


def generate_diverse_cppns(n: int = 5) -> list:
    """Generate diverse CPPNs by sampling from prior."""
    cpps = []
    set_global_seed(42)

    attempts = 0
    max_attempts = n * 3

    while len(cpps) < n and attempts < max_attempts:
        cppn = CPPN()
        cpps.append({'cppn': cppn, 'seed': len(cpps)})
        attempts += 1

    return cpps


def render_cppn_at_resolution(cppn: CPPN, resolution: int) -> np.ndarray:
    """Render CPPN at 64x64 resolution, return grayscale image."""
    coords = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(coords, coords)
    img = cppn.activate(x, y)
    return (img > 0.5).astype(np.uint8)


def compute_reconstruction_quality(img: np.ndarray) -> float:
    """
    Proxy for DIP reconstruction quality.

    For the alignment principle to hold:
    - High-order (structured) images should be easy to reconstruct
    - Low-order (random) images should be hard to reconstruct

    We use a simple proxy: How well can we predict this image from
    its own structure? We measure this by:
    - Smooth images with long-range correlations: high quality (>0.8)
    - Structured but detailed: medium (0.5-0.8)
    - Random/noisy: low (<0.5)

    This is computed as the correlation between the image and a
    smoothed version of itself.
    """
    from scipy.ndimage import gaussian_filter

    # Smooth the image
    smoothed = gaussian_filter(img.astype(float), sigma=2.0)

    # Correlation between original and smoothed = predictability
    img_flat = img.flatten()
    smooth_flat = smoothed.flatten()

    # Normalize
    img_std = np.std(img_flat)
    smooth_std = np.std(smooth_flat)

    if img_std < 1e-6 or smooth_std < 1e-6:
        return 0.5  # Constant image: medium quality

    correlation = np.corrcoef(img_flat, smooth_flat)[0, 1]
    # Map from [-1, 1] to [0, 1]
    quality = (correlation + 1) / 2

    return float(quality)


def main():
    print("=" * 70)
    print("TRIAGE-3: ALIGNMENT PRINCIPLE CRITICAL TEST")
    print("=" * 70)
    print("\nQuestion: Does order alignment predict reconstruction quality?")
    print("At 64×64 with normalized metric v2")
    print()

    resolution = 64

    # Step 1: Generate diverse CPPNs
    print(f"Generating {5} diverse CPPNs...")
    cpps = generate_diverse_cppns(n=5)

    if len(cpps) < 3:
        print(f"ERROR: Only generated {len(cpps)}/5 CPPNs")
        return {"error": "insufficient_cppns"}

    print(f"Generated {len(cpps)} CPPNs\n")

    # Step 2: Measure baseline order (natural image order)
    # For this simplified test, we use the median CPPN order as baseline
    print("Measuring CPPN orders at 64×64...")
    results = []

    for i, cppn_data in enumerate(cpps):
        cppn = cppn_data['cppn']

        # Render at 64×64
        img = render_cppn_at_resolution(cppn, resolution)

        # Measure order with normalized metric
        order = order_multiplicative_v2(img, resolution_ref=32)

        # Measure reconstruction quality
        quality = compute_reconstruction_quality(img)

        results.append({
            'cppn_id': i,
            'order': order,
            'reconstruction_quality': quality,
        })

        print(f"  CPPN {i}: order={order:.4f}, quality={quality:.4f}")

    # Step 3: Compute baseline order (median)
    orders = np.array([r['order'] for r in results])
    baseline_order = np.median(orders)

    print(f"\nBaseline order (median): {baseline_order:.4f}")

    # Step 4: Compute deltas and analyze correlation
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    deltas = np.array([abs(r['order'] - baseline_order) for r in results])
    qualities = np.array([r['reconstruction_quality'] for r in results])

    # The alignment principle predicts:
    # Low delta (close to baseline) → HIGH reconstruction quality
    # High delta (far from baseline) → LOW reconstruction quality
    # So we expect: r(quality, -delta) < 0 (negative correlation)

    # Compute correlations
    pearson_r, pearson_p = pearsonr(-deltas, qualities)
    spearman_r, spearman_p = spearmanr(-deltas, qualities)

    print(f"\nReconstruction Quality vs -|delta|:")
    print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
    print(f"  Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.6f}")

    print(f"\nDetailed results:")
    print(f"{'CPPN':<6} {'Order':<8} {'Delta':<8} {'Quality':<10}")
    print("-" * 32)
    for i, r in enumerate(results):
        delta = abs(r['order'] - baseline_order)
        print(f"{i:<6} {r['order']:<8.4f} {delta:<8.4f} {r['reconstruction_quality']:<10.4f}")

    # GO/NO-GO Decision
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)

    output_data = {
        'metadata': {
            'n_cppns': len(cpps),
            'resolution': resolution,
            'baseline_order': float(baseline_order),
        },
        'measurements': results,
        'analysis': {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
        }
    }

    if pearson_r < -0.6:
        status = "GO"
        recommendation = "Alignment principle validated (r < -0.6). Paper claims supported."
    elif pearson_r < -0.4:
        status = "CAUTION"
        recommendation = "Moderate alignment (r ∈ [-0.6, -0.4]). Caveat needed."
    else:
        status = "NO-GO"
        recommendation = "Alignment principle fails (r > -0.4). Remove from paper."

    output_data['status'] = status
    output_data['recommendation'] = recommendation

    print(f"\nSTATUS: {status}")
    print(f"RECOMMENDATION: {recommendation}")

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/triage')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'alignment_quick_test.json'

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


if __name__ == '__main__':
    main()
