#!/usr/bin/env python3
"""
TRIAGE-2: Metric Normalization Quick Test

Question: Does adaptive edge_gate fix RES-069's metric divergence?

Method:
- Test 10 high-order CPPNs at 32 and 64
- Compute order with both v1 (original) and v2 (normalized) metrics
- Calculate ρ(order_32, order_64) for each metric

GO criteria: ρ_new(32,64) > 0.8
NO-GO criteria: ρ_new(32,64) < 0.7
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr
import json
from pathlib import Path

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, order_multiplicative_v2, set_global_seed
)


def sample_high_order_cppns(n_samples: int = 10) -> list:
    """
    Generate diverse CPPNs by sampling from prior.
    Keep ones that achieve reasonable order (>0.15).
    """
    cpps = []
    attempts = 0
    max_attempts = n_samples * 3

    set_global_seed(42)

    while len(cpps) < n_samples and attempts < max_attempts:
        # Create a random CPPN
        cppn = CPPN()

        # Render at 32x32 to check order
        coords = np.linspace(-1, 1, 32)
        x, y = np.meshgrid(coords, coords)
        img = cppn.activate(x, y)
        img_binary = (img > 0.5).astype(np.uint8)

        # Measure order
        order = order_multiplicative(img_binary)

        # Keep if order is reasonable
        if order > 0.15:
            cpps.append(cppn)

        attempts += 1

    if len(cpps) < n_samples:
        print(f"Warning: Only sampled {len(cpps)}/{n_samples} high-order CPPNs")

    return cpps


def render_and_measure(cppn: CPPN, resolution: int) -> dict:
    """Render CPPN at given resolution, measure order with both metrics."""
    # Create coordinate grid
    coords = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(coords, coords)

    # Render CPPN
    img = cppn.activate(x, y)
    img_binary = (img > 0.5).astype(np.uint8)

    # Measure order with both metrics
    order_v1 = order_multiplicative(img_binary)
    order_v2 = order_multiplicative_v2(img_binary, resolution_ref=32)

    return {
        'order_v1': order_v1,
        'order_v2': order_v2,
        'image_mean': np.mean(img_binary),
    }


def main():
    print("=" * 70)
    print("TRIAGE-2: METRIC NORMALIZATION QUICK TEST")
    print("=" * 70)

    set_global_seed(42)
    resolutions = [32, 64]
    n_cppns = 10

    print(f"\nSampling {n_cppns} high-order CPPNs...")
    cppns = sample_high_order_cppns(n_samples=n_cppns)

    if not cppns:
        print("ERROR: Could not sample any high-order CPPNs")
        return {"error": "sampling_failed"}

    print(f"Successfully sampled {len(cppns)} CPPNs\n")

    # Store results for each CPPN
    results = {
        'metadata': {
            'n_cppns': len(cppns),
            'resolutions': resolutions,
            'reference_resolution': 32,
        },
        'measurements': []
    }

    # Measure each CPPN at both resolutions
    print(f"{'CPPN':<6} {'Res':<4} {'Order-V1':<10} {'Order-V2':<10}")
    print("-" * 30)

    for i, cppn in enumerate(cppns):
        cppn_data = {'cppn_id': i}

        for res in resolutions:
            measurement = render_and_measure(cppn, res)
            key = f'res_{res}'
            cppn_data[key] = measurement
            print(f"{i:<6} {res:<4} {measurement['order_v1']:<10.4f} {measurement['order_v2']:<10.4f}")

        results['measurements'].append(cppn_data)

    # Compute correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Extract order values at each resolution
    order_v1_32 = np.array([m['res_32']['order_v1'] for m in results['measurements']])
    order_v1_64 = np.array([m['res_64']['order_v1'] for m in results['measurements']])
    order_v2_32 = np.array([m['res_32']['order_v2'] for m in results['measurements']])
    order_v2_64 = np.array([m['res_64']['order_v2'] for m in results['measurements']])

    # Compute Spearman correlations (more robust for small n)
    rho_v1, p_v1 = spearmanr(order_v1_32, order_v1_64)
    rho_v2, p_v2 = spearmanr(order_v2_32, order_v2_64)

    print(f"\nOriginal Metric (v1):")
    print(f"  ρ(order_32, order_64) = {rho_v1:.4f}, p = {p_v1:.6f}")
    print(f"  Mean order @ 32: {np.mean(order_v1_32):.4f}")
    print(f"  Mean order @ 64: {np.mean(order_v1_64):.4f}")
    print(f"  Delta: {np.mean(order_v1_64) - np.mean(order_v1_32):.4f}")

    print(f"\nNormalized Metric (v2):")
    print(f"  ρ(order_32, order_64) = {rho_v2:.4f}, p = {p_v2:.6f}")
    print(f"  Mean order @ 32: {np.mean(order_v2_32):.4f}")
    print(f"  Mean order @ 64: {np.mean(order_v2_64):.4f}")
    print(f"  Delta: {np.mean(order_v2_64) - np.mean(order_v2_32):.4f}")

    # GO/NO-GO criteria
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)

    results['analysis'] = {
        'correlation_v1': {
            'rho': float(rho_v1),
            'p_value': float(p_v1),
            'mean_32': float(np.mean(order_v1_32)),
            'mean_64': float(np.mean(order_v1_64)),
        },
        'correlation_v2': {
            'rho': float(rho_v2),
            'p_value': float(p_v2),
            'mean_32': float(np.mean(order_v2_32)),
            'mean_64': float(np.mean(order_v2_64)),
        },
    }

    if rho_v2 > 0.8:
        status = "GO"
        recommendation = "Normalized metric validated (ρ > 0.8). Proceed to TRIAGE-3."
    elif rho_v2 > 0.7:
        status = "CAUTION"
        recommendation = "Moderate improvement (ρ > 0.7). May proceed with caveat."
    else:
        status = "NO-GO"
        recommendation = "Normalization insufficient (ρ < 0.7). Must fix metric further."

    results['status'] = status
    results['recommendation'] = recommendation

    print(f"\nSTATUS: {status}")
    print(f"RECOMMENDATION: {recommendation}")

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/triage')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'metric_normalization_quick.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
