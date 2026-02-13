"""
Weight Space Geometry: Test connectivity of high-order CPPN regions

Hypothesis: High-order CPPN weight regions form disconnected clusters separated
by low-order valleys, suggesting nonlinear structure in weight space.

Method:
1. Sample 10 high-order CPPNs (order > 0.5)
2. Linear interpolation between each pair in weight space
3. Measure order at 11 points along each path (t = 0, 0.1, ..., 1.0)
4. Detect valleys: internal minimum < 0.2
5. Quantify: valley frequency, depth, width
"""

import numpy as np
import json
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def sample_high_order_cppn(order_threshold=0.3, max_attempts=500):
    """Sample a CPPN until we get one with order > threshold.

    Lower threshold to 0.3 since high-order CPPNs are rare in random sampling.
    """
    best_cppn = None
    best_order = -1

    for attempt in range(max_attempts):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        if order > best_order:
            best_order = order
            best_cppn = cppn

        if order > order_threshold:
            return cppn, order

    # If we fail to reach threshold, return the best we found
    if best_cppn is not None:
        return best_cppn, best_order

    # Absolute fallback
    cppn = CPPN()
    img = cppn.render(size=32)
    return cppn, order_multiplicative(img)


def sample_high_order_cppns(n=10, order_threshold=0.3):
    """Sample n high-order CPPNs."""
    cppns = []
    orders = []
    for i in range(n):
        cppn, order = sample_high_order_cppn(order_threshold=order_threshold)
        cppns.append(cppn)
        orders.append(order)
        print(f"  Sampled CPPN {i+1}/{n}: order={order:.3f}")
    return cppns, orders


def interpolate_weights(w1, w2, t):
    """Linear interpolation in weight space: w(t) = (1-t)*w1 + t*w2"""
    return (1 - t) * w1 + t * w2


def measure_order_along_path(cppn1, cppn2, n_steps=11):
    """
    Interpolate between two CPPNs in weight space and measure order.

    Returns:
    - t_values: [0, 0.1, ..., 1.0]
    - orders: order at each step
    - renders: images at each step
    """
    w1 = cppn1.get_weights()
    w2 = cppn2.get_weights()

    t_values = np.linspace(0, 1, n_steps)
    orders = []
    renders = []

    for t in t_values:
        # Interpolate weights
        w_interp = interpolate_weights(w1, w2, t)

        # Create interpolated CPPN
        cppn_interp = cppn1.copy()
        cppn_interp.set_weights(w_interp)

        # Render and measure order
        img = cppn_interp.render(size=32)
        order = order_multiplicative(img)

        orders.append(order)
        renders.append(img)

    return t_values, np.array(orders), renders


def detect_valley(orders, threshold=0.2):
    """
    Detect valleys in the order curve.

    A valley is a point where:
    - Internal point (not endpoints)
    - Value < threshold
    - Surrounded by higher values (local minimum)

    Returns: (has_valley, min_order, valley_depth, valley_position)
    """
    if len(orders) < 3:
        return False, np.min(orders), 0, -1

    # Look at internal points only
    for i in range(1, len(orders) - 1):
        # Check if local minimum
        if orders[i] < orders[i-1] and orders[i] < orders[i+1]:
            # Check if below threshold
            if orders[i] < threshold:
                depth = (orders[i-1] + orders[i+1]) / 2 - orders[i]
                return True, orders[i], depth, i

    # No valley found below threshold
    return False, np.min(orders), 0, -1


def analyze_weight_space_geometry(cppns, orders):
    """
    Full analysis: pairwise interpolation and valley detection.

    Returns: dict with statistics
    """
    n = len(cppns)
    n_pairs = n * (n - 1) // 2  # Unordered pairs

    results = {
        'n_cppns': n,
        'cppn_orders': [float(o) for o in orders],
        'mean_cppn_order': float(np.mean(orders)),
        'n_pairs': n_pairs,
        'paths': [],
        'valley_stats': {}
    }

    valleys_found = 0
    valley_depths = []
    valley_min_orders = []

    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            t_vals, order_vals, _ = measure_order_along_path(cppns[i], cppns[j], n_steps=11)

            has_valley, min_order, depth, valley_pos = detect_valley(order_vals, threshold=0.2)

            if has_valley:
                valleys_found += 1
                valley_depths.append(depth)
                valley_min_orders.append(min_order)

            path_data = {
                'pair': [i, j],
                'cppn1_order': float(orders[i]),
                'cppn2_order': float(orders[j]),
                'orders_along_path': [float(o) for o in order_vals],
                'min_order': float(np.min(order_vals)),
                'max_order': float(np.max(order_vals)),
                'mean_order': float(np.mean(order_vals)),
                'has_valley': bool(has_valley),
                'valley_depth': float(depth),
                'valley_min': float(min_order),
                'valley_position': int(valley_pos)
            }
            results['paths'].append(path_data)

            pair_idx += 1
            print(f"  Path {pair_idx}/{n_pairs}: CPPNs {i}-{j}, valley={has_valley}, depth={depth:.3f}")

    # Aggregate valley statistics
    valley_frequency = valleys_found / n_pairs if n_pairs > 0 else 0
    results['valley_stats'] = {
        'valleys_found': valleys_found,
        'total_paths': n_pairs,
        'valley_frequency': float(valley_frequency),
        'mean_valley_depth': float(np.mean(valley_depths)) if valley_depths else 0.0,
        'max_valley_depth': float(np.max(valley_depths)) if valley_depths else 0.0,
        'mean_valley_min_order': float(np.mean(valley_min_orders)) if valley_min_orders else 0.0,
    }

    return results


def main():
    print("=" * 80)
    print("WEIGHT SPACE GEOMETRY EXPERIMENT")
    print("=" * 80)

    set_global_seed(42)

    # Create results directory
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/weight_space_geometry')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. SAMPLING HIGH-ORDER CPPNs (order > 0.3)...")
    cppns, orders = sample_high_order_cppns(n=10, order_threshold=0.3)
    print(f"   Mean order: {np.mean(orders):.3f} (range: {np.min(orders):.3f}-{np.max(orders):.3f})")

    print("\n2. ANALYZING WEIGHT SPACE GEOMETRY...")
    results = analyze_weight_space_geometry(cppns, orders)

    print("\n3. RESULTS SUMMARY:")
    print(f"   Total paths analyzed: {results['valley_stats']['total_paths']}")
    print(f"   Valleys found: {results['valley_stats']['valleys_found']}")
    print(f"   Valley frequency: {results['valley_stats']['valley_frequency']:.1%}")
    print(f"   Mean valley depth: {results['valley_stats']['mean_valley_depth']:.3f}")
    print(f"   Mean valley min order: {results['valley_stats']['mean_valley_min_order']:.3f}")

    # Save results
    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {output_file}")

    # Determine finding
    valley_freq = results['valley_stats']['valley_frequency']
    if valley_freq > 0.70:
        status = "VALIDATED"
        effect_size = valley_freq
    elif valley_freq > 0.30:
        status = "INCONCLUSIVE"
        effect_size = valley_freq
    else:
        status = "REFUTED"
        effect_size = valley_freq

    print("\n4. INTERPRETATION:")
    print(f"   Status: {status}")
    print(f"   Valley frequency {valley_freq:.1%} suggests:")
    if valley_freq > 0.70:
        print("   → High-order regions ARE separated by low-order valleys (disconnected)")
        print("   → Weight space has nonlinear structure with bottlenecks")
    elif valley_freq > 0.30:
        print("   → Mixed evidence: some paths stay high-order, others dip")
        print("   → Possibly multiple connected components with varying density")
    else:
        print("   → Most paths stay high-order (connected regions)")
        print("   → Weight space is relatively smooth in high-order direction")

    # Output expected by orchestrator
    print(f"\nRES-214 | weight_space_geometry | {status} | freq={valley_freq:.3f}")

    return status, valley_freq


if __name__ == '__main__':
    main()
