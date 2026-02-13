"""
RES-153: Geodesic paths (order-constrained) between optima are shorter than Euclidean

Tests whether paths that maintain minimum order threshold are shorter than straight lines.
Related to RES-141 (barriers exist on linear paths) and RES-147 (minima form submanifolds).

If order-constrained geodesics are LONGER, it confirms weight space has complex topology
where maintaining order requires detours. If SHORTER, it suggests curved shortcuts exist.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy.stats import mannwhitneyu
import json
from pathlib import Path


def sample_high_order_cppn(target_order=0.3, max_attempts=5000):
    """Sample a CPPN with order above threshold."""
    for _ in range(max_attempts):
        cppn = CPPN()
        img = cppn.render(32)
        if order_multiplicative(img) >= target_order:
            return cppn, order_multiplicative(img)
    return None, 0.0


def euclidean_distance(w1, w2):
    """Standard Euclidean distance in weight space."""
    return np.linalg.norm(w1 - w2)


def path_length(waypoints):
    """Total path length through sequence of weight vectors."""
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += euclidean_distance(waypoints[i], waypoints[i + 1])
    return total


def order_along_path(cppn, w_start, w_end, n_points=20):
    """Compute order at evenly spaced points along linear path."""
    orders = []
    for t in np.linspace(0, 1, n_points):
        w = w_start + t * (w_end - w_start)
        cppn.set_weights(w)
        img = cppn.render(32)
        orders.append(order_multiplicative(img))
    return np.array(orders)


def find_order_constrained_path(cppn, w_start, w_end, min_order=0.1,
                                 max_waypoints=10, max_attempts=100):
    """
    Find a path from w_start to w_end that maintains order >= min_order.

    Uses greedy waypoint insertion: when linear path drops below min_order,
    sample random intermediate points that maintain order and are closer to goal.

    Returns: list of weight vectors forming valid path, or None if failed
    """
    waypoints = [w_start.copy(), w_end.copy()]

    for iteration in range(max_waypoints):
        # Check if current path is valid
        valid = True
        violation_idx = -1

        for i in range(len(waypoints) - 1):
            orders = order_along_path(cppn, waypoints[i], waypoints[i + 1], n_points=10)
            if np.min(orders) < min_order:
                valid = False
                violation_idx = i
                break

        if valid:
            return waypoints

        # Need to insert waypoint between violation_idx and violation_idx + 1
        w_a = waypoints[violation_idx]
        w_b = waypoints[violation_idx + 1]

        # Try to find valid intermediate point
        best_waypoint = None
        best_progress = -1  # Progress toward w_b

        for attempt in range(max_attempts):
            # Sample near midpoint with random perturbation
            midpoint = (w_a + w_b) / 2
            perturbation = np.random.randn(len(midpoint)) * 0.5
            candidate = midpoint + perturbation

            cppn.set_weights(candidate)
            img = cppn.render(32)
            order = order_multiplicative(img)

            if order >= min_order:
                # Check progress toward goal (how much closer to w_b from w_a)
                dist_to_b = euclidean_distance(candidate, w_b)
                dist_a_to_b = euclidean_distance(w_a, w_b)
                progress = 1 - dist_to_b / dist_a_to_b

                if progress > best_progress:
                    best_progress = progress
                    best_waypoint = candidate.copy()

        if best_waypoint is None:
            return None  # Failed to find valid waypoint

        # Insert the waypoint
        waypoints.insert(violation_idx + 1, best_waypoint)

    return None  # Exceeded max waypoints


def run_experiment(n_pairs=50, min_order=0.15, target_order=0.3, seed=42):
    """
    Main experiment: compare geodesic vs Euclidean path lengths.

    For each pair of high-order CPPNs:
    1. Compute Euclidean distance (straight line)
    2. Find order-constrained path via waypoint insertion
    3. Compute path length along constrained geodesic
    4. Record ratio: geodesic_length / euclidean_length
    """
    np.random.seed(seed)

    print(f"Sampling {n_pairs * 2} high-order CPPNs (target order >= {target_order})...")

    cppns = []
    for i in range(n_pairs * 2):
        cppn, order = sample_high_order_cppn(target_order=target_order)
        if cppn is not None:
            cppns.append((cppn, order))
        if (i + 1) % 20 == 0:
            print(f"  Sampled {len(cppns)} / {n_pairs * 2}")

    if len(cppns) < n_pairs * 2:
        print(f"Warning: Only found {len(cppns)} high-order CPPNs")

    # Form pairs
    pairs = [(cppns[i], cppns[i + 1]) for i in range(0, min(len(cppns) - 1, n_pairs * 2), 2)]

    results = {
        'euclidean_distances': [],
        'geodesic_lengths': [],
        'length_ratios': [],
        'n_waypoints': [],
        'path_found': [],
        'barrier_depths': [],  # Min order on straight line
    }

    print(f"\nTesting {len(pairs)} pairs...")

    for idx, ((cppn1, order1), (cppn2, order2)) in enumerate(pairs):
        w1 = cppn1.get_weights()
        w2 = cppn2.get_weights()

        # Use cppn1's architecture as template
        template = cppn1.copy()

        # Euclidean distance
        euc_dist = euclidean_distance(w1, w2)
        results['euclidean_distances'].append(euc_dist)

        # Check barrier on straight line
        orders_straight = order_along_path(template, w1, w2, n_points=30)
        barrier_depth = np.min(orders_straight)
        results['barrier_depths'].append(barrier_depth)

        # Find order-constrained path
        geodesic = find_order_constrained_path(template, w1, w2, min_order=min_order)

        if geodesic is not None:
            results['path_found'].append(True)
            results['n_waypoints'].append(len(geodesic))
            geo_length = path_length(geodesic)
            results['geodesic_lengths'].append(geo_length)
            results['length_ratios'].append(geo_length / euc_dist)
        else:
            results['path_found'].append(False)
            results['n_waypoints'].append(-1)
            results['geodesic_lengths'].append(np.nan)
            results['length_ratios'].append(np.nan)

        if (idx + 1) % 10 == 0:
            found = sum(results['path_found'])
            print(f"  Processed {idx + 1}/{len(pairs)}, paths found: {found}/{idx + 1}")

    return results


def analyze_results(results):
    """Statistical analysis of path length comparison."""
    ratios = [r for r in results['length_ratios'] if not np.isnan(r)]

    if len(ratios) < 10:
        return {'status': 'inconclusive', 'reason': 'too few valid paths'}

    ratios = np.array(ratios)

    # Test if ratio differs from 1.0 (geodesic = euclidean)
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    n = len(ratios)

    # One-sample t-test against ratio = 1.0
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(ratios, 1.0)

    # Cohen's d against null of ratio = 1.0
    cohens_d = (mean_ratio - 1.0) / std_ratio

    # Additional stats
    success_rate = sum(results['path_found']) / len(results['path_found'])
    mean_waypoints = np.mean([w for w in results['n_waypoints'] if w > 0])
    mean_barrier = np.mean(results['barrier_depths'])

    # Validate: paths are longer when barriers exist
    barriers = np.array(results['barrier_depths'])
    valid_mask = np.array(results['path_found'])

    analysis = {
        'n_pairs': len(results['path_found']),
        'n_valid_paths': len(ratios),
        'success_rate': success_rate,
        'mean_length_ratio': float(mean_ratio),
        'std_length_ratio': float(std_ratio),
        'median_length_ratio': float(np.median(ratios)),
        'min_ratio': float(np.min(ratios)),
        'max_ratio': float(np.max(ratios)),
        'mean_waypoints': float(mean_waypoints),
        'mean_barrier_depth': float(mean_barrier),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
    }

    # Determine outcome
    # Hypothesis: geodesic shorter than euclidean (ratio < 1.0)
    # Reality check: if barriers exist, geodesics should be LONGER (ratio > 1.0)

    if p_value < 0.01 and abs(cohens_d) > 0.5:
        if mean_ratio > 1.0:
            analysis['status'] = 'refuted'  # Geodesics are LONGER, not shorter
            analysis['interpretation'] = 'Geodesics are longer than Euclidean paths due to order barriers'
        else:
            analysis['status'] = 'validated'  # Geodesics are shorter (curved shortcuts exist)
            analysis['interpretation'] = 'Curved shortcuts exist that maintain order'
    else:
        analysis['status'] = 'inconclusive'
        analysis['interpretation'] = 'Effect size or significance below threshold'

    return analysis


if __name__ == '__main__':
    print("="*60)
    print("RES-153: Geodesic vs Euclidean Path Length Comparison")
    print("="*60)

    results = run_experiment(n_pairs=50, min_order=0.15, target_order=0.3)

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    analysis = analyze_results(results)

    print(f"\nSample size: {analysis['n_pairs']} pairs, {analysis['n_valid_paths']} valid paths")
    print(f"Path success rate: {analysis['success_rate']:.1%}")
    print(f"\nLength ratio (geodesic/euclidean):")
    print(f"  Mean:   {analysis['mean_length_ratio']:.3f} (+/- {analysis['std_length_ratio']:.3f})")
    print(f"  Median: {analysis['median_length_ratio']:.3f}")
    print(f"  Range:  [{analysis['min_ratio']:.3f}, {analysis['max_ratio']:.3f}]")
    print(f"\nMean waypoints per path: {analysis['mean_waypoints']:.1f}")
    print(f"Mean barrier depth (min order on straight line): {analysis['mean_barrier_depth']:.3f}")
    print(f"\nStatistical test (H0: ratio = 1.0):")
    print(f"  t = {analysis['t_statistic']:.3f}, p = {analysis['p_value']:.2e}")
    print(f"  Cohen's d = {analysis['cohens_d']:.3f}")
    print(f"\nStatus: {analysis['status'].upper()}")
    print(f"Interpretation: {analysis['interpretation']}")

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/geodesic_paths')
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': 'RES-153',
        'hypothesis': 'Geodesic paths (order-constrained) between optima are shorter than Euclidean',
        'results': {k: [float(x) if isinstance(x, (np.floating, np.integer)) else x
                       for x in v] if isinstance(v, list) else v
                   for k, v in results.items()},
        'analysis': analysis,
    }

    with open(output_dir / 'geodesic_path_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'geodesic_path_results.json'}")
