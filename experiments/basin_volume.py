"""
RES-172: Basin Volume Experiment

Hypothesis: High-order basins occupy smaller volume in weight space than low-order regions.

Method:
1. Sample many random CPPNs from the prior
2. Compute order for each
3. Estimate volume fraction at different order thresholds
4. For CPPNs at different order levels, estimate local basin volume via ball-sampling

The key insight is that if high-order basins are rare (small volume), then uniform
sampling from the prior should produce few high-order samples.
"""

import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def estimate_volume_fraction(n_samples: int, order_threshold: float, seed: int = 42) -> dict:
    """Estimate what fraction of weight space lies above order threshold."""
    set_global_seed(seed)

    orders = []
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        orders.append(order)

    orders = np.array(orders)
    fraction_above = np.mean(orders >= order_threshold)

    return {
        'n_samples': n_samples,
        'threshold': order_threshold,
        'fraction_above': float(fraction_above),
        'mean_order': float(np.mean(orders)),
        'std_order': float(np.std(orders)),
        'max_order': float(np.max(orders)),
        'percentiles': {
            '10': float(np.percentile(orders, 10)),
            '25': float(np.percentile(orders, 25)),
            '50': float(np.percentile(orders, 50)),
            '75': float(np.percentile(orders, 75)),
            '90': float(np.percentile(orders, 90)),
            '95': float(np.percentile(orders, 95)),
            '99': float(np.percentile(orders, 99)),
        }
    }


def estimate_local_basin_volume(cppn: CPPN, n_samples: int, radius: float, order_tolerance: float = 0.1) -> float:
    """
    Estimate local basin volume by sampling within a ball and counting
    how many samples stay within order_tolerance of the center.

    Returns the fraction of samples that stay in basin (proxy for volume).
    """
    center_img = cppn.render(32)
    center_order = order_multiplicative(center_img)
    center_weights = cppn.get_weights()
    dim = len(center_weights)

    in_basin = 0
    for _ in range(n_samples):
        # Sample uniformly from ball of given radius
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)
        r = radius * np.random.uniform(0, 1) ** (1/dim)  # Uniform in ball
        perturbed_weights = center_weights + r * direction

        test_cppn = cppn.copy()
        test_cppn.set_weights(perturbed_weights)
        test_img = test_cppn.render(32)
        test_order = order_multiplicative(test_img)

        if abs(test_order - center_order) <= order_tolerance:
            in_basin += 1

    return in_basin / n_samples


def compare_basin_volumes_at_order_levels(n_centers: int, n_samples_per_center: int,
                                          radius: float, seed: int = 42) -> dict:
    """
    Compare local basin volumes for CPPNs at different order levels.

    Strategy: Generate many CPPNs, bin by order level, estimate local basin volume for each.
    """
    set_global_seed(seed)

    # First, generate a pool of CPPNs
    pool_size = n_centers * 20  # Over-generate to get diverse orders
    print(f"Generating {pool_size} CPPNs...")
    cppns_and_orders = []
    for i in range(pool_size):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        cppns_and_orders.append((cppn, order))
        if (i+1) % 1000 == 0:
            print(f"  Generated {i+1}/{pool_size}")

    orders = np.array([o for _, o in cppns_and_orders])

    # Define order bins
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.30]  # Low to high order
    bins = []
    for i, t in enumerate(thresholds):
        if i == len(thresholds) - 1:
            upper = 1.0
        else:
            upper = thresholds[i+1]
        mask = (orders >= t) & (orders < upper)
        bin_cppns = [(c, o) for (c, o), m in zip(cppns_and_orders, mask) if m]
        bins.append({
            'lower': t,
            'upper': upper,
            'count': len(bin_cppns),
            'cppns': bin_cppns[:n_centers]  # Take up to n_centers from each bin
        })

    # Estimate local basin volume for each bin
    results = []
    for bin_data in bins:
        print(f"\nProcessing order bin [{bin_data['lower']}, {bin_data['upper']}), n={len(bin_data['cppns'])}")
        if len(bin_data['cppns']) < 5:
            print("  Skipping - too few samples")
            continue

        volumes = []
        orders_in_bin = []
        for j, (cppn, order) in enumerate(bin_data['cppns']):
            vol = estimate_local_basin_volume(cppn, n_samples_per_center, radius)
            volumes.append(vol)
            orders_in_bin.append(order)
            if (j+1) % 10 == 0:
                print(f"  Processed {j+1}/{len(bin_data['cppns'])}")

        results.append({
            'order_range': f"[{bin_data['lower']}, {bin_data['upper']})",
            'lower': bin_data['lower'],
            'upper': bin_data['upper'],
            'n_centers': len(bin_data['cppns']),
            'mean_volume': float(np.mean(volumes)),
            'std_volume': float(np.std(volumes)),
            'mean_order': float(np.mean(orders_in_bin)),
            'volumes': volumes
        })

    return {
        'bins': results,
        'radius': radius,
        'n_samples_per_center': n_samples_per_center
    }


def main():
    print("=" * 60)
    print("RES-172: Basin Volume Experiment")
    print("=" * 60)

    # Part 1: Global volume fraction estimates
    print("\n" + "=" * 40)
    print("Part 1: Global Volume Fraction")
    print("=" * 40)

    n_samples = 10000
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    set_global_seed(42)
    orders = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        orders.append(order)
        if (i+1) % 2000 == 0:
            print(f"  Sampled {i+1}/{n_samples}")

    orders = np.array(orders)

    global_results = {
        'n_samples': n_samples,
        'mean_order': float(np.mean(orders)),
        'std_order': float(np.std(orders)),
        'volume_fractions': {}
    }

    print(f"\nMean order from prior: {np.mean(orders):.4f}")
    print(f"Std order from prior: {np.std(orders):.4f}")
    print(f"Max order observed: {np.max(orders):.4f}")
    print("\nVolume fractions (fraction of weight space above threshold):")
    for t in thresholds:
        frac = np.mean(orders >= t)
        global_results['volume_fractions'][str(t)] = float(frac)
        print(f"  Order >= {t:.2f}: {frac:.4f} ({frac*100:.2f}%)")

    global_results['percentiles'] = {
        '50': float(np.percentile(orders, 50)),
        '90': float(np.percentile(orders, 90)),
        '95': float(np.percentile(orders, 95)),
        '99': float(np.percentile(orders, 99)),
    }

    # Part 2: Local basin volume comparison
    print("\n" + "=" * 40)
    print("Part 2: Local Basin Volume by Order Level")
    print("=" * 40)

    local_results = compare_basin_volumes_at_order_levels(
        n_centers=50,
        n_samples_per_center=100,
        radius=0.5,  # Perturbation radius in weight space
        seed=123
    )

    # Statistical analysis
    print("\n" + "=" * 40)
    print("Statistical Analysis")
    print("=" * 40)

    if len(local_results['bins']) >= 2:
        # Compare lowest and highest order bins
        low_bin = local_results['bins'][0]
        high_bin = local_results['bins'][-1]

        low_vols = np.array(low_bin['volumes'])
        high_vols = np.array(high_bin['volumes'])

        # t-test
        t_stat, p_value = stats.ttest_ind(low_vols, high_vols)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(low_vols) + np.var(high_vols)) / 2)
        cohens_d = (np.mean(low_vols) - np.mean(high_vols)) / (pooled_std + 1e-10)

        # Correlation between order and local volume
        all_orders = []
        all_volumes = []
        for bin_data in local_results['bins']:
            for i, vol in enumerate(bin_data['volumes']):
                # Approximate order by bin midpoint
                mid_order = (bin_data['lower'] + bin_data['upper']) / 2
                all_orders.append(mid_order)
                all_volumes.append(vol)

        if len(all_orders) > 2:
            r_corr, p_corr = stats.pearsonr(all_orders, all_volumes)
        else:
            r_corr, p_corr = 0, 1

        print(f"\nLow-order bin volume: {np.mean(low_vols):.4f} +/- {np.std(low_vols):.4f}")
        print(f"High-order bin volume: {np.mean(high_vols):.4f} +/- {np.std(high_vols):.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        print(f"Order-volume correlation: r={r_corr:.4f}, p={p_corr:.6f}")

        analysis = {
            'low_order_volume': float(np.mean(low_vols)),
            'high_order_volume': float(np.mean(high_vols)),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'correlation_r': float(r_corr),
            'correlation_p': float(p_corr)
        }
    else:
        analysis = {'error': 'Not enough bins for comparison'}

    # Summary
    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)

    # Determine validation status
    if 'error' not in analysis:
        if analysis['p_value'] < 0.01 and abs(analysis['cohens_d']) > 0.5:
            if analysis['cohens_d'] > 0:
                status = "VALIDATED: High-order basins have smaller local volume"
            else:
                status = "REFUTED: High-order basins have LARGER local volume"
        else:
            status = "INCONCLUSIVE: Effect size or significance below threshold"
    else:
        status = "INCONCLUSIVE: " + analysis.get('error', 'Unknown error')

    print(f"\nStatus: {status}")

    # Save results
    all_results = {
        'experiment_id': 'RES-172',
        'hypothesis': 'High-order basins occupy smaller volume in weight space than low-order regions',
        'global_volume_analysis': global_results,
        'local_volume_analysis': {
            'radius': local_results['radius'],
            'n_samples_per_center': local_results['n_samples_per_center'],
            'bins': [{k: v for k, v in b.items() if k != 'volumes'} for b in local_results['bins']]
        },
        'statistical_analysis': analysis,
        'status': status
    }

    with open('/Users/matt/Development/monochrome_noise_converger/results/basin_volume/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to results/basin_volume/results.json")

    return all_results


if __name__ == '__main__':
    results = main()
