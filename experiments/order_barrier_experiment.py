"""
RES-141: Test whether linear interpolation paths between high-order optima
pass through low-order barrier regions.

Hypothesis: Linear interpolation paths between high-order optima pass through
low-order barrier regions in weight space.

Method:
1. Generate many random CPPNs, keep those with high order (top 10%)
2. Take pairs of high-order CPPNs and linearly interpolate weights
3. Measure order along interpolation path
4. Test if minimum order along path is significantly lower than endpoints

If validated: The order landscape has barrier structure, explaining why
gradient methods struggle - they would need to cross low-order valleys.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats
import json


def generate_high_order_cppns(n_candidates: int, top_fraction: float, size: int = 32) -> list:
    """Generate CPPNs and return the top fraction by order."""
    cppns = []
    orders = []

    for _ in range(n_candidates):
        cppn = CPPN()
        img = cppn.render(size)
        order = order_multiplicative(img)
        cppns.append(cppn)
        orders.append(order)

    # Sort by order and take top fraction
    sorted_indices = np.argsort(orders)[::-1]
    n_top = max(2, int(n_candidates * top_fraction))
    top_indices = sorted_indices[:n_top]

    return [(cppns[i], orders[i]) for i in top_indices]


def interpolate_weights(w1: np.ndarray, w2: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation between weight vectors."""
    return (1 - alpha) * w1 + alpha * w2


def measure_path_order(cppn1: CPPN, cppn2: CPPN, n_steps: int = 21, size: int = 32) -> dict:
    """Measure order along linear interpolation path between two CPPNs."""
    w1 = cppn1.get_weights()
    w2 = cppn2.get_weights()

    alphas = np.linspace(0, 1, n_steps)
    orders = []

    # Create a template CPPN for evaluation
    template = cppn1.copy()

    for alpha in alphas:
        w = interpolate_weights(w1, w2, alpha)
        template.set_weights(w)
        img = template.render(size)
        order = order_multiplicative(img)
        orders.append(order)

    orders = np.array(orders)

    return {
        'alphas': alphas.tolist(),
        'orders': orders.tolist(),
        'start_order': orders[0],
        'end_order': orders[-1],
        'min_order': float(np.min(orders)),
        'min_location': float(alphas[np.argmin(orders)]),
        'endpoint_mean': (orders[0] + orders[-1]) / 2,
        'interior_min': float(np.min(orders[1:-1])) if len(orders) > 2 else float(np.min(orders)),
        'barrier_depth': float((orders[0] + orders[-1]) / 2 - np.min(orders[1:-1])) if len(orders) > 2 else 0.0,
        'relative_barrier': float(1 - np.min(orders[1:-1]) / ((orders[0] + orders[-1]) / 2 + 1e-10)) if len(orders) > 2 else 0.0
    }


def run_experiment(
    n_candidates: int = 500,
    top_fraction: float = 0.10,
    n_paths: int = 50,
    n_steps: int = 21,
    size: int = 32,
    seed: int = 42
) -> dict:
    """Run the barrier experiment."""
    set_global_seed(seed)

    print(f"Generating {n_candidates} CPPNs and selecting top {top_fraction*100}%...")
    high_order_cppns = generate_high_order_cppns(n_candidates, top_fraction, size)
    print(f"Selected {len(high_order_cppns)} high-order CPPNs")
    print(f"Order range: {high_order_cppns[-1][1]:.3f} to {high_order_cppns[0][1]:.3f}")

    # Generate random pairs
    n_high = len(high_order_cppns)
    paths = []

    # Take random pairs (without replacement)
    np.random.seed(seed + 1)
    indices = np.random.permutation(n_high)

    for i in range(min(n_paths, n_high // 2)):
        idx1, idx2 = indices[2*i], indices[2*i + 1]
        cppn1, order1 = high_order_cppns[idx1]
        cppn2, order2 = high_order_cppns[idx2]

        path_info = measure_path_order(cppn1, cppn2, n_steps, size)
        path_info['pair_idx'] = (int(idx1), int(idx2))
        paths.append(path_info)

    # Analyze results
    barrier_depths = [p['barrier_depth'] for p in paths]
    relative_barriers = [p['relative_barrier'] for p in paths]
    interior_mins = [p['interior_min'] for p in paths]
    endpoint_means = [p['endpoint_mean'] for p in paths]

    # Statistical tests
    # H0: interior_min >= endpoint_mean (no barrier)
    # H1: interior_min < endpoint_mean (barrier exists)
    t_stat, t_pvalue = stats.ttest_rel(interior_mins, endpoint_means, alternative='less')

    # Effect size (paired Cohen's d)
    diff = np.array(endpoint_means) - np.array(interior_mins)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    # What fraction of paths have barriers?
    has_barrier = [d > 0.01 for d in barrier_depths]  # Threshold for "meaningful" barrier
    barrier_fraction = np.mean(has_barrier)

    # Correlation: do higher endpoint orders have deeper barriers?
    corr, corr_p = stats.pearsonr(endpoint_means, barrier_depths)

    results = {
        'n_candidates': n_candidates,
        'top_fraction': top_fraction,
        'n_paths': len(paths),
        'n_steps': n_steps,

        # Barrier statistics
        'mean_barrier_depth': float(np.mean(barrier_depths)),
        'std_barrier_depth': float(np.std(barrier_depths)),
        'median_barrier_depth': float(np.median(barrier_depths)),
        'mean_relative_barrier': float(np.mean(relative_barriers)),

        # Endpoint vs interior comparison
        'mean_endpoint_order': float(np.mean(endpoint_means)),
        'mean_interior_min': float(np.mean(interior_mins)),

        # Statistical tests
        't_statistic': float(t_stat),
        'p_value': float(t_pvalue),
        'cohens_d': float(cohens_d),

        # Barrier prevalence
        'fraction_with_barrier': float(barrier_fraction),

        # Correlation
        'endpoint_barrier_corr': float(corr),
        'endpoint_barrier_p': float(corr_p),

        # Individual paths for detailed analysis
        'paths': paths
    }

    return results


def main():
    print("=" * 60)
    print("RES-141: Order Barrier Experiment")
    print("=" * 60)
    print("\nHypothesis: Linear interpolation paths between high-order")
    print("optima pass through low-order barrier regions.\n")

    results = run_experiment(
        n_candidates=500,
        top_fraction=0.10,
        n_paths=50,
        n_steps=21,
        size=32,
        seed=42
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nEndpoint order (mean of start/end): {results['mean_endpoint_order']:.4f}")
    print(f"Interior minimum order: {results['mean_interior_min']:.4f}")
    print(f"Mean barrier depth: {results['mean_barrier_depth']:.4f}")
    print(f"Mean relative barrier: {results['mean_relative_barrier']*100:.1f}%")

    print(f"\nFraction of paths with barrier (>0.01): {results['fraction_with_barrier']*100:.1f}%")

    print(f"\nStatistical test (interior < endpoints):")
    print(f"  t-statistic: {results['t_statistic']:.3f}")
    print(f"  p-value: {results['p_value']:.2e}")
    print(f"  Cohen's d: {results['cohens_d']:.3f}")

    print(f"\nEndpoint-barrier correlation: r={results['endpoint_barrier_corr']:.3f}, p={results['endpoint_barrier_p']:.2e}")

    # Determine validation status
    significant = results['p_value'] < 0.01
    large_effect = results['cohens_d'] > 0.5
    prevalent = results['fraction_with_barrier'] > 0.5

    print("\n" + "=" * 60)
    if significant and large_effect:
        print("CONCLUSION: VALIDATED")
        print(f"Barrier effect is significant (p={results['p_value']:.2e}) and large (d={results['cohens_d']:.2f})")
    elif significant:
        print("CONCLUSION: INCONCLUSIVE")
        print(f"Barrier is significant (p={results['p_value']:.2e}) but effect size small (d={results['cohens_d']:.2f})")
    else:
        print("CONCLUSION: REFUTED")
        print(f"No significant barrier effect (p={results['p_value']:.2e})")
    print("=" * 60)

    # Save results
    import os
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/order_barrier', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/order_barrier/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/order_barrier/results.json")

    return results


if __name__ == '__main__':
    main()
