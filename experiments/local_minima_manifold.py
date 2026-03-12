#!/usr/bin/env python3
"""
RES-147: Local order minima form connected submanifolds rather than isolated points

Tests whether local minima in the order landscape are isolated points or
connected manifolds by:
1. Finding local minima via gradient descent from random starts
2. Checking if nearby minima can be connected by paths that stay near-minimum
3. Computing dimensionality of the minimum-finding basin

If minima form manifolds, we expect:
- Multiple nearby minima that can be connected by low-order paths
- High local dimensionality around minima
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy.stats import pearsonr, ttest_ind
from scipy.optimize import minimize

set_global_seed(42)


def get_order(weights: np.ndarray, cppn: CPPN) -> float:
    """Compute order for given weights."""
    cppn.set_weights(weights)
    img = cppn.render(32)
    return order_multiplicative(img)


def find_local_minimum(cppn: CPPN, max_iter: int = 100) -> tuple:
    """Find local minimum via gradient descent (minimize -order)."""
    w0 = cppn.get_weights()

    def neg_order(w):
        return -get_order(w, cppn)

    result = minimize(neg_order, w0, method='L-BFGS-B',
                     options={'maxiter': max_iter, 'gtol': 1e-6})
    return result.x, -result.fun


def path_stays_near_minimum(w1: np.ndarray, w2: np.ndarray, cppn: CPPN,
                           n_steps: int = 10, threshold: float = 0.8) -> bool:
    """Check if linear path between two points stays above threshold of endpoints."""
    o1 = get_order(w1, cppn)
    o2 = get_order(w2, cppn)
    min_endpoint = min(o1, o2)
    target = min_endpoint * threshold  # Path must stay above 80% of minimum order

    for t in np.linspace(0, 1, n_steps):
        w_interp = (1 - t) * w1 + t * w2
        o_interp = get_order(w_interp, cppn)
        if o_interp < target:
            return False
    return True


def estimate_local_dimension(cppn: CPPN, w_min: np.ndarray,
                            n_directions: int = 20, step_size: float = 0.1) -> float:
    """Estimate local dimensionality by counting flat directions."""
    o_min = get_order(w_min, cppn)
    flat_count = 0

    for _ in range(n_directions):
        direction = np.random.randn(len(w_min))
        direction /= np.linalg.norm(direction)

        # Check if order stays stable in this direction
        w_plus = w_min + step_size * direction
        w_minus = w_min - step_size * direction

        o_plus = get_order(w_plus, cppn)
        o_minus = get_order(w_minus, cppn)

        # Direction is "flat" if order doesn't drop much
        if o_plus > 0.8 * o_min and o_minus > 0.8 * o_min:
            flat_count += 1

    return flat_count / n_directions


def main():
    print("RES-147: Testing if local minima form connected submanifolds")
    print("=" * 60)

    n_trials = 50
    n_minima_per_trial = 5

    connected_pairs = []
    isolated_pairs = []
    local_dims = []

    for trial in range(n_trials):
        if trial % 10 == 0:
            print(f"Trial {trial}/{n_trials}...")

        # Create CPPN and find multiple local minima
        cppn = CPPN()
        minima = []

        for _ in range(n_minima_per_trial):
            # Random starting point
            cppn_copy = cppn.copy()
            cppn_copy.set_weights(np.random.randn(len(cppn.get_weights())))

            w_min, o_min = find_local_minimum(cppn_copy)
            if o_min > 0.1:  # Only consider non-trivial minima
                minima.append((w_min, o_min))

        if len(minima) < 2:
            continue

        # Test connectivity between pairs
        for i in range(len(minima)):
            for j in range(i + 1, len(minima)):
                w1, o1 = minima[i]
                w2, o2 = minima[j]

                dist = np.linalg.norm(w1 - w2)
                connected = path_stays_near_minimum(w1, w2, cppn)

                if connected:
                    connected_pairs.append((dist, min(o1, o2)))
                else:
                    isolated_pairs.append((dist, min(o1, o2)))

        # Estimate local dimension at first minimum
        if minima:
            w_first, _ = minima[0]
            local_dim = estimate_local_dimension(cppn, w_first)
            local_dims.append(local_dim)

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    n_connected = len(connected_pairs)
    n_isolated = len(isolated_pairs)
    total_pairs = n_connected + n_isolated

    print(f"\nConnectivity Analysis:")
    print(f"  Connected pairs: {n_connected}/{total_pairs} ({100*n_connected/total_pairs:.1f}%)")
    print(f"  Isolated pairs: {n_isolated}/{total_pairs} ({100*n_isolated/total_pairs:.1f}%)")

    if connected_pairs and isolated_pairs:
        conn_dists = [p[0] for p in connected_pairs]
        isol_dists = [p[0] for p in isolated_pairs]

        print(f"\n  Connected pair distance: {np.mean(conn_dists):.3f} +/- {np.std(conn_dists):.3f}")
        print(f"  Isolated pair distance: {np.mean(isol_dists):.3f} +/- {np.std(isol_dists):.3f}")

        # T-test on distances
        t_stat, p_val = ttest_ind(conn_dists, isol_dists)
        d = (np.mean(conn_dists) - np.mean(isol_dists)) / np.sqrt(
            (np.var(conn_dists) + np.var(isol_dists)) / 2)

        print(f"\n  Distance difference: t={t_stat:.2f}, p={p_val:.2e}, d={d:.2f}")

    print(f"\nLocal Dimension Analysis:")
    print(f"  Mean flat fraction: {np.mean(local_dims):.3f} +/- {np.std(local_dims):.3f}")
    print(f"  Implies local dim ~{np.mean(local_dims) * 5:.1f}D out of 5D weight space")

    # Hypothesis test: Are minima more connected than isolated?
    connectivity_rate = n_connected / total_pairs if total_pairs > 0 else 0

    # Under null (isolated points), connectivity should be ~0
    # Test if significantly above baseline using binomtest (scipy >= 1.7)
    from scipy.stats import binomtest
    result = binomtest(n_connected, total_pairs, p=0.1, alternative='greater')
    p_manifold = result.pvalue

    print(f"\n  Connectivity rate: {connectivity_rate:.3f}")
    print(f"  P(manifold structure): {p_manifold:.4f}")

    # Effect size: connectivity rate vs baseline expectation of 0.1
    baseline = 0.1
    effect = (connectivity_rate - baseline) / np.sqrt(baseline * (1 - baseline) / total_pairs)

    print(f"\nFINAL VERDICT:")
    print(f"  Effect size (z): {effect:.2f}")
    print(f"  P-value: {p_manifold:.4f}")

    if connectivity_rate > 0.3 and p_manifold < 0.01:
        print("  STATUS: VALIDATED - Minima form connected submanifolds")
        status = "validated"
    elif connectivity_rate < 0.15:
        print("  STATUS: REFUTED - Minima are largely isolated points")
        status = "refuted"
    else:
        print("  STATUS: INCONCLUSIVE - Mixed evidence")
        status = "inconclusive"

    return {
        "status": status,
        "n_connected": n_connected,
        "n_isolated": n_isolated,
        "connectivity_rate": connectivity_rate,
        "mean_local_dim": float(np.mean(local_dims)),
        "effect_size": effect,
        "p_value": p_manifold
    }


if __name__ == "__main__":
    results = main()
    print(f"\nResults dict: {results}")
