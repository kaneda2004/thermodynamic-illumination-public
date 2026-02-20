"""
RES-186: Multi-slice ESS vs Single-slice ESS comparison

HYPOTHESIS: Multi-slice ESS (multiple nu vectors) finds valid proposals
faster than single-slice ESS

RATIONALE: Standard ESS defines a single ellipse per sample via one random
direction nu from the prior. If this direction doesn't intersect the
constraint region well, many contractions are needed. Multi-slice ESS
maintains K directions and can switch between them, potentially finding
valid proposals with fewer total evaluations.

METHOD:
1. Run nested sampling trials with single-slice ESS (baseline, K=1)
2. Run trials with multi-slice ESS (K=3, K=5)
3. Compare: contractions per accepted sample, total evaluations, final order

SUCCESS CRITERIA:
- Effect size d > 0.5 in contractions/sample
- p < 0.01 (two-tailed t-test)
"""

import numpy as np
import sys
import os
import json
from scipy import stats
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, PRIOR_SIGMA, order_multiplicative, log_prior, set_global_seed
)


def single_slice_ess(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
):
    """Standard ESS with one nu vector (baseline)."""
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0
    total_evals = 0

    for restart in range(max_restarts):
        nu = np.random.randn(n_params) * PRIOR_SIGMA
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi
        n_contractions = 0

        while n_contractions < max_contractions:
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)
            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_fn(proposal_img)
            total_evals += 1

            if proposal_order >= threshold:
                return proposal_cppn, proposal_img, proposal_order, total_contractions + n_contractions, total_evals, True

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

            if phi_max - phi_min < 1e-10:
                break

        total_contractions += n_contractions

    current_img = cppn.render(image_size)
    return cppn, current_img, order_fn(current_img), total_contractions, total_evals, False


def multi_slice_ess(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    n_slices: int = 3,
    max_contractions_per_slice: int = 30,
    max_rounds: int = 5
):
    """
    Multi-slice ESS: maintain K independent slices and cycle through them.

    Instead of exhausting one slice before restarting, we contract each
    slice in a round-robin fashion, potentially finding the constraint
    region faster by exploring multiple directions simultaneously.
    """
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0
    total_evals = 0

    for round_idx in range(max_rounds):
        # Initialize K slices with independent random directions
        slices = []
        for _ in range(n_slices):
            nu = np.random.randn(n_params) * PRIOR_SIGMA
            phi = np.random.uniform(0, 2 * np.pi)
            phi_min = phi - 2 * np.pi
            phi_max = phi
            slices.append({
                'nu': nu,
                'phi': phi,
                'phi_min': phi_min,
                'phi_max': phi_max,
                'active': True,
                'contractions': 0
            })

        # Round-robin through slices
        active_count = n_slices
        while active_count > 0:
            for s in slices:
                if not s['active']:
                    continue

                # Try proposal on this slice
                proposal_w = current_w * np.cos(s['phi']) + s['nu'] * np.sin(s['phi'])
                proposal_cppn = cppn.copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_fn(proposal_img)
                total_evals += 1

                if proposal_order >= threshold:
                    return proposal_cppn, proposal_img, proposal_order, total_contractions, total_evals, True

                # Contract this slice
                if s['phi'] < 0:
                    s['phi_min'] = s['phi']
                else:
                    s['phi_max'] = s['phi']

                s['phi'] = np.random.uniform(s['phi_min'], s['phi_max'])
                s['contractions'] += 1
                total_contractions += 1

                # Deactivate if exhausted
                if s['contractions'] >= max_contractions_per_slice or s['phi_max'] - s['phi_min'] < 1e-10:
                    s['active'] = False
                    active_count -= 1

    current_img = cppn.render(image_size)
    return cppn, current_img, order_fn(current_img), total_contractions, total_evals, False


def run_ns_comparison(
    n_live: int = 50,
    n_iterations: int = 200,
    image_size: int = 32,
    seed: int = 42,
    method: str = "single",
    n_slices: int = 3
):
    """Run nested sampling and track contractions/evaluations."""
    set_global_seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({'cppn': cppn, 'image': img, 'order': order})

    contractions_per_sample = []
    evals_per_sample = []
    final_orders = []
    successes = 0

    for iteration in range(n_iterations):
        # Find worst
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']

        # Select random seed from valid points
        valid_seeds = [i for i in range(n_live)
                      if i != worst_idx and live_points[i]['order'] >= threshold]
        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = np.random.choice(valid_seeds)
        seed_cppn = live_points[seed_idx]['cppn']

        # Run ESS variant
        if method == "single":
            new_cppn, new_img, new_order, contractions, evals, success = single_slice_ess(
                seed_cppn, threshold, image_size, order_multiplicative
            )
        else:
            new_cppn, new_img, new_order, contractions, evals, success = multi_slice_ess(
                seed_cppn, threshold, image_size, order_multiplicative, n_slices=n_slices
            )

        contractions_per_sample.append(contractions)
        evals_per_sample.append(evals)

        if success:
            successes += 1
            live_points[worst_idx] = {'cppn': new_cppn, 'image': new_img, 'order': new_order}

        final_orders.append(max(lp['order'] for lp in live_points))

    return {
        'contractions': contractions_per_sample,
        'evals': evals_per_sample,
        'final_orders': final_orders,
        'success_rate': successes / n_iterations,
        'mean_contractions': np.mean(contractions_per_sample),
        'mean_evals': np.mean(evals_per_sample),
        'max_order': max(lp['order'] for lp in live_points)
    }


def main():
    print("=" * 70)
    print("RES-186: Multi-slice ESS vs Single-slice ESS")
    print("=" * 70)

    n_trials = 20
    n_live = 50
    n_iterations = 200

    single_results = []
    multi3_results = []
    multi5_results = []

    print(f"\nRunning {n_trials} trials each...")
    print("-" * 70)

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...", end=" ")

        # Use same seed base for fair comparison
        seed_base = trial * 1000

        r_single = run_ns_comparison(
            n_live=n_live, n_iterations=n_iterations,
            seed=seed_base, method="single"
        )
        single_results.append(r_single)

        r_multi3 = run_ns_comparison(
            n_live=n_live, n_iterations=n_iterations,
            seed=seed_base, method="multi", n_slices=3
        )
        multi3_results.append(r_multi3)

        r_multi5 = run_ns_comparison(
            n_live=n_live, n_iterations=n_iterations,
            seed=seed_base, method="multi", n_slices=5
        )
        multi5_results.append(r_multi5)

        print(f"Single: {r_single['mean_contractions']:.1f} contr, "
              f"Multi-3: {r_multi3['mean_contractions']:.1f} contr, "
              f"Multi-5: {r_multi5['mean_contractions']:.1f} contr")

    # Aggregate statistics
    single_contractions = [r['mean_contractions'] for r in single_results]
    multi3_contractions = [r['mean_contractions'] for r in multi3_results]
    multi5_contractions = [r['mean_contractions'] for r in multi5_results]

    single_evals = [r['mean_evals'] for r in single_results]
    multi3_evals = [r['mean_evals'] for r in multi3_results]
    multi5_evals = [r['mean_evals'] for r in multi5_results]

    single_orders = [r['max_order'] for r in single_results]
    multi3_orders = [r['max_order'] for r in multi3_results]
    multi5_orders = [r['max_order'] for r in multi5_results]

    single_success = [r['success_rate'] for r in single_results]
    multi3_success = [r['success_rate'] for r in multi3_results]
    multi5_success = [r['success_rate'] for r in multi5_results]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nMean Contractions per Sample:")
    print(f"  Single-slice:  {np.mean(single_contractions):.2f} +/- {np.std(single_contractions):.2f}")
    print(f"  Multi-slice-3: {np.mean(multi3_contractions):.2f} +/- {np.std(multi3_contractions):.2f}")
    print(f"  Multi-slice-5: {np.mean(multi5_contractions):.2f} +/- {np.std(multi5_contractions):.2f}")

    print("\nMean Evaluations per Sample:")
    print(f"  Single-slice:  {np.mean(single_evals):.2f} +/- {np.std(single_evals):.2f}")
    print(f"  Multi-slice-3: {np.mean(multi3_evals):.2f} +/- {np.std(multi3_evals):.2f}")
    print(f"  Multi-slice-5: {np.mean(multi5_evals):.2f} +/- {np.std(multi5_evals):.2f}")

    print("\nFinal Max Order:")
    print(f"  Single-slice:  {np.mean(single_orders):.4f} +/- {np.std(single_orders):.4f}")
    print(f"  Multi-slice-3: {np.mean(multi3_orders):.4f} +/- {np.std(multi3_orders):.4f}")
    print(f"  Multi-slice-5: {np.mean(multi5_orders):.4f} +/- {np.std(multi5_orders):.4f}")

    print("\nSuccess Rate:")
    print(f"  Single-slice:  {np.mean(single_success):.3f}")
    print(f"  Multi-slice-3: {np.mean(multi3_success):.3f}")
    print(f"  Multi-slice-5: {np.mean(multi5_success):.3f}")

    # Statistical tests: Single vs Multi-3
    t_contr, p_contr = stats.ttest_rel(single_contractions, multi3_contractions)
    t_evals, p_evals = stats.ttest_rel(single_evals, multi3_evals)
    t_order, p_order = stats.ttest_rel(single_orders, multi3_orders)

    # Effect sizes (Cohen's d for paired samples)
    diff_contr = np.array(single_contractions) - np.array(multi3_contractions)
    d_contr = np.mean(diff_contr) / np.std(diff_contr)

    diff_evals = np.array(single_evals) - np.array(multi3_evals)
    d_evals = np.mean(diff_evals) / np.std(diff_evals)

    diff_order = np.array(multi3_orders) - np.array(single_orders)
    d_order = np.mean(diff_order) / np.std(diff_order) if np.std(diff_order) > 0 else 0

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Single vs Multi-3)")
    print("=" * 70)

    print(f"\nContractions: t={t_contr:.3f}, p={p_contr:.2e}, d={d_contr:.3f}")
    print(f"Evaluations:  t={t_evals:.3f}, p={p_evals:.2e}, d={d_evals:.3f}")
    print(f"Max Order:    t={t_order:.3f}, p={p_order:.2e}, d={d_order:.3f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if p_contr < 0.01 and d_contr > 0.5:
        verdict = "VALIDATED"
        reason = f"Multi-slice uses fewer contractions with strong effect (d={d_contr:.2f}, p={p_contr:.2e})"
    elif p_contr < 0.01 and d_contr < -0.5:
        verdict = "REFUTED"
        reason = f"Multi-slice uses MORE contractions with strong effect (d={d_contr:.2f}, p={p_contr:.2e})"
    elif p_contr >= 0.01:
        verdict = "INCONCLUSIVE"
        reason = f"No significant difference (p={p_contr:.2e})"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"Effect too small (d={d_contr:.2f})"

    print(f"\nVERDICT: {verdict}")
    print(f"REASON: {reason}")

    # Save results
    os.makedirs("results/multi_slice_ess", exist_ok=True)
    results = {
        'hypothesis': 'Multi-slice ESS finds valid proposals faster than single-slice ESS',
        'verdict': verdict,
        'metrics': {
            'single_mean_contractions': float(np.mean(single_contractions)),
            'multi3_mean_contractions': float(np.mean(multi3_contractions)),
            'multi5_mean_contractions': float(np.mean(multi5_contractions)),
            'single_mean_evals': float(np.mean(single_evals)),
            'multi3_mean_evals': float(np.mean(multi3_evals)),
            'multi5_mean_evals': float(np.mean(multi5_evals)),
            'single_max_order': float(np.mean(single_orders)),
            'multi3_max_order': float(np.mean(multi3_orders)),
            'multi5_max_order': float(np.mean(multi5_orders)),
            't_contractions': float(t_contr),
            'p_contractions': float(p_contr),
            'd_contractions': float(d_contr),
            't_evals': float(t_evals),
            'p_evals': float(p_evals),
            'd_evals': float(d_evals)
        },
        'n_trials': n_trials,
        'n_live': n_live,
        'n_iterations': n_iterations
    }

    with open("results/multi_slice_ess/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/multi_slice_ess/results.json")

    return verdict, d_contr, p_contr


if __name__ == "__main__":
    verdict, effect_size, p_value = main()
