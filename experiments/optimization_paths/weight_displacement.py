"""
RES-193: Successful optimization paths have larger cumulative weight displacement

Hypothesis: Successful optimization paths (reaching high order) accumulate more total
weight displacement than unsuccessful paths.

Rationale:
- RES-185 showed gradients are zero in low-order regions, blocking gradient direction analysis
- But optimization still succeeds sometimes - the question is what distinguishes winners
- Perhaps successful paths explore more weight space (larger total displacement)
- Or perhaps they stay closer to origin (smaller displacement, better basin)

Approach:
1. Use Elliptical Slice Sampling (ESS) as optimization since gradients are often zero
2. Track cumulative weight displacement per path
3. Compare displacement between successful (final order > 0.1) and unsuccessful paths

Effect size threshold: Cohen's d > 0.5 and p < 0.01
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from scipy import stats
import json
from dataclasses import dataclass

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

# Configuration
N_PATHS = 200  # Number of optimization paths
N_STEPS = 50  # Steps per path (ESS is more efficient than gradient)
ORDER_SUCCESS_THRESHOLD = 0.1  # Final order above this = successful path

set_global_seed(42)


def elliptical_slice_step(cppn: CPPN, current_order: float, order_func=order_multiplicative):
    """
    One step of Elliptical Slice Sampling.
    Returns: (new_order, weight_change_norm)
    """
    current_weights = cppn.get_weights()
    n_weights = len(current_weights)

    # Sample from prior (standard normal)
    nu = np.random.randn(n_weights)

    # Initial angle and bracket
    angle = np.random.uniform(0, 2 * np.pi)
    angle_min = angle - 2 * np.pi
    angle_max = angle

    # Threshold (current order is the threshold)
    threshold = current_order

    # ESS loop
    max_iterations = 100
    for _ in range(max_iterations):
        # Propose new weights on ellipse
        new_weights = current_weights * np.cos(angle) + nu * np.sin(angle)
        cppn.set_weights(new_weights)

        # Evaluate
        img = cppn.render(32)
        new_order = order_func(img)

        if new_order > threshold:
            # Accept
            weight_change = np.linalg.norm(new_weights - current_weights)
            return new_order, weight_change

        # Shrink bracket
        if angle < 0:
            angle_min = angle
        else:
            angle_max = angle

        # Resample angle
        angle = np.random.uniform(angle_min, angle_max)

    # If we exit the loop without acceptance, stay at current position
    cppn.set_weights(current_weights)
    return current_order, 0.0


def run_ess_path(seed):
    """
    Run one ESS optimization path.
    Returns path statistics.
    """
    np.random.seed(seed)
    cppn = CPPN()  # Random initialization
    start_weights = cppn.get_weights().copy()

    orders = []
    cumulative_displacement = 0.0
    displacements_per_step = []

    img = cppn.render(32)
    current_order = order_multiplicative(img)
    orders.append(current_order)

    for step in range(N_STEPS):
        new_order, step_displacement = elliptical_slice_step(cppn, current_order)
        current_order = new_order
        orders.append(current_order)
        displacements_per_step.append(step_displacement)
        cumulative_displacement += step_displacement

    final_weights = cppn.get_weights()
    total_displacement = np.linalg.norm(final_weights - start_weights)

    return {
        'final_order': current_order,
        'max_order': max(orders),
        'cumulative_displacement': cumulative_displacement,
        'total_displacement': total_displacement,
        'mean_step_displacement': np.mean(displacements_per_step),
        'std_step_displacement': np.std(displacements_per_step),
        'n_nonzero_steps': sum(1 for d in displacements_per_step if d > 1e-10),
        'orders': orders,
    }


def main():
    print("RES-193: Successful optimization paths have larger cumulative weight displacement")
    print("="*70)
    print(f"Running {N_PATHS} ESS paths, {N_STEPS} steps each")
    print(f"Success threshold: final order > {ORDER_SUCCESS_THRESHOLD}")
    print()

    results = []
    for i in range(N_PATHS):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{N_PATHS}")
        result = run_ess_path(seed=i*7 + 456)
        results.append(result)

    # Split into successful vs unsuccessful paths
    successful = [r for r in results if r['final_order'] >= ORDER_SUCCESS_THRESHOLD]
    unsuccessful = [r for r in results if r['final_order'] < ORDER_SUCCESS_THRESHOLD]

    print(f"\nSuccessful paths: {len(successful)}")
    print(f"Unsuccessful paths: {len(unsuccessful)}")

    # Check if we have enough samples
    if len(successful) < 10 or len(unsuccessful) < 10:
        # Adjust threshold dynamically
        median_order = np.median([r['final_order'] for r in results])
        print(f"\nAdjusting threshold to median: {median_order:.4f}")
        successful = [r for r in results if r['final_order'] >= median_order]
        unsuccessful = [r for r in results if r['final_order'] < median_order]
        print(f"  Successful (above median): {len(successful)}")
        print(f"  Unsuccessful (below median): {len(unsuccessful)}")

    if len(successful) < 5 or len(unsuccessful) < 5:
        print("Not enough samples in each group for analysis")
        summary = {
            'status': 'inconclusive',
            'reason': 'Insufficient samples in groups',
            'n_successful': len(successful),
            'n_unsuccessful': len(unsuccessful)
        }
    else:
        # Extract displacement values
        succ_cumulative = [r['cumulative_displacement'] for r in successful]
        unsucc_cumulative = [r['cumulative_displacement'] for r in unsuccessful]

        succ_total = [r['total_displacement'] for r in successful]
        unsucc_total = [r['total_displacement'] for r in unsuccessful]

        succ_nonzero = [r['n_nonzero_steps'] for r in successful]
        unsucc_nonzero = [r['n_nonzero_steps'] for r in unsuccessful]

        # Statistics - cumulative displacement
        mean_succ_cumul = np.mean(succ_cumulative)
        mean_unsucc_cumul = np.mean(unsucc_cumulative)

        u_stat, p_value = stats.mannwhitneyu(succ_cumulative, unsucc_cumulative, alternative='two-sided')

        pooled_std = np.sqrt((np.std(succ_cumulative)**2 + np.std(unsucc_cumulative)**2) / 2)
        if pooled_std > 1e-10:
            cohens_d = (mean_succ_cumul - mean_unsucc_cumul) / pooled_std
        else:
            cohens_d = 0.0

        # Correlation between cumulative displacement and final order
        all_cumul = [r['cumulative_displacement'] for r in results]
        all_final = [r['final_order'] for r in results]
        rho, rho_p = stats.spearmanr(all_cumul, all_final)

        # Print results
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")

        print(f"\nCumulative weight displacement:")
        print(f"  Successful paths:   mean={mean_succ_cumul:.3f}, std={np.std(succ_cumulative):.3f}")
        print(f"  Unsuccessful paths: mean={mean_unsucc_cumul:.3f}, std={np.std(unsucc_cumulative):.3f}")
        print(f"\n  Mann-Whitney p-value: {p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        print(f"\nTotal displacement (start to end):")
        print(f"  Successful paths:   mean={np.mean(succ_total):.3f}")
        print(f"  Unsuccessful paths: mean={np.mean(unsucc_total):.3f}")

        print(f"\nNon-zero steps (moves accepted):")
        print(f"  Successful paths:   mean={np.mean(succ_nonzero):.1f}/{N_STEPS}")
        print(f"  Unsuccessful paths: mean={np.mean(unsucc_nonzero):.1f}/{N_STEPS}")

        print(f"\nCorrelation: cumulative displacement vs final order")
        print(f"  Spearman rho: {rho:.3f} (p={rho_p:.2e})")

        # Order statistics
        print(f"\nOrder statistics:")
        print(f"  Successful mean final order: {np.mean([r['final_order'] for r in successful]):.4f}")
        print(f"  Unsuccessful mean final order: {np.mean([r['final_order'] for r in unsuccessful]):.4f}")

        # Determine status
        if p_value < 0.01 and abs(cohens_d) > 0.5:
            if cohens_d > 0:
                status = 'validated'
                conclusion = "Successful paths have LARGER cumulative weight displacement"
            else:
                status = 'refuted'
                conclusion = "Successful paths have SMALLER cumulative weight displacement (opposite direction)"
        elif p_value < 0.05:
            status = 'inconclusive'
            conclusion = f"Trend exists (p={p_value:.3f}) but effect size ({cohens_d:.2f}) below threshold"
        else:
            status = 'refuted'
            conclusion = f"No significant difference (p={p_value:.3f}, d={cohens_d:.2f})"

        print(f"\n{'='*70}")
        print(f"CONCLUSION: {status.upper()}")
        print(f"{conclusion}")
        print(f"{'='*70}")

        # Build summary
        summary = {
            'experiment_id': 'RES-193',
            'hypothesis': 'Successful optimization paths have larger cumulative weight displacement',
            'status': status,
            'conclusion': conclusion,
            'n_paths': N_PATHS,
            'n_steps': N_STEPS,
            'success_threshold': ORDER_SUCCESS_THRESHOLD,
            'n_successful': len(successful),
            'n_unsuccessful': len(unsuccessful),
            'metrics': {
                'mean_succ_cumulative_displacement': float(mean_succ_cumul),
                'mean_unsucc_cumulative_displacement': float(mean_unsucc_cumul),
                'mean_succ_total_displacement': float(np.mean(succ_total)),
                'mean_unsucc_total_displacement': float(np.mean(unsucc_total)),
                'mean_succ_nonzero_steps': float(np.mean(succ_nonzero)),
                'mean_unsucc_nonzero_steps': float(np.mean(unsucc_nonzero)),
                'mann_whitney_p': float(p_value),
                'cohens_d': float(cohens_d),
                'spearman_rho': float(rho),
                'spearman_p': float(rho_p),
            },
            'order_stats': {
                'mean_final_successful': float(np.mean([r['final_order'] for r in successful])),
                'mean_final_unsuccessful': float(np.mean([r['final_order'] for r in unsuccessful])),
                'success_rate': len(successful) / N_PATHS,
            }
        }

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '../../results/optimization_paths/weight_displacement_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == '__main__':
    main()
