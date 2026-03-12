"""
RES-185: Early gradient direction stability predicts optimization path success to high-order

Hypothesis: Successful optimization paths (reaching high order) have more stable/consistent
gradient directions in early iterations compared to paths that get stuck at low order.

Rationale:
- RES-145 showed gradient magnitude increases during NS progression
- RES-161 showed trajectory curvature is constant
- RES-136 found weight change correlations similar for high/low order outcomes
- But nobody tested: does early gradient direction CONSISTENCY predict ultimate success?

Approach:
1. Run many gradient descent optimization paths from random starts
2. Compute gradient direction (normalized) at each step
3. Measure early-window gradient consistency (mean pairwise cosine similarity)
4. Compare consistency between successful (final order > 0.3) and unsuccessful paths

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
N_STEPS = 100  # Steps per path
LEARNING_RATE = 0.05
EPSILON = 1e-5  # For finite difference gradient
EARLY_WINDOW = 20  # First 20 steps for "early" analysis
ORDER_SUCCESS_THRESHOLD = 0.05  # Final order above this = successful path (realistic given RES-034)

set_global_seed(42)


def finite_difference_gradient(cppn: CPPN, order_func=order_multiplicative, eps=EPSILON):
    """Compute gradient of order metric w.r.t. weights via finite differences."""
    base_weights = cppn.get_weights()
    n_weights = len(base_weights)

    base_img = cppn.render(32)
    base_order = order_func(base_img)

    gradient = np.zeros(n_weights)
    for i in range(n_weights):
        w_plus = base_weights.copy()
        w_plus[i] += eps
        cppn.set_weights(w_plus)
        img_plus = cppn.render(32)
        order_plus = order_func(img_plus)

        w_minus = base_weights.copy()
        w_minus[i] -= eps
        cppn.set_weights(w_minus)
        img_minus = cppn.render(32)
        order_minus = order_func(img_minus)

        gradient[i] = (order_plus - order_minus) / (2 * eps)

    # Restore original weights
    cppn.set_weights(base_weights)
    return gradient


def normalize(v):
    """Normalize vector to unit length, return zeros if input is zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def compute_direction_consistency(directions):
    """
    Compute mean pairwise cosine similarity of a list of direction vectors.
    Higher values = more consistent direction.

    Returns tuple: (consistency, n_valid_pairs, mean_norm)
    """
    n = len(directions)
    if n < 2:
        return 0.0, 0, 0.0

    similarities = []
    norms = [np.linalg.norm(d) for d in directions]
    mean_norm = np.mean(norms)

    for i in range(n):
        for j in range(i+1, n):
            # Cosine similarity
            d1 = directions[i]
            d2 = directions[j]
            norm1, norm2 = norms[i], norms[j]
            if norm1 > 1e-10 and norm2 > 1e-10:
                sim = np.dot(d1, d2) / (norm1 * norm2)
                similarities.append(sim)

    n_valid = len(similarities)
    mean_sim = np.mean(similarities) if similarities else 0.0
    return mean_sim, n_valid, mean_norm


def run_optimization_path(seed):
    """
    Run one gradient ascent optimization path.
    Returns: (final_order, early_consistency, full_trajectory_orders, early_directions)
    """
    np.random.seed(seed)
    cppn = CPPN()  # Random initialization

    orders = []
    gradients = []  # Store raw gradients, not normalized

    for step in range(N_STEPS):
        img = cppn.render(32)
        order = order_multiplicative(img)
        orders.append(order)

        # Compute gradient
        gradient = finite_difference_gradient(cppn)
        gradients.append(gradient.copy())  # Store raw gradient

        # Gradient ascent step
        weights = cppn.get_weights()
        new_weights = weights + LEARNING_RATE * gradient
        cppn.set_weights(new_weights)

    # Final order
    final_img = cppn.render(32)
    final_order = order_multiplicative(final_img)
    orders.append(final_order)

    # Compute gradient magnitude statistics
    grad_norms = [np.linalg.norm(g) for g in gradients]
    early_grad_norms = grad_norms[:EARLY_WINDOW]
    late_grad_norms = grad_norms[-EARLY_WINDOW:]

    # Early window consistency
    early_consistency, early_n_valid, early_mean_norm = compute_direction_consistency(gradients[:EARLY_WINDOW])

    # Also compute late consistency for comparison
    late_consistency, late_n_valid, late_mean_norm = compute_direction_consistency(gradients[-EARLY_WINDOW:])

    return {
        'final_order': final_order,
        'early_consistency': early_consistency,
        'late_consistency': late_consistency,
        'early_n_valid': early_n_valid,
        'late_n_valid': late_n_valid,
        'early_mean_norm': early_mean_norm,
        'late_mean_norm': late_mean_norm,
        'early_mean_grad_norm': np.mean(early_grad_norms),
        'late_mean_grad_norm': np.mean(late_grad_norms),
        'orders': orders,
        'max_order': max(orders),
    }


def main():
    print("RES-185: Early gradient direction stability predicts optimization path success")
    print("="*70)
    print(f"Running {N_PATHS} optimization paths, {N_STEPS} steps each")
    print(f"Early window: first {EARLY_WINDOW} steps")
    print(f"Success threshold: final order > {ORDER_SUCCESS_THRESHOLD}")
    print()

    results = []
    for i in range(N_PATHS):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{N_PATHS}")
        result = run_optimization_path(seed=i*7 + 123)
        results.append(result)

    # Split into successful vs unsuccessful paths
    successful = [r for r in results if r['final_order'] >= ORDER_SUCCESS_THRESHOLD]
    unsuccessful = [r for r in results if r['final_order'] < ORDER_SUCCESS_THRESHOLD]

    print(f"\nSuccessful paths: {len(successful)}")
    print(f"Unsuccessful paths: {len(unsuccessful)}")

    # Debug: Check gradient norms
    all_early_norms = [r['early_mean_norm'] for r in results]
    all_late_norms = [r['late_mean_norm'] for r in results]
    all_early_valid = [r['early_n_valid'] for r in results]
    all_early_grad_norms = [r['early_mean_grad_norm'] for r in results]
    all_late_grad_norms = [r['late_mean_grad_norm'] for r in results]
    print(f"\nDebug - Gradient statistics:")
    print(f"  Raw early grad norms: {np.mean(all_early_grad_norms):.6f} (max={np.max(all_early_grad_norms):.6f})")
    print(f"  Raw late grad norms: {np.mean(all_late_grad_norms):.6f} (max={np.max(all_late_grad_norms):.6f})")
    print(f"  Early mean norm (for cosine): {np.mean(all_early_norms):.6f}")
    print(f"  Late mean norm (for cosine): {np.mean(all_late_norms):.6f}")
    print(f"  Early valid pairs: {np.mean(all_early_valid):.1f}")

    if len(successful) < 5 or len(unsuccessful) < 5:
        print("Not enough samples in each group for analysis")
        summary = {
            'status': 'inconclusive',
            'reason': 'Insufficient samples in groups',
            'n_successful': len(successful),
            'n_unsuccessful': len(unsuccessful)
        }
    else:
        # Extract early consistency values
        succ_early = [r['early_consistency'] for r in successful]
        unsucc_early = [r['early_consistency'] for r in unsuccessful]

        succ_late = [r['late_consistency'] for r in successful]
        unsucc_late = [r['late_consistency'] for r in unsuccessful]

        # Statistics
        mean_succ_early = np.mean(succ_early)
        mean_unsucc_early = np.mean(unsucc_early)

        mean_succ_late = np.mean(succ_late)
        mean_unsucc_late = np.mean(unsucc_late)

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(succ_early, unsucc_early, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((np.std(succ_early)**2 + np.std(unsucc_early)**2) / 2)
        if pooled_std > 1e-10:
            cohens_d = (mean_succ_early - mean_unsucc_early) / pooled_std
        else:
            cohens_d = 0.0

        # Spearman correlation between early consistency and final order
        all_early = [r['early_consistency'] for r in results]
        all_final = [r['final_order'] for r in results]
        rho, rho_p = stats.spearmanr(all_early, all_final)

        # Print results
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"\nEarly gradient direction consistency (first {EARLY_WINDOW} steps):")
        print(f"  Successful paths:   mean={mean_succ_early:.4f}, std={np.std(succ_early):.4f}")
        print(f"  Unsuccessful paths: mean={mean_unsucc_early:.4f}, std={np.std(unsucc_early):.4f}")
        print(f"\n  Mann-Whitney p-value: {p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        print(f"\nLate gradient direction consistency (last {EARLY_WINDOW} steps):")
        print(f"  Successful paths:   mean={mean_succ_late:.4f}")
        print(f"  Unsuccessful paths: mean={mean_unsucc_late:.4f}")

        print(f"\nCorrelation: early consistency vs final order")
        print(f"  Spearman rho: {rho:.3f} (p={rho_p:.2e})")

        # Order statistics
        print(f"\nOrder statistics:")
        print(f"  Successful mean final order: {np.mean([r['final_order'] for r in successful]):.3f}")
        print(f"  Unsuccessful mean final order: {np.mean([r['final_order'] for r in unsuccessful]):.3f}")

        # Determine status
        if p_value < 0.01 and abs(cohens_d) > 0.5:
            if cohens_d > 0:
                status = 'validated'
                conclusion = "Successful paths have MORE consistent early gradient directions"
            else:
                status = 'refuted'
                conclusion = "Successful paths have LESS consistent early gradient directions"
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
            'experiment_id': 'RES-185',
            'hypothesis': 'Early gradient direction stability predicts optimization path success to high-order',
            'status': status,
            'conclusion': conclusion,
            'n_paths': N_PATHS,
            'n_steps': N_STEPS,
            'early_window': EARLY_WINDOW,
            'success_threshold': ORDER_SUCCESS_THRESHOLD,
            'n_successful': len(successful),
            'n_unsuccessful': len(unsuccessful),
            'metrics': {
                'mean_succ_early_consistency': float(mean_succ_early),
                'mean_unsucc_early_consistency': float(mean_unsucc_early),
                'mean_succ_late_consistency': float(mean_succ_late),
                'mean_unsucc_late_consistency': float(mean_unsucc_late),
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
    output_path = os.path.join(os.path.dirname(__file__), '../../results/optimization_paths/early_gradient_stability_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == '__main__':
    main()
