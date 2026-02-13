#!/usr/bin/env python3
"""
RES-228: Order-Predictive Sampling - Using Eff_Dim During Sampling as Adaptive Guide

HYPOTHESIS: Using PARTIAL effective dimensionality measurements taken during sampling
(every 10-20 samples) as a feedback signal to dynamically adjust n_live achieves
≥2× speedup to reach order > 0.5 while maintaining success rate ≥90%.

RATIONALE:
- RES-218: Eff_dim collapses from ~4.12D to ~1.45D as order increases to 0.5
- RES-221: Pre-sampling eff_dim doesn't strongly predict order (r < 0.4)
- RES-220: Constrained sampling (exploiting known manifolds) achieves 3-5× speedup
- NEW: What if we measure eff_dim DURING sampling to detect when we're collapsing?

CORE INSIGHT: If effective dimensionality DROPS during sampling, the posterior
is collapsing (exploring a low-D manifold). We should INCREASE exploration (n_live)
to escape local minima. If eff_dim STAYS HIGH, we're still exploring randomly;
keep current n_live or decrease slightly to exploit.

METHOD:
1. Initialize baseline: 20 CPPNs sampled with fixed n_live=100 to order > 0.5
   Record: total samples needed, success rate

2. Implement adaptive strategy:
   - Start with n_live=100 (same baseline)
   - Every 20 samples, compute eff_dim of current weight samples
   - Decision rule:
     * If eff_dim > 3.0 (not collapsing): keep or slightly INCREASE n_live (→120)
     * If 2.0 < eff_dim ≤ 3.0 (normal): keep n_live
     * If eff_dim ≤ 2.0 (collapsing): INCREASE n_live (→150) - exploit manifold
     * If eff_dim decreases >20% from last check: INCREASE n_live (sharp collapse)
   - Run on same 20 CPPNs
   - Record: total samples needed, success rate

3. Calculate metrics:
   - Speedup = baseline_samples / adaptive_samples
   - Success rate comparison
   - Verify speedup ≥ 2.0× AND success ≥ 90%

EXPECTED OUTCOME:
Adaptive sampling detects the eff_dim collapse signal and responds by increasing
exploration, efficiently finding high-order configurations.
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from scipy import stats
from typing import Tuple, List, Dict
import random

# Ensure working directory
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA
)


# ============================================================================
# EFFECTIVE DIMENSIONALITY MEASUREMENT
# ============================================================================

def compute_effective_dimensionality_from_samples(weight_samples: np.ndarray) -> Dict[str, float]:
    """
    Compute effective dimensionality from a batch of weight samples.

    weight_samples: shape (n_samples, n_params)

    Returns metrics including:
    - effective_dim: Renyi entropy measure
    - first_pc_var: fraction of variance in first PC
    - n_components_90: PCs needed for 90% variance
    """
    if weight_samples.shape[0] < 2:
        # Not enough samples for meaningful PCA
        return {
            'effective_dim': np.nan,
            'first_pc_var': np.nan,
            'n_components_90': np.nan,
        }

    # Center
    W_centered = weight_samples - weight_samples.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Explained variance
    explained_var = (S ** 2) / np.sum(S ** 2)
    cumsum_var = np.cumsum(explained_var)

    # Effective dimensionality (Renyi entropy: 1/sum(p_i^2))
    eff_dim = 1.0 / np.sum(explained_var ** 2) if np.sum(explained_var ** 2) > 0 else len(explained_var)

    # Components needed for 90% variance
    n_comp_90 = int(np.argmax(cumsum_var >= 0.90) + 1 if np.any(cumsum_var >= 0.90) else len(explained_var))

    return {
        'effective_dim': float(eff_dim),
        'first_pc_var': float(explained_var[0]) if len(explained_var) > 0 else np.nan,
        'n_components_90': n_comp_90,
        'explained_variance_fraction': float(explained_var[0])
    }


# ============================================================================
# ADAPTIVE NESTED SAMPLING WITH EFF_DIM FEEDBACK
# ============================================================================

def nested_sample_with_adaptive_n_live(
    cppn: CPPN,
    order_target: float = 0.5,
    initial_n_live: int = 100,
    check_interval: int = 20,
    max_iterations: int = 300,
    image_size: int = 32,
    seed: int = None
) -> Dict:
    """
    Run nested sampling with adaptive n_live based on effective dimensionality feedback.

    Strategy:
    - Every check_interval samples, measure eff_dim
    - Adjust n_live based on eff_dim dynamics
    - Return samples needed to reach order_target and final order
    """
    if seed is not None:
        set_global_seed(seed)

    n_live = initial_n_live
    live_points = []  # List of (log_likelihood, weight_dict)
    weight_samples = []  # For eff_dim computation
    sample_count = 0
    iteration = 0

    # Initialize live points
    while len(live_points) < n_live and iteration < max_iterations:
        cppn_candidate = CPPN()
        img = cppn_candidate.render(image_size)
        order = order_multiplicative(img)
        log_L = log_prior(cppn_candidate)  # Prior as likelihood

        if not np.isfinite(log_L):
            log_L = -1e10

        live_points.append({
            'log_L': log_L,
            'order': order,
            'weights': cppn_candidate.get_weights().flatten(),
            'cppn': cppn_candidate
        })
        weight_samples.append(live_points[-1]['weights'])
        iteration += 1
        sample_count += 1

    if len(live_points) < n_live:
        return {
            'success': False,
            'reason': 'Failed to initialize live points',
            'samples_used': sample_count,
            'final_order': 0.0,
            'n_live_adjustments': [],
            'eff_dim_history': []
        }

    # Nested sampling loop
    dead_points = []
    eff_dim_history = []
    n_live_adjustments = []
    last_eff_dim = None
    check_counter = 0

    while iteration < max_iterations:
        # Sort live points by log_likelihood
        live_points.sort(key=lambda x: x['log_L'])

        # Check termination: best point achieved target order?
        best_order = max([p['order'] for p in live_points])

        if best_order >= order_target:
            # Success! Record final state
            return {
                'success': True,
                'samples_used': sample_count,
                'final_order': float(best_order),
                'iterations': iteration,
                'n_live_adjustments': n_live_adjustments,
                'eff_dim_history': eff_dim_history,
                'n_live_final': n_live
            }

        # ====================================================================
        # ADAPTIVE N_LIVE LOGIC (every check_interval samples)
        # ====================================================================
        check_counter += 1

        if check_counter >= check_interval and len(weight_samples) >= 5:
            # Measure current effective dimensionality
            weight_array = np.array(weight_samples[-min(20, len(weight_samples)):])  # Last 20 samples
            eff_dim_metrics = compute_effective_dimensionality_from_samples(weight_array)
            current_eff_dim = eff_dim_metrics['effective_dim']

            if not np.isnan(current_eff_dim):
                eff_dim_history.append({
                    'iteration': iteration,
                    'eff_dim': current_eff_dim,
                    'n_live': n_live,
                    'best_order': float(best_order)
                })

                # Decision logic
                old_n_live = n_live

                if last_eff_dim is not None:
                    # Check for sharp collapse (>20% drop)
                    collapse_rate = (last_eff_dim - current_eff_dim) / (last_eff_dim + 1e-6)
                    if collapse_rate > 0.20 and current_eff_dim < 3.0:
                        # Sharp collapse detected - increase exploration
                        n_live = int(n_live * 1.4)
                        n_live_adjustments.append({
                            'iteration': iteration,
                            'reason': 'sharp_collapse',
                            'old_n_live': old_n_live,
                            'new_n_live': n_live,
                            'eff_dim': current_eff_dim
                        })

                # Absolute level logic
                if current_eff_dim <= 2.0:
                    # Collapsing - exploit manifold
                    if n_live < 150:
                        n_live = int(min(n_live * 1.3, 150))
                        n_live_adjustments.append({
                            'iteration': iteration,
                            'reason': 'low_eff_dim',
                            'old_n_live': old_n_live,
                            'new_n_live': n_live,
                            'eff_dim': current_eff_dim
                        })
                elif current_eff_dim > 3.5:
                    # High-D exploration - keep searching
                    if n_live > 80:
                        n_live = max(int(n_live * 0.9), 80)
                        n_live_adjustments.append({
                            'iteration': iteration,
                            'reason': 'high_eff_dim',
                            'old_n_live': old_n_live,
                            'new_n_live': n_live,
                            'eff_dim': current_eff_dim
                        })

                last_eff_dim = current_eff_dim

            check_counter = 0

        # ====================================================================
        # STANDARD NESTED SAMPLING STEP
        # ====================================================================

        # Remove worst point
        worst_idx = 0
        worst_point = live_points.pop(worst_idx)
        dead_points.append(worst_point)

        # Replace with new sample from prior (simplified: uniform sampling)
        # In realistic version, would use MCMC in constrained volume
        new_cppn = CPPN()
        img = new_cppn.render(image_size)
        order = order_multiplicative(img)
        log_L = log_prior(new_cppn)

        if not np.isfinite(log_L):
            log_L = -1e10

        live_points.append({
            'log_L': log_L,
            'order': order,
            'weights': new_cppn.get_weights().flatten(),
            'cppn': new_cppn
        })
        weight_samples.append(live_points[-1]['weights'])

        iteration += 1
        sample_count += 1

    # Max iterations exceeded
    best_order = max([p['order'] for p in live_points + dead_points])
    return {
        'success': False,
        'reason': 'Max iterations exceeded',
        'samples_used': sample_count,
        'final_order': float(best_order),
        'iterations': iteration,
        'n_live_adjustments': n_live_adjustments,
        'eff_dim_history': eff_dim_history,
        'n_live_final': n_live
    }


def nested_sample_baseline(
    cppn: CPPN,
    order_target: float = 0.5,
    n_live: int = 100,
    max_iterations: int = 300,
    image_size: int = 32,
    seed: int = None
) -> Dict:
    """
    Baseline: fixed n_live nested sampling (no adaptation).
    """
    if seed is not None:
        set_global_seed(seed)

    live_points = []
    sample_count = 0
    iteration = 0

    # Initialize
    while len(live_points) < n_live and iteration < max_iterations:
        cppn_candidate = CPPN()
        img = cppn_candidate.render(image_size)
        order = order_multiplicative(img)
        log_L = log_prior(cppn_candidate)

        if not np.isfinite(log_L):
            log_L = -1e10

        live_points.append({
            'log_L': log_L,
            'order': order,
            'weights': cppn_candidate.get_weights().flatten()
        })
        iteration += 1
        sample_count += 1

    if len(live_points) < n_live:
        return {
            'success': False,
            'reason': 'Failed to initialize',
            'samples_used': sample_count,
            'final_order': 0.0
        }

    dead_points = []

    # Sampling loop
    while iteration < max_iterations:
        live_points.sort(key=lambda x: x['log_L'])
        best_order = max([p['order'] for p in live_points])

        if best_order >= order_target:
            return {
                'success': True,
                'samples_used': sample_count,
                'final_order': float(best_order),
                'iterations': iteration
            }

        worst_idx = 0
        worst_point = live_points.pop(worst_idx)
        dead_points.append(worst_point)

        new_cppn = CPPN()
        img = new_cppn.render(image_size)
        order = order_multiplicative(img)
        log_L = log_prior(new_cppn)

        if not np.isfinite(log_L):
            log_L = -1e10

        live_points.append({
            'log_L': log_L,
            'order': order,
            'weights': new_cppn.get_weights().flatten()
        })

        iteration += 1
        sample_count += 1

    best_order = max([p['order'] for p in live_points + dead_points])
    return {
        'success': False,
        'reason': 'Max iterations exceeded',
        'samples_used': sample_count,
        'final_order': float(best_order),
        'iterations': iteration
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("RES-228: Order-Predictive Sampling - Adaptive n_live via Eff_Dim Feedback")
    print("=" * 80)

    set_global_seed(42)

    N_CPPNS = 15  # Test on 15 CPPNs for faster iteration
    IMAGE_SIZE = 32
    ORDER_TARGET = 0.5

    print(f"\nEXPERIMENT CONFIG:")
    print(f"  N_CPPNS: {N_CPPNS}")
    print(f"  Target order: {ORDER_TARGET}")
    print(f"  Initial n_live (both methods): 100")
    print(f"  Check interval: 20 samples")

    # ========================================================================
    # BASELINE: Fixed n_live sampling
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE - Fixed n_live=100 sampling")
    print("=" * 80)

    baseline_results = []
    baseline_samples_total = 0
    baseline_successes = 0

    for i in range(N_CPPNS):
        if i % 5 == 0:
            print(f"  Running {i}/{N_CPPNS}...")

        cppn = CPPN()
        result = nested_sample_baseline(
            cppn,
            order_target=ORDER_TARGET,
            n_live=100,
            seed=42 + i
        )

        baseline_results.append(result)
        baseline_samples_total += result['samples_used']
        if result['success']:
            baseline_successes += 1

    baseline_avg_samples = baseline_samples_total / N_CPPNS
    baseline_success_rate = baseline_successes / N_CPPNS

    print(f"\nBASELINE RESULTS:")
    print(f"  Avg samples: {baseline_avg_samples:.1f}")
    print(f"  Success rate: {baseline_success_rate:.1%} ({baseline_successes}/{N_CPPNS})")

    # ========================================================================
    # ADAPTIVE: Eff_dim-guided n_live
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: ADAPTIVE - Eff_dim-guided n_live adjustments")
    print("=" * 80)

    adaptive_results = []
    adaptive_samples_total = 0
    adaptive_successes = 0

    for i in range(N_CPPNS):
        if i % 5 == 0:
            print(f"  Running {i}/{N_CPPNS}...")

        cppn = CPPN()
        result = nested_sample_with_adaptive_n_live(
            cppn,
            order_target=ORDER_TARGET,
            initial_n_live=100,
            check_interval=20,
            seed=42 + i
        )

        adaptive_results.append(result)
        adaptive_samples_total += result['samples_used']
        if result['success']:
            adaptive_successes += 1

    adaptive_avg_samples = adaptive_samples_total / N_CPPNS
    adaptive_success_rate = adaptive_successes / N_CPPNS

    print(f"\nADAPTIVE RESULTS:")
    print(f"  Avg samples: {adaptive_avg_samples:.1f}")
    print(f"  Success rate: {adaptive_success_rate:.1%} ({adaptive_successes}/{N_CPPNS})")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS & METRICS")
    print("=" * 80)

    speedup = baseline_avg_samples / adaptive_avg_samples if adaptive_avg_samples > 0 else 0
    success_delta = adaptive_success_rate - baseline_success_rate

    print(f"\nSPEEDUP: {speedup:.2f}x")
    print(f"  Baseline avg: {baseline_avg_samples:.1f} samples")
    print(f"  Adaptive avg: {adaptive_avg_samples:.1f} samples")

    print(f"\nSUCCESS RATE COMPARISON:")
    print(f"  Baseline: {baseline_success_rate:.1%}")
    print(f"  Adaptive: {adaptive_success_rate:.1%}")
    print(f"  Delta: {success_delta:+.1%}")

    # Validation criteria
    print(f"\nVALIDATION:")
    speedup_ok = speedup >= 2.0
    success_ok = adaptive_success_rate >= 0.90

    print(f"  Speedup ≥ 2.0×? {speedup_ok} (actual: {speedup:.2f}x)")
    print(f"  Success ≥ 90%? {success_ok} (actual: {adaptive_success_rate:.1%})")

    conclusion = "VALIDATED" if (speedup_ok and success_ok) else "REFUTED"
    print(f"\nCONCLUSION: {conclusion}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/predictive_manifold_sampling')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'hypothesis': 'Using eff_dim during sampling guides adaptive n_live for ≥2× speedup',
        'method': 'Eff_dim-guided adaptive n_live during nested sampling',
        'n_cppns': N_CPPNS,
        'target_order': ORDER_TARGET,
        'baseline': {
            'avg_samples': float(baseline_avg_samples),
            'success_rate': float(baseline_success_rate),
            'successes': int(baseline_successes),
            'total': int(N_CPPNS),
            'detailed_results': [
                {
                    'cppn_id': i,
                    'samples_used': r['samples_used'],
                    'success': r['success'],
                    'final_order': r['final_order']
                }
                for i, r in enumerate(baseline_results)
            ]
        },
        'adaptive': {
            'avg_samples': float(adaptive_avg_samples),
            'success_rate': float(adaptive_success_rate),
            'successes': int(adaptive_successes),
            'total': int(N_CPPNS),
            'detailed_results': [
                {
                    'cppn_id': i,
                    'samples_used': r['samples_used'],
                    'success': r['success'],
                    'final_order': r['final_order'],
                    'n_live_adjustments': r.get('n_live_adjustments', []),
                    'eff_dim_history': r.get('eff_dim_history', [])
                }
                for i, r in enumerate(adaptive_results)
            ]
        },
        'metrics': {
            'speedup': float(speedup),
            'success_rate_delta': float(success_delta),
            'speedup_threshold_met': bool(speedup_ok),
            'success_threshold_met': bool(success_ok)
        },
        'conclusion': conclusion
    }

    results_file = results_dir / 'res_228_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    return results


if __name__ == '__main__':
    main()
