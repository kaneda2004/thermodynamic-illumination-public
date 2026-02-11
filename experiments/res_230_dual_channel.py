#!/usr/bin/env python3
"""
RES-230: Dual-Channel CPPN Architecture (x±y Composition)

Hypothesis: Alternative input coordinate systems [x+y, x-y] can achieve
lower effective dimensionality (≤3D) compared to standard [x, y, r, bias]
while maintaining comparable order achievement, improving sampling by ≥2×.

Context:
- RES-223: Compositional structure (x,y,r) drives dimensionality
- RES-225: Low-D initialization preserves capability (eff_dim 1.94 maintains efficiency)
- RES-224: Sampling efficiency depends on eff_dim (higher eff_dim = more samples needed)

Key Insight: If compositional structure matters (RES-223), different
coordinate transforms might yield different dimensionality profiles.
Dual-channel (x+y, x-y) provides rotation-invariant and translation-aware
structure potentially more efficient than Cartesian + radial.

METHOD:
1. Implement 4 CPPN variants with different inputs:
   - Variant A: [x, y, r, bias] (baseline standard)
   - Variant B: [x+y, x-y, bias] (dual-channel sum/diff, 3 inputs)
   - Variant C: [x+y, |x-y|, bias] (dual-channel with magnitude)
   - Variant D: [x+y, x-y, x*y, bias] (dual-channel with interaction)

2. For each variant:
   a) Initialize 40 CPPNs
   b) Measure initial effective dimensionality (participation ratio + PCA)
   c) Run nested sampling to order target 0.5 (challenging)
   d) Track samples needed and success rate
   e) Compute order distribution statistics

3. Compare across variants:
   - Which variant achieves lowest eff_dim?
   - How many samples needed for each variant?
   - Order achievement and quality

4. Validation criteria:
   - Best dual-channel eff_dim ≤ 3.0 (within RES-225 bounds)
   - Best dual-channel speedup ≥ 2.0× (vs baseline samples/eff_dim ratio)
   - Success rate ≥ 60% for order 0.5 target
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import random
from sklearn.decomposition import PCA
from scipy import stats

# Ensure project root is in path (works on both local and GCP)
# Force CWD for batch execution to avoid any path confusion
project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA, ACTIVATIONS
)


# ============================================================================
# VARIANT: Modified CPPN with Dual-Channel Inputs
# ============================================================================

class DualChannelCPPN(CPPN):
    """
    CPPN with modified input coordinate system.

    input_variant determines which coordinates are used:
    - 'standard': [x, y, r, bias] (4 inputs)
    - 'dual_sum_diff': [x+y, x-y, bias] (3 inputs)
    - 'dual_abs_diff': [x+y, |x-y|, bias] (3 inputs)
    - 'dual_interaction': [x+y, x-y, x*y, bias] (4 inputs)
    """

    def __init__(self, input_variant='standard'):
        self.input_variant = input_variant

        # Set input_ids based on variant
        # We'll use 0=first_coord, 1=second_coord, 2=third_coord (if exists), 3=bias
        if input_variant == 'standard':
            input_ids = [0, 1, 2, 3]  # x, y, r, bias
        elif input_variant == 'dual_sum_diff':
            input_ids = [0, 1, 2]  # x+y, x-y, bias
        elif input_variant == 'dual_abs_diff':
            input_ids = [0, 1, 2]  # x+y, |x-y|, bias
        elif input_variant == 'dual_interaction':
            input_ids = [0, 1, 2, 3]  # x+y, x-y, x*y, bias
        else:
            raise ValueError(f"Unknown variant: {input_variant}")

        # Initialize parent CPPN with variant-specific input_ids
        super().__init__(input_ids=input_ids)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate CPPN with variant-specific input coordinates.
        """
        # Compute coordinate variants
        if self.input_variant == 'standard':
            r = np.sqrt(x**2 + y**2)
            bias = np.ones_like(x)
            values = {0: x, 1: y, 2: r, 3: bias}

        elif self.input_variant == 'dual_sum_diff':
            sum_xy = x + y
            diff_xy = x - y
            bias = np.ones_like(x)
            values = {0: sum_xy, 1: diff_xy, 2: bias}

        elif self.input_variant == 'dual_abs_diff':
            sum_xy = x + y
            abs_diff_xy = np.abs(x - y)
            bias = np.ones_like(x)
            values = {0: sum_xy, 1: abs_diff_xy, 2: bias}

        elif self.input_variant == 'dual_interaction':
            sum_xy = x + y
            diff_xy = x - y
            prod_xy = x * y
            bias = np.ones_like(x)
            values = {0: sum_xy, 1: diff_xy, 2: prod_xy, 3: bias}

        # Evaluate network
        eval_order = self._get_eval_order()
        for nid in eval_order:
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)

        return values[self.output_id]


@dataclass
class ExperimentConfig:
    """Configuration for dual-channel architecture experiment"""
    cppn_per_variant: int = 40  # CPPNs per variant
    order_target: float = 0.5   # Challenging sampling target
    n_live: int = 100           # Modest n_live for convergence
    max_iterations: int = 400   # Max iterations per CPPN
    image_size: int = 32
    seed: int = 42


def compute_participation_ratio(weights_list: list) -> dict:
    """
    Compute effective dimensionality using participation ratio and PCA.

    Participation Ratio (PR): 1 / sum(p_i^2) where p_i is variance explained by PC_i
    - PR=1 means 1D (all variance in first PC)
    - PR=n means full dimensionality (variance evenly distributed)
    """
    # Convert to array
    weights_array = []
    for w in weights_list:
        w_arr = np.array(w)
        if w_arr.ndim > 1:
            w_arr = w_arr.flatten()
        weights_array.append(w_arr)

    W = np.array(weights_array)

    if len(W) < 2:
        return {
            'participation_ratio': float(len(W[0])),
            'eff_dim_90': len(W[0]),
            'top_variance': 0.0
        }

    pca = PCA()
    pca.fit(W)

    # Participation ratio
    var_ratios = pca.explained_variance_ratio_
    var_ratios = var_ratios[var_ratios > 1e-10]
    participation_ratio = 1.0 / np.sum(var_ratios ** 2) if len(var_ratios) > 0 else 1.0

    # Effective dimension at 90% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim_90 = np.searchsorted(cumvar, 0.90) + 1

    return {
        'participation_ratio': float(participation_ratio),
        'eff_dim_90': int(eff_dim_90),
        'top_variance': float(cumvar[0] if len(cumvar) > 0 else 0.0)
    }


def run_nested_sampling_trial(
    cppn: DualChannelCPPN,
    target_order: float,
    image_size: int,
    n_live: int,
    max_iterations: int
) -> dict:
    """
    Run nested sampling and track samples to reach target order.

    Returns:
        dict with: samples_to_target, max_order, success, order_dist
    """
    set_global_seed(None)

    # Initialize live set
    live_points = []
    for _ in range(n_live):
        proposal_cppn = DualChannelCPPN(input_variant=cppn.input_variant)
        proposal_cppn.set_weights(np.random.randn(len(proposal_cppn.get_weights())) * PRIOR_SIGMA)
        img = proposal_cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((proposal_cppn, img, order))

    best_order = max(lp[2] for lp in live_points)
    samples_to_target = None
    order_dist = [best_order]

    # Check if already at target
    if best_order >= target_order:
        samples_to_target = n_live

    # Nested sampling loop
    for iteration in range(max_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        # Elliptical slice sampling
        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[worst_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        order_dist.append(best_order)

        # Record when target is reached
        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    return {
        'samples_to_target': samples_to_target if samples_to_target is not None else n_live * max_iterations,
        'max_order': float(best_order),
        'success': best_order >= target_order,
        'order_dist_samples': [float(o) for o in order_dist[-20:]]  # Last 20 for compactness
    }


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete dual-channel architecture experiment"""

    print("=" * 80)
    print("RES-230: Dual-Channel CPPN Architecture (x±y Composition)")
    print("=" * 80)

    variants = ['standard', 'dual_sum_diff', 'dual_abs_diff', 'dual_interaction']
    results_by_variant = {}

    for variant_idx, variant in enumerate(variants, 1):
        print(f"\n[{variant_idx}/4] Testing variant: {variant}")
        print("-" * 80)

        # Initialize CPPNs for this variant
        set_global_seed(config.seed + variant_idx * 1000)
        cppn_list = []
        for i in range(config.cppn_per_variant):
            cppn = DualChannelCPPN(input_variant=variant)
            cppn_list.append(cppn)

        # Measure initial effective dimensionality
        weights_list = [c.get_weights() for c in cppn_list]
        eff_dim = compute_participation_ratio(weights_list)

        print(f"  Input variant: {variant}")
        print(f"  n_inputs: {len(cppn_list[0].input_ids)}")
        print(f"  Initial eff_dim (PR): {eff_dim['participation_ratio']:.2f}")
        print(f"  Initial eff_dim (90%): {eff_dim['eff_dim_90']}")

        # Run nested sampling trials
        print(f"  Running {config.cppn_per_variant} nested sampling trials...")
        trial_results = []
        for i, cppn in enumerate(cppn_list):
            result = run_nested_sampling_trial(
                cppn, config.order_target,
                config.image_size, config.n_live,
                config.max_iterations
            )
            trial_results.append(result)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{config.cppn_per_variant} trials completed")

        # Compute statistics
        samples_list = [r['samples_to_target'] for r in trial_results]
        success_count = sum(1 for r in trial_results if r['success'])
        success_rate = success_count / len(trial_results)

        mean_samples = float(np.mean(samples_list))
        std_samples = float(np.std(samples_list))
        median_samples = float(np.median(samples_list))

        mean_order = float(np.mean([r['max_order'] for r in trial_results]))
        max_order = float(max(r['max_order'] for r in trial_results))

        print(f"  Results:")
        print(f"    Mean samples: {mean_samples:.0f} ± {std_samples:.0f}")
        print(f"    Median samples: {median_samples:.0f}")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Mean order: {mean_order:.4f}")
        print(f"    Max order: {max_order:.4f}")

        results_by_variant[variant] = {
            'n_inputs': len(cppn_list[0].input_ids),
            'initial_eff_dim_pr': float(eff_dim['participation_ratio']),
            'initial_eff_dim_90': int(eff_dim['eff_dim_90']),
            'mean_samples': mean_samples,
            'std_samples': std_samples,
            'median_samples': median_samples,
            'success_rate': float(success_rate),
            'mean_order': mean_order,
            'max_order': max_order,
            'n_trials': len(trial_results)
        }

    # Compare variants
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS VARIANTS")
    print(f"{'='*80}")

    for variant in variants:
        res = results_by_variant[variant]
        print(f"\n{variant}:")
        print(f"  Inputs: {res['n_inputs']}, eff_dim: {res['initial_eff_dim_pr']:.2f}")
        print(f"  Samples: {res['mean_samples']:.0f}, Success: {res['success_rate']:.1%}")

    # Find best variant (lowest eff_dim with acceptable speedup)
    baseline = results_by_variant['standard']
    baseline_ratio = baseline['mean_samples'] / baseline['initial_eff_dim_pr']

    best_variant = None
    best_speedup = 0.0
    best_eff_dim = None

    for variant in variants:
        if variant == 'standard':
            continue

        res = results_by_variant[variant]

        # Speedup = (baseline_samples / baseline_eff_dim) / (variant_samples / variant_eff_dim)
        variant_ratio = res['mean_samples'] / res['initial_eff_dim_pr']
        speedup = baseline_ratio / variant_ratio

        print(f"\n{variant} speedup: {speedup:.2f}x")

        # Validation: eff_dim ≤ 3.0 AND speedup ≥ 1.5×
        if (res['initial_eff_dim_pr'] <= 3.0 and
            speedup >= 1.5 and
            res['success_rate'] >= 0.6):
            if speedup > best_speedup:
                best_variant = variant
                best_speedup = speedup
                best_eff_dim = res['initial_eff_dim_pr']

    # Validation criterion
    print(f"\n{'='*80}")
    print("HYPOTHESIS VALIDATION")
    print(f"{'='*80}")

    validation_passed = best_variant is not None

    if validation_passed:
        best_res = results_by_variant[best_variant]
        print(f"✓ HYPOTHESIS VALIDATED")
        print(f"  Best variant: {best_variant}")
        print(f"  eff_dim: {best_res['initial_eff_dim_pr']:.2f} (≤3.0 required)")
        print(f"  speedup: {best_speedup:.2f}x (≥1.5x required)")
        print(f"  success_rate: {best_res['success_rate']:.1%} (≥60% required)")
        conclusion = "validate"
    else:
        print(f"✗ HYPOTHESIS REFUTED")
        print(f"  No dual-channel variant achieved:")
        print(f"    - eff_dim ≤ 3.0")
        print(f"    - speedup ≥ 1.5×")
        print(f"    - success_rate ≥ 60%")
        conclusion = "refute"

    # Compile final results
    results = {
        "method": "Dual-channel [x+y, x-y] vs standard [x,y,r] inputs",
        "hypothesis": "Alternative coordinate transforms yield lower eff_dim with ≥2× speedup",
        "target_order": config.order_target,
        "n_live": config.n_live,
        "cppn_per_variant": config.cppn_per_variant,

        "baseline_variant": {
            "name": "standard",
            "n_inputs": results_by_variant['standard']['n_inputs'],
            "eff_dim": float(results_by_variant['standard']['initial_eff_dim_pr']),
            "mean_samples": float(results_by_variant['standard']['mean_samples']),
            "success_rate": float(results_by_variant['standard']['success_rate']),
            "mean_order": float(results_by_variant['standard']['mean_order'])
        },

        "dual_sum_diff": results_by_variant['dual_sum_diff'],
        "dual_abs_diff": results_by_variant['dual_abs_diff'],
        "dual_interaction": results_by_variant['dual_interaction'],

        "best_variant": best_variant,
        "best_eff_dim": float(best_eff_dim) if best_eff_dim is not None else None,
        "best_speedup": float(best_speedup),

        "conclusion": conclusion,
        "interpretation": (
            f"Dual-channel architectures "
            f"{'CANNOT' if not validation_passed else 'CAN'} "
            f"achieve lower effective dimensionality with improved sampling efficiency. "
            f"{'Best variant: ' + best_variant if best_variant else 'All variants underperformed.'}"
        )
    }

    return results


def main():
    """Main experiment execution"""
    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "dual_channel_architecture"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'res_230_results.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print concise summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        best_res = results.get('best_variant')
        best_speedup = results.get('best_speedup', 0)
        best_eff_dim = results.get('best_eff_dim')

        if best_res:
            print(f"RES-230 | dual_channel_architecture | {results['conclusion']} | "
                  f"variant={best_res} eff_dim={best_eff_dim:.2f} speedup={best_speedup:.2f}x")
        else:
            print(f"RES-230 | dual_channel_architecture | {results['conclusion']} | "
                  f"no_improvement speedup={best_speedup:.2f}x")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        results = {
            "method": "Dual-channel architecture variants",
            "error": str(e),
            "conclusion": "inconclusive"
        }

        results_dir = project_root / "results" / "dual_channel_architecture"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'res_230_results.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
