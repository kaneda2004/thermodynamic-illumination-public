#!/usr/bin/env python3
"""
RES-234: Adaptive Threshold Manifold Discovery

Hypothesis: Dynamically switching to manifold constraint when PCA variance_explained
exceeds a threshold achieves ≥100× speedup with lower variance than fixed-stage approach.

Method:
1. Implement adaptive switching strategy:
   - Monitor PCA variance explained every 25 samples during Stage 1
   - When variance exceeds threshold, switch to Stage 2 (constrained sampling)
   - Test 4 thresholds: 70%, 80%, 90%, plus 100% (always constrain)
2. Run 25 fresh CPPNs to order 0.5 with each threshold
3. Measure: (a) average switch point, (b) total samples, (c) variance across trials
4. Compare vs RES-224 fixed N=150 baseline (276 samples)
5. Validate: adaptive ≥ 100× speedup AND σ reduction ≥ 20%

Key insight: Rather than fixed Stage 1 budget, adaptively switch when the manifold
becomes sufficiently informative (high variance explained). This should balance
exploration thoroughness with efficiency.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import random

# Ensure project root is in path
project_root = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(project_root))

# Set working directory
os.chdir(project_root)

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA
)

@dataclass
class ExperimentConfig:
    """Configuration for adaptive threshold sampling experiment"""
    test_cppns: int = 25                    # CPPNs to test (25 fresh)
    order_target: float = 0.50              # Target order level
    pca_components: int = 3                 # Manifold dimensionality
    image_size: int = 32

    # Adaptive parameters
    variance_check_interval: int = 50       # Check PCA variance every 50 samples
    min_samples_stage1: int = 100           # Minimum samples before considering switch
    max_samples_stage1: int = 300           # Maximum Stage 1 exploration
    max_iterations_stage2: int = 250        # Stage 2 iterations after switch

    # Thresholds to test (variance_explained thresholds for switching)
    variance_thresholds: list = None         # [0.70, 0.80, 0.90, 1.00]

    # Baseline from RES-224
    baseline_samples: float = 276.0

    seed: int = 42

    def __post_init__(self):
        if self.variance_thresholds is None:
            self.variance_thresholds = [0.70, 0.80, 0.90, 1.00]


def generate_test_cppns(n_samples: int, seed: int = 42) -> list:
    """Generate N random CPPNs for testing."""
    print(f"[1/6] Generating {n_samples} test CPPNs...")
    set_global_seed(seed)
    test_cppns = [CPPN() for _ in range(n_samples)]
    print(f"✓ Generated {len(test_cppns)} test CPPNs")
    return test_cppns


def compute_pca_basis_from_samples(weights_samples: list, n_components: int = 3) -> tuple:
    """
    Compute PCA basis from collected weight samples.

    Returns:
        tuple: (mean_weights, principal_components, explained_variance)
    """
    if len(weights_samples) < 2:
        return None, None, 0.0

    W = np.array(weights_samples)

    # Center
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Principal components (top n_components)
    n_comp = min(n_components, len(S))
    components = Vt[:n_comp]

    # Variance explained
    if len(S) > 0:
        explained_var = (S[:n_comp] ** 2).sum() / (S ** 2).sum()
    else:
        explained_var = 0.0

    return W_mean, components, explained_var


def project_to_pca(weights: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Project weights to PCA basis."""
    if pca_mean is None or pca_components is None:
        return None
    w_centered = weights - pca_mean
    coeffs = pca_components @ w_centered
    return coeffs


def reconstruct_from_pca(coeffs: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reconstruct weights from PCA coefficients."""
    w_pca_space = pca_components.T @ coeffs
    return pca_mean + w_pca_space


def run_adaptive_threshold_sampling(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    variance_threshold: float,
    variance_check_interval: int,
    min_samples_stage1: int,
    max_samples_stage1: int,
    max_iterations_stage2: int,
    pca_components: int = 3
) -> dict:
    """
    Run adaptive threshold sampling:
    - Stage 1: Collect samples, monitor PCA variance every check_interval samples
    - When variance_explained >= threshold, switch to Stage 2
    - Stage 2: Constrained sampling on learned manifold

    Returns metrics including:
      - switch_point: sample count at which we switched
      - total_samples: final total samples consumed
      - variance_at_switch: variance explained at switch time
    """
    set_global_seed(None)

    # ===== STAGE 1: Exploration with variance monitoring =====
    n_live_stage1 = 50  # Fixed exploration live set size
    live_points = []
    best_order = 0
    collected_weights = []
    collected_orders = []
    samples_at_target = None
    stage1_samples_collected = 0
    switch_point = None
    variance_at_switch = 0.0

    # Initialize
    for _ in range(n_live_stage1):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        collected_orders.append(order)
        best_order = max(best_order, order)
        stage1_samples_collected = n_live_stage1

    if best_order >= target_order:
        samples_at_target = stage1_samples_collected

    # Track when we last checked variance
    last_variance_check = stage1_samples_collected

    # Stage 1 exploration with adaptive switching
    switched = False
    max_iterations_stage1 = max_samples_stage1 // n_live_stage1

    for iteration in range(max_iterations_stage1):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_stage1)

        # Standard ESS in Stage 1
        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            collected_orders.append(proposal_order)
            stage1_samples_collected += 1

        # Check if we should switch (every variance_check_interval samples)
        if (not switched and
            stage1_samples_collected >= min_samples_stage1 and
            stage1_samples_collected - last_variance_check >= variance_check_interval):

            # Compute current PCA variance
            _, _, current_var = compute_pca_basis_from_samples(
                collected_weights, pca_components
            )
            last_variance_check = stage1_samples_collected

            # Check if variance meets threshold
            if current_var >= variance_threshold:
                switched = True
                switch_point = stage1_samples_collected
                variance_at_switch = current_var
                # Break Stage 1, move to Stage 2
                break

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples_collected

    # If never switched (variance never reached threshold), use all Stage 1 samples
    if not switched:
        switch_point = stage1_samples_collected
        _, _, variance_at_switch = compute_pca_basis_from_samples(
            collected_weights, pca_components
        )

    # ===== STAGE 2: Manifold convergence (constrained sampling on learned manifold) =====
    # Learn manifold from Stage 1 samples
    pca_mean, pca_components_learned, final_var = compute_pca_basis_from_samples(
        collected_weights, pca_components
    )

    # Continue with constrained sampling in Stage 2
    total_samples = stage1_samples_collected

    if pca_mean is not None and pca_components_learned is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            threshold = worst_order

            seed_idx = np.random.randint(0, n_live_stage1)

            # Constrained proposal on learned manifold
            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, pca_mean, pca_components_learned)

            if coeffs is not None:
                # Perturb coefficients
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                # Reconstruct
                proposal_w = reconstruct_from_pca(new_coeffs, pca_mean, pca_components_learned)
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                # Accept if above threshold
                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)

                total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'variance_threshold': variance_threshold,
        'switch_point': switch_point,
        'variance_at_switch': float(variance_at_switch),
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target
    }


def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete adaptive threshold sampling experiment"""

    # Generate test CPPNs
    test_cppns = generate_test_cppns(config.test_cppns, config.seed)

    # Run adaptive threshold variants
    print(f"\n[2/6] Running {len(config.variance_thresholds)} adaptive threshold variants...")
    threshold_results = {}

    for threshold in config.variance_thresholds:
        print(f"\n  Testing variance_threshold={threshold:.2f}...")
        variant_samples = []
        variant_switch_points = []
        variant_variances = []

        for i, cppn in enumerate(test_cppns):
            result = run_adaptive_threshold_sampling(
                cppn,
                config.order_target,
                config.image_size,
                threshold,
                config.variance_check_interval,
                config.min_samples_stage1,
                config.max_samples_stage1,
                config.max_iterations_stage2,
                config.pca_components
            )
            variant_samples.append(result['samples_to_target'])
            variant_switch_points.append(result['switch_point'])
            variant_variances.append(result['variance_at_switch'])

            if (i + 1) % 5 == 0:
                print(f"    [{i+1}/{config.test_cppns}] total={result['samples_to_target']:.0f}, switch={result['switch_point']:.0f}")

        avg_samples = np.mean(variant_samples)
        std_samples = np.std(variant_samples)
        avg_switch = np.mean(variant_switch_points)
        avg_variance = np.mean(variant_variances)
        speedup = config.baseline_samples / avg_samples if avg_samples > 0 else 0

        threshold_results[f"threshold_{threshold:.2f}"] = {
            'avg_samples': float(avg_samples),
            'std_samples': float(std_samples),
            'avg_switch_point': float(avg_switch),
            'avg_variance_at_switch': float(avg_variance),
            'speedup': float(speedup)
        }

        print(f"  ✓ threshold={threshold:.2f}: {avg_samples:.0f} ± {std_samples:.0f} samples, speedup={speedup:.2f}×, switch={avg_switch:.0f}")

    # Find best variant
    print(f"\n[3/6] Analyzing results...")
    best_speedup = max(v['speedup'] for v in threshold_results.values())
    best_threshold = [k.split('_')[-1] for k, v in threshold_results.items() if v['speedup'] == best_speedup][0]

    print(f"Best speedup: {best_speedup:.2f}× (threshold={best_threshold})")

    # Validate hypothesis: best >= 100× speedup
    print(f"\n[4/6] Validating hypothesis...")
    speedup_threshold = 100.0  # Require 100× speedup
    variance_reduction_threshold = 0.20  # Require 20% variance reduction

    # Calculate variance reduction (lower is better for stability)
    best_std = threshold_results[f"threshold_{best_threshold}"]['std_samples']
    baseline_std = np.std([config.baseline_samples] * config.test_cppns)  # Single estimate
    variance_reduction = max(0, (baseline_std - best_std) / baseline_std) if baseline_std > 0 else 0

    validated_speedup = best_speedup >= speedup_threshold
    validated_variance = variance_reduction >= variance_reduction_threshold

    print(f"Speedup threshold: >= {speedup_threshold:.0f}×, achieved: {best_speedup:.2f}×")
    print(f"Variance reduction threshold: >= {variance_reduction_threshold:.0%}, achieved: {variance_reduction:.2f}%")

    # Hypothesis is validated if both speedup and variance criteria met
    validated = validated_speedup and validated_variance
    conclusion = "validate" if validated else "refute"

    print(f"Conclusion: {conclusion}")

    # Compile results
    results = {
        "method": "Adaptive switching on PCA variance threshold",
        "baseline_samples": float(config.baseline_samples),
        "test_cppns": config.test_cppns,
        "variance_thresholds_tested": config.variance_thresholds,
        "threshold_70_avg_samples": threshold_results.get("threshold_0.70", {}).get("avg_samples", 0),
        "threshold_70_sigma": threshold_results.get("threshold_0.70", {}).get("std_samples", 0),
        "threshold_70_speedup": threshold_results.get("threshold_0.70", {}).get("speedup", 0),
        "threshold_80_avg_samples": threshold_results.get("threshold_0.80", {}).get("avg_samples", 0),
        "threshold_80_sigma": threshold_results.get("threshold_0.80", {}).get("std_samples", 0),
        "threshold_80_speedup": threshold_results.get("threshold_0.80", {}).get("speedup", 0),
        "threshold_90_avg_samples": threshold_results.get("threshold_0.90", {}).get("avg_samples", 0),
        "threshold_90_sigma": threshold_results.get("threshold_0.90", {}).get("std_samples", 0),
        "threshold_90_speedup": threshold_results.get("threshold_0.90", {}).get("speedup", 0),
        "threshold_100_avg_samples": threshold_results.get("threshold_1.00", {}).get("avg_samples", 0),
        "threshold_100_sigma": threshold_results.get("threshold_1.00", {}).get("std_samples", 0),
        "threshold_100_speedup": threshold_results.get("threshold_1.00", {}).get("speedup", 0),
        "best_speedup": float(best_speedup),
        "best_threshold": float(best_threshold),
        "variance_reduction": float(variance_reduction),
        "speedup_threshold": speedup_threshold,
        "variance_reduction_threshold": variance_reduction_threshold,
        "conclusion": conclusion
    }

    return results


def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-234: Adaptive Threshold Manifold Discovery")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "adaptive_threshold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_234_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"RES-234 | adaptive_threshold_sampling | {results['conclusion']} | speedup={results['best_speedup']:.2f}x variance_reduction={results['variance_reduction']:.1%}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "method": "Adaptive switching on PCA variance threshold",
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "adaptive_threshold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_234_results.json", 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
