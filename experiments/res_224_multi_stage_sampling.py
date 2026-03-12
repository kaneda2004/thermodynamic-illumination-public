#!/usr/bin/env python3
"""
RES-224: Multi-Stage Sampling (Exploration → Manifold Convergence)

Hypothesis: Two-stage sampling (broad exploration to find manifold, then constrained
within manifold) achieves >= 3× speedup vs single-stage sampling.

Method:
1. Implement two-stage algorithm:
   - Stage 1: Run standard NS for N samples to discover high-order region
   - Stage 2: Compute manifold from Stage 1 samples, switch to constrained sampling
2. Test 4 variants with different N: N=50, N=100, N=150, N=200
3. Run each variant on 20 CPPNs to order 0.5
4. Compare: total samples (Stage 1 + Stage 2) vs baseline single-stage
5. Validate if: optimal two-stage >= 3× speedup

Key insight: Unlike RES-220 (fixed reference manifold), we adaptively discover the
manifold during Stage 1 of the run itself, so the manifold adapts to the specific
CPPN being sampled.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import random

# Ensure project root is in path (works on both local and GCP)
local_path = Path('/Users/matt/Development/monochrome_noise_converger')
if local_path.exists():
    project_root = local_path
else:
    # On GCP, use current working directory (should be ~/repo)
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA
)

@dataclass
class ExperimentConfig:
    """Configuration for two-stage sampling experiment"""
    test_cppns: int = 20                    # CPPNs to test
    order_target: float = 0.50              # Target order level
    baseline_n_live: int = 100              # Baseline single-stage n_live
    baseline_max_iterations: int = 300      # Baseline max iterations

    # Two-stage variant configs (N = exploration budget for Stage 1)
    stage1_budgets: list = None             # N values: [50, 100, 150, 200]

    pca_components: int = 3                 # Manifold dimensionality
    max_iterations_stage2: int = 250        # Stage 2 iterations after manifold learned
    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.stage1_budgets is None:
            self.stage1_budgets = [50, 100, 150, 200]

def generate_test_cppns(n_samples: int, seed: int = 42) -> list:
    """Generate N random CPPNs for testing."""
    print(f"[1/5] Generating {n_samples} test CPPNs...")
    set_global_seed(seed)
    test_cppns = [CPPN() for _ in range(n_samples)]
    print(f"✓ Generated {len(test_cppns)} test CPPNs")
    return test_cppns

def compute_pca_basis_from_samples(weights_samples: list, n_components: int = 3) -> tuple:
    """
    Compute PCA basis from collected weight samples.

    Args:
        weights_samples: List of weight matrices from Stage 1 sampling
        n_components: Number of principal components

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

def run_baseline_single_stage(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    n_live: int,
    max_iterations: int
) -> dict:
    """
    Run standard single-stage nested sampling to target order.
    Tracks total samples consumed.
    """
    set_global_seed(None)

    live_points = []
    best_order = 0
    samples_to_target = None

    # Initialize live set
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_to_target = n_live

    # Nested sampling loop
    for iteration in range(max_iterations):
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live)

        # Standard ESS proposal
        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        # Record when target reached
        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    if samples_to_target is None:
        samples_to_target = n_live * max_iterations

    return {
        'total_samples': samples_to_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }

def run_two_stage_sampling(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    pca_components: int = 3
) -> dict:
    """
    Run two-stage sampling:
    - Stage 1: Collect stage1_budget samples with standard NS
    - Stage 2: Learn manifold, do constrained NS

    Returns metrics including total samples to reach target.
    """
    set_global_seed(None)

    # ===== STAGE 1: Exploration (collect samples to learn manifold) =====
    n_live_stage1 = 50  # Fixed exploration live set size
    live_points = []
    best_order = 0
    collected_weights = []
    collected_orders = []
    samples_at_target = None

    # Initialize
    for _ in range(n_live_stage1):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        collected_orders.append(order)
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    # Stage 1 exploration iterations (collect samples for manifold)
    stage1_samples_collected = n_live_stage1
    for iteration in range(stage1_budget // n_live_stage1):
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

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples_collected

    # ===== STAGE 2: Manifold convergence (constrained sampling on learned manifold) =====
    # Learn manifold from Stage 1 samples
    pca_mean, pca_components_learned, explained_var = compute_pca_basis_from_samples(
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
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'pca_variance_explained': float(explained_var) if pca_mean is not None else 0.0,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target
    }

def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete two-stage sampling experiment"""

    # Generate test CPPNs
    test_cppns = generate_test_cppns(config.test_cppns, config.seed)

    # Run baseline (single-stage) on all test CPPNs
    print(f"\n[2/5] Running baseline single-stage sampling on {config.test_cppns} CPPNs...")
    baseline_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_baseline_single_stage(
            cppn,
            config.order_target,
            config.image_size,
            config.baseline_n_live,
            config.baseline_max_iterations
        )
        baseline_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] baseline samples: {result['total_samples']:.0f}")

    baseline_samples = [r['total_samples'] for r in baseline_results]
    avg_baseline = np.mean(baseline_samples)

    print(f"✓ Baseline: {avg_baseline:.0f} ± {np.std(baseline_samples):.0f} samples")

    # Run two-stage variants
    print(f"\n[3/5] Running {len(config.stage1_budgets)} two-stage variants...")
    variant_results = {}

    for budget_N in config.stage1_budgets:
        print(f"\n  Testing Stage1_N={budget_N}...")
        variant_samples = []

        for i, cppn in enumerate(test_cppns):
            result = run_two_stage_sampling(
                cppn,
                config.order_target,
                config.image_size,
                budget_N,
                config.max_iterations_stage2,
                config.pca_components
            )
            variant_samples.append(result['samples_to_target'])

            if (i + 1) % 5 == 0:
                print(f"    [{i+1}/{config.test_cppns}] total samples: {result['samples_to_target']:.0f}")

        avg_variant = np.mean(variant_samples)
        speedup = avg_baseline / avg_variant if avg_variant > 0 else 0

        variant_results[f"stage1_N_{budget_N}"] = {
            'avg_samples': float(avg_variant),
            'std_samples': float(np.std(variant_samples)),
            'speedup': float(speedup)
        }

        print(f"  ✓ N={budget_N}: {avg_variant:.0f} ± {np.std(variant_samples):.0f} samples, speedup={speedup:.2f}×")

    # Find best variant
    print(f"\n[4/5] Analyzing results...")
    best_speedup = max(v['speedup'] for v in variant_results.values())
    optimal_N = [k.split('_')[-1] for k, v in variant_results.items() if v['speedup'] == best_speedup][0]

    print(f"Best speedup: {best_speedup:.2f}× (N={optimal_N})")

    # Validate hypothesis: best >= 3×
    print(f"\n[5/5] Validating hypothesis...")
    speedup_threshold = 3.0
    validated = best_speedup >= speedup_threshold
    conclusion = "validate" if validated else "refute"

    print(f"Threshold: >= {speedup_threshold}×")
    print(f"Result: {best_speedup:.2f}×")
    print(f"Conclusion: {conclusion}")

    # Compile results
    results = {
        "method": "Two-stage sampling with 4 exploration budgets",
        "baseline_samples": float(avg_baseline),
        "baseline_samples_std": float(np.std(baseline_samples)),
        "stage1_50_total": variant_results.get("stage1_N_50", {}).get("avg_samples", 0),
        "stage1_100_total": variant_results.get("stage1_N_100", {}).get("avg_samples", 0),
        "stage1_150_total": variant_results.get("stage1_N_150", {}).get("avg_samples", 0),
        "stage1_200_total": variant_results.get("stage1_N_200", {}).get("avg_samples", 0),
        "best_speedup": float(best_speedup),
        "optimal_N": int(optimal_N),
        "speedup_threshold": speedup_threshold,
        "conclusion": conclusion
    }

    return results

def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-224: Multi-Stage Sampling (Exploration → Manifold Convergence)")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "multi_stage_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_224_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"RES-224 | multi_stage_sampling | {results['conclusion']} | speedup={results['best_speedup']:.2f}x")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "method": "Two-stage sampling with 4 exploration budgets",
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "multi_stage_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_224_results.json", 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
