#!/usr/bin/env python3
"""
RES-233: Hybrid Multi-Manifold Sampling

Hypothesis: Mixture of 2D/3D/5D manifolds avoids premature constraint, achieves ≥110× speedup

Method:
1. Implement hybrid sampling from mixture of manifold bases:
   - Stage 1 (N=50): Broad exploration, compute 2D, 3D, 5D PCA bases
   - Stage 2+: Sample from weighted mixture:
     * Variant A: Fixed weights (50% 2D, 30% 3D, 20% 5D)
     * Variant B: Decay weights (favor low-D over time)
     * Variant C: Adaptive weights (adjust based on acceptance rates)
2. Run 20 fresh CPPNs per variant to order 0.5
3. Measure: total samples, acceptance rates, manifold preference evolution
4. Compare vs RES-224 baseline (276 samples, 92× speedup)
5. Validate: best ≥ 110× speedup
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
    """Configuration for hybrid multi-manifold sampling experiment"""
    test_cppns: int = 20                    # CPPNs to test
    order_target: float = 0.50              # Target order level
    n_live_baseline: int = 100              # Baseline single-stage n_live
    baseline_max_iterations: int = 300      # Baseline max iterations

    # Hybrid manifold configs
    stage1_budget: int = 50                 # Exploration budget
    max_iterations_stage2: int = 250        # Stage 2 iterations
    pca_components: list = None             # [2, 3, 5] dimensions

    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.pca_components is None:
            self.pca_components = [2, 3, 5]

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

def run_baseline_single_stage(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    n_live: int,
    max_iterations: int
) -> dict:
    """Run standard single-stage nested sampling to target order."""
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

def run_hybrid_manifold_sampling(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    pca_dims: list = None,
    weight_variant: str = "fixed"
) -> dict:
    """
    Run hybrid multi-manifold sampling with mixture of 2D/3D/5D bases.

    Variants:
    - "fixed": Static weights (50% 2D, 30% 3D, 20% 5D)
    - "decay": Decay weights over time (favor low-D early)
    - "adaptive": Adjust based on acceptance rates per manifold
    """
    if pca_dims is None:
        pca_dims = [2, 3, 5]

    set_global_seed(None)

    # ===== STAGE 1: Exploration (collect samples to learn manifolds) =====
    n_live_stage1 = 50
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

    # Stage 1 exploration iterations
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

    # ===== STAGE 2: Multi-manifold convergence =====
    # Learn manifolds from Stage 1 samples
    pca_bases = {}
    for dim in pca_dims:
        mean, components, var = compute_pca_basis_from_samples(collected_weights, dim)
        pca_bases[dim] = {
            'mean': mean,
            'components': components,
            'variance_explained': var
        }

    # Initialize tracking for adaptive variant
    acceptance_counts = {dim: 0 for dim in pca_dims}
    proposal_counts = {dim: 0 for dim in pca_dims}

    # Continue with multi-manifold constrained sampling in Stage 2
    total_samples = stage1_samples_collected

    for iteration in range(max_iterations_stage2):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_stage1)

        # Choose manifold dimension based on variant
        if weight_variant == "fixed":
            # Fixed weights: 50% 2D, 30% 3D, 20% 5D
            r = np.random.rand()
            if r < 0.5:
                chosen_dim = 2
            elif r < 0.8:
                chosen_dim = 3
            else:
                chosen_dim = 5

        elif weight_variant == "decay":
            # Decay weights: favor lower dimensions over time
            progress = iteration / max_iterations_stage2
            # Decay weights: [w2, w3, w5] starts [0.6, 0.3, 0.1], ends [0.4, 0.3, 0.3]
            w2 = 0.6 - 0.2 * progress
            w3 = 0.3
            w5 = 0.1 + 0.2 * progress
            weights = np.array([w2, w3, w5]) / (w2 + w3 + w5)
            chosen_dim = np.random.choice(pca_dims, p=weights)

        elif weight_variant == "adaptive":
            # Adaptive: adjust based on acceptance rates
            total_proposals = sum(proposal_counts.values())
            if total_proposals > 0:
                acceptance_rates = {dim: acceptance_counts[dim] / max(1, proposal_counts[dim])
                                   for dim in pca_dims}
                # Normalize to weights (higher acceptance = higher weight)
                weights = np.array([acceptance_rates.get(dim, 0.1) for dim in pca_dims])
                weights = weights / (weights.sum() + 1e-8)
                weights = np.clip(weights, 0.05, 0.95)  # Avoid extreme weights
                weights = weights / weights.sum()
            else:
                weights = np.array([0.5, 0.3, 0.2])

            chosen_dim = np.random.choice(pca_dims, p=weights)
            proposal_counts[chosen_dim] += 1

        # Propose on chosen manifold
        base = pca_bases[chosen_dim]
        if base['mean'] is not None and base['components'] is not None:
            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, base['mean'], base['components'])

            if coeffs is not None:
                # Perturb coefficients
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                # Reconstruct
                proposal_w = reconstruct_from_pca(new_coeffs, base['mean'], base['components'])
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                # Accept if above threshold
                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)
                    if weight_variant == "adaptive":
                        acceptance_counts[chosen_dim] += 1

                total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target,
        'weight_variant': weight_variant
    }

def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete hybrid multi-manifold sampling experiment"""

    # Generate test CPPNs
    test_cppns = generate_test_cppns(config.test_cppns, config.seed)

    # Run baseline (single-stage) on all test CPPNs
    print(f"\n[2/6] Running baseline single-stage sampling on {config.test_cppns} CPPNs...")
    baseline_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_baseline_single_stage(
            cppn,
            config.order_target,
            config.image_size,
            config.n_live_baseline,
            config.baseline_max_iterations
        )
        baseline_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] baseline samples: {result['total_samples']:.0f}")

    baseline_samples = [r['total_samples'] for r in baseline_results]
    avg_baseline = np.mean(baseline_samples)

    print(f"✓ Baseline: {avg_baseline:.0f} ± {np.std(baseline_samples):.0f} samples")

    # Run three hybrid manifold variants
    print(f"\n[3/6] Running hybrid manifold variant A (fixed weights)...")
    fixed_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_hybrid_manifold_sampling(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components,
            weight_variant="fixed"
        )
        fixed_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total samples: {result['total_samples']:.0f}")

    fixed_samples = [r['total_samples'] for r in fixed_results]
    avg_fixed = np.mean(fixed_samples)
    speedup_fixed = avg_baseline / avg_fixed if avg_fixed > 0 else 0

    print(f"✓ Fixed weights: {avg_fixed:.0f} ± {np.std(fixed_samples):.0f} samples, speedup={speedup_fixed:.2f}×")

    print(f"\n[4/6] Running hybrid manifold variant B (decay weights)...")
    decay_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_hybrid_manifold_sampling(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components,
            weight_variant="decay"
        )
        decay_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total samples: {result['total_samples']:.0f}")

    decay_samples = [r['total_samples'] for r in decay_results]
    avg_decay = np.mean(decay_samples)
    speedup_decay = avg_baseline / avg_decay if avg_decay > 0 else 0

    print(f"✓ Decay weights: {avg_decay:.0f} ± {np.std(decay_samples):.0f} samples, speedup={speedup_decay:.2f}×")

    print(f"\n[5/6] Running hybrid manifold variant C (adaptive weights)...")
    adaptive_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_hybrid_manifold_sampling(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components,
            weight_variant="adaptive"
        )
        adaptive_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total samples: {result['total_samples']:.0f}")

    adaptive_samples = [r['total_samples'] for r in adaptive_results]
    avg_adaptive = np.mean(adaptive_samples)
    speedup_adaptive = avg_baseline / avg_adaptive if avg_adaptive > 0 else 0

    print(f"✓ Adaptive weights: {avg_adaptive:.0f} ± {np.std(adaptive_samples):.0f} samples, speedup={speedup_adaptive:.2f}×")

    # Find best variant
    print(f"\n[6/6] Analyzing results...")
    speedups = {
        'fixed': speedup_fixed,
        'decay': speedup_decay,
        'adaptive': speedup_adaptive
    }
    best_variant = max(speedups, key=speedups.get)
    best_speedup = speedups[best_variant]

    print(f"\nResults summary:")
    print(f"  Fixed:    {speedup_fixed:.2f}× speedup")
    print(f"  Decay:    {speedup_decay:.2f}× speedup")
    print(f"  Adaptive: {speedup_adaptive:.2f}× speedup")
    print(f"\nBest: {best_variant} with {best_speedup:.2f}× speedup")

    # Validate hypothesis: best >= 110×
    print(f"\nValidation:")
    speedup_threshold = 110.0
    validated = best_speedup >= speedup_threshold
    conclusion = "validate" if validated else "refute"

    print(f"  Threshold: >= {speedup_threshold}×")
    print(f"  Achieved: {best_speedup:.2f}×")
    print(f"  Conclusion: {conclusion}")

    # Compile results
    results = {
        "method": "Mixture of 2D/3D/5D manifold bases with adaptive weighting",
        "baseline_samples": float(avg_baseline),
        "baseline_samples_std": float(np.std(baseline_samples)),
        "fixed_weights_samples": float(avg_fixed),
        "fixed_weights_std": float(np.std(fixed_samples)),
        "fixed_speedup": float(speedup_fixed),
        "decay_weights_samples": float(avg_decay),
        "decay_weights_std": float(np.std(decay_samples)),
        "decay_speedup": float(speedup_decay),
        "adaptive_weights_samples": float(avg_adaptive),
        "adaptive_weights_std": float(np.std(adaptive_samples)),
        "adaptive_speedup": float(speedup_adaptive),
        "best_speedup": float(best_speedup),
        "best_variant": best_variant,
        "speedup_threshold": speedup_threshold,
        "conclusion": conclusion
    }

    return results

def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-233: Hybrid Multi-Manifold Sampling")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "hybrid_manifold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_233_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"RES-233 | hybrid_manifold_sampling | {results['conclusion']} | best_speedup={results['best_speedup']:.2f}x variant={results['best_variant']}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "method": "Mixture of 2D/3D/5D manifold bases with adaptive weighting",
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "hybrid_manifold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_233_results.json", 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
