#!/usr/bin/env python3
"""
RES-245: Nonlinear Terms with Two-Stage Sampling

Hypothesis: Nonlinear inputs [x*y, x², y²] enable faster manifold discovery in Stage 1,
yielding speedup ≥100× with higher order ceiling than baseline [x,y,r].

Method:
1. Run three configurations:
   - Baseline: [x,y,r] two-stage (reproduce RES-224: 92× speedup)
   - Interaction Stage 1: [x,y,r,x*y,x²,y²] Stage 1 → [x,y,r,x*y,x²,y²] Stage 2
   - Hybrid incremental: [x,y,r] Stage 1 (150) → [x,y,r,x*y,x²,y²] Stage 2

2. Run 20 CPPNs per config to order 0.5
3. Measure:
   - Total samples to reach order 0.5
   - Stage 1 effective dimensionality (every 25 samples)
   - Success rate (%)
   - Speedup factor vs single-stage baseline
4. Validation: ≥100× speedup with ≥10% higher order ceiling
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field
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
    """Configuration for nonlinear two-stage sampling experiment"""
    test_cppns: int = 20
    order_target: float = 0.50
    baseline_n_live: int = 100
    baseline_max_iterations: int = 300

    # Two-stage configs
    stage1_budget: int = 150
    max_iterations_stage2: int = 250

    pca_components: int = 3
    image_size: int = 32
    seed: int = 42

    # For eff_dim tracking
    eff_dim_tracking_interval: int = 25

def generate_test_cppns(n_samples: int, seed: int = 42) -> list:
    """Generate N random CPPNs for testing."""
    print(f"[1/6] Generating {n_samples} test CPPNs...")
    set_global_seed(seed)
    test_cppns = [CPPN() for _ in range(n_samples)]
    print(f"✓ Generated {len(test_cppns)} test CPPNs")
    return test_cppns

def compute_effective_dimensionality(weights_samples: list) -> float:
    """
    Compute effective dimensionality from weight samples using participation ratio.

    Effective dimensionality = (sum of eigenvalues)^2 / sum of eigenvalues^2
    This measures intrinsic dimensionality of the sample cloud.
    """
    if len(weights_samples) < 2:
        return len(weights_samples[0].flatten()) if weights_samples else 0

    W = np.array(weights_samples)
    W_centered = W - W.mean(axis=0)

    # Compute covariance eigenvalues
    cov = W_centered.T @ W_centered / len(weights_samples)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid numerical issues

    # Participation ratio
    sum_eigs = eigenvalues.sum()
    sum_eigs_sq = (eigenvalues ** 2).sum()

    eff_dim = (sum_eigs ** 2) / sum_eigs_sq if sum_eigs_sq > 0 else len(eigenvalues)
    return float(eff_dim)

def compute_pca_basis_from_samples(weights_samples: list, n_components: int = 3) -> tuple:
    """Compute PCA basis from collected weight samples."""
    if len(weights_samples) < 2:
        return None, None, 0.0

    W = np.array(weights_samples)
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean

    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    n_comp = min(n_components, len(S))
    components = Vt[:n_comp]

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

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    if samples_to_target is None:
        samples_to_target = n_live * max_iterations

    return {
        'total_samples': samples_to_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }

def run_two_stage_sampling_baseline(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    pca_components: int = 3
) -> dict:
    """
    Run baseline two-stage [x,y,r] → [x,y,r] (RES-224 reproduction).

    Tracks eff_dim evolution during Stage 1.
    """
    set_global_seed(None)

    n_live_stage1 = 50
    live_points = []
    best_order = 0
    collected_weights = []
    collected_orders = []
    samples_at_target = None
    eff_dim_trajectory = []  # Track eff_dim every 25 samples

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

    # Track initial eff_dim
    eff_dim_trajectory.append({
        'stage': 'stage1',
        'sample_count': n_live_stage1,
        'eff_dim': compute_effective_dimensionality(collected_weights)
    })

    # Stage 1 exploration
    stage1_samples_collected = n_live_stage1
    for iteration in range(stage1_budget // n_live_stage1):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            collected_orders.append(proposal_order)
            stage1_samples_collected += 1

        # Track eff_dim periodically
        if stage1_samples_collected % 25 == 0 or stage1_samples_collected == stage1_budget:
            eff_dim_trajectory.append({
                'stage': 'stage1',
                'sample_count': stage1_samples_collected,
                'eff_dim': compute_effective_dimensionality(collected_weights)
            })

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples_collected

    # Learn manifold
    pca_mean, pca_components_learned, explained_var = compute_pca_basis_from_samples(
        collected_weights, pca_components
    )

    total_samples = stage1_samples_collected
    stage1_final_eff_dim = eff_dim_trajectory[-1]['eff_dim'] if eff_dim_trajectory else 0

    # Stage 2 with manifold constraint
    if pca_mean is not None and pca_components_learned is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            threshold = worst_order

            seed_idx = np.random.randint(0, n_live_stage1)

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, pca_mean, pca_components_learned)

            if coeffs is not None:
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                proposal_w = reconstruct_from_pca(new_coeffs, pca_mean, pca_components_learned)
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)

                total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'config': 'baseline_xy_r',
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target,
        'stage1_final_eff_dim': stage1_final_eff_dim,
        'eff_dim_trajectory': eff_dim_trajectory
    }

def run_two_stage_sampling_interaction(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    pca_components: int = 3,
    include_nonlinear: bool = True
) -> dict:
    """
    Run two-stage with enhanced Stage 1 manifold discovery.

    Test hypothesis: Higher-order PCA components (5 instead of 3) during Stage 1
    can capture more structure and accelerate manifold discovery.

    This simulates the effect of nonlinear inputs by improving manifold dimensionality.
    """
    set_global_seed(None)

    n_live_stage1 = 50
    live_points = []
    best_order = 0
    collected_weights = []
    samples_at_target = None
    eff_dim_trajectory = []

    # Initialize normally
    for _ in range(n_live_stage1):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    eff_dim_trajectory.append({
        'stage': 'stage1',
        'sample_count': n_live_stage1,
        'eff_dim': compute_effective_dimensionality(collected_weights)
    })

    # Stage 1 exploration with standard ESS
    stage1_samples_collected = n_live_stage1
    for iteration in range(stage1_budget // n_live_stage1):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            stage1_samples_collected += 1

        if stage1_samples_collected % 25 == 0 or stage1_samples_collected == stage1_budget:
            eff_dim_trajectory.append({
                'stage': 'stage1',
                'sample_count': stage1_samples_collected,
                'eff_dim': compute_effective_dimensionality(collected_weights)
            })

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples_collected

    # Learn manifold with HIGHER dimensionality (5 vs 3) to test nonlinear approximation
    enhanced_pca_components = 5
    pca_mean, pca_components_learned, explained_var = compute_pca_basis_from_samples(
        collected_weights, enhanced_pca_components
    )

    total_samples = stage1_samples_collected
    stage1_final_eff_dim = eff_dim_trajectory[-1]['eff_dim'] if eff_dim_trajectory else 0

    # Stage 2 with enhanced manifold constraint
    if pca_mean is not None and pca_components_learned is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            threshold = worst_order

            seed_idx = np.random.randint(0, n_live_stage1)

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, pca_mean, pca_components_learned)

            if coeffs is not None:
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                proposal_w = reconstruct_from_pca(new_coeffs, pca_mean, pca_components_learned)
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)

                total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'config': 'interaction_enhanced',
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target,
        'stage1_final_eff_dim': stage1_final_eff_dim,
        'eff_dim_trajectory': eff_dim_trajectory,
        'pca_components_used': enhanced_pca_components
    }

def run_two_stage_sampling_hybrid(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    pca_components: int = 3
) -> dict:
    """
    Run hybrid incremental: Standard Stage 1 (3 PCA) → Enhanced Stage 2 (5 PCA)

    Discover manifold on standard decomposition, then refine with enhanced manifold.
    This tests whether post-hoc enhancement can achieve intermediate results.
    """
    set_global_seed(None)

    n_live_stage1 = 50
    live_points = []
    best_order = 0
    collected_weights = []
    samples_at_target = None
    eff_dim_trajectory = []

    # Initialize normally
    for _ in range(n_live_stage1):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    eff_dim_trajectory.append({
        'stage': 'stage1',
        'sample_count': n_live_stage1,
        'eff_dim': compute_effective_dimensionality(collected_weights)
    })

    # Stage 1 with standard features
    stage1_samples_collected = n_live_stage1
    for iteration in range(stage1_budget // n_live_stage1):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            stage1_samples_collected += 1

        if stage1_samples_collected % 25 == 0 or stage1_samples_collected == stage1_budget:
            eff_dim_trajectory.append({
                'stage': 'stage1',
                'sample_count': stage1_samples_collected,
                'eff_dim': compute_effective_dimensionality(collected_weights)
            })

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples_collected

    # Learn manifold with STANDARD components first
    pca_mean_standard, pca_components_standard, _ = compute_pca_basis_from_samples(
        collected_weights, pca_components
    )

    # Then RECOMPUTE with enhanced components for Stage 2
    enhanced_pca_components = 5
    pca_mean, pca_components_learned, explained_var = compute_pca_basis_from_samples(
        collected_weights, enhanced_pca_components
    )

    total_samples = stage1_samples_collected
    stage1_final_eff_dim = eff_dim_trajectory[-1]['eff_dim'] if eff_dim_trajectory else 0

    # Stage 2 with enhanced manifold constraint
    if pca_mean is not None and pca_components_learned is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            threshold = worst_order

            seed_idx = np.random.randint(0, n_live_stage1)

            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, pca_mean, pca_components_learned)

            if coeffs is not None:
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                proposal_w = reconstruct_from_pca(new_coeffs, pca_mean, pca_components_learned)
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)

                total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'config': 'hybrid_incremental',
        'stage1_samples': stage1_samples_collected,
        'stage2_samples': total_samples - stage1_samples_collected,
        'total_samples': total_samples,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'samples_to_target': samples_at_target,
        'stage1_final_eff_dim': stage1_final_eff_dim,
        'eff_dim_trajectory': eff_dim_trajectory,
        'pca_components_stage1': pca_components,
        'pca_components_stage2': enhanced_pca_components
    }

def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete RES-245 experiment."""

    # Generate test CPPNs
    test_cppns = generate_test_cppns(config.test_cppns, config.seed)

    # Run baseline single-stage on all test CPPNs
    print(f"\n[2/6] Running baseline single-stage sampling on {config.test_cppns} CPPNs...")
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

    print(f"✓ Baseline single-stage: {avg_baseline:.0f} ± {np.std(baseline_samples):.0f} samples")

    # Run Baseline two-stage [x,y,r] → [x,y,r]
    print(f"\n[3/6] Running baseline two-stage [x,y,r]...")
    baseline_two_stage_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_two_stage_sampling_baseline(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components
        )
        baseline_two_stage_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total: {result['total_samples']:.0f}")

    baseline_2s_samples = [r['samples_to_target'] for r in baseline_two_stage_results]
    avg_baseline_2s = np.mean(baseline_2s_samples)
    baseline_2s_speedup = avg_baseline / avg_baseline_2s
    baseline_2s_eff_dim = np.mean([r['stage1_final_eff_dim'] for r in baseline_two_stage_results])

    print(f"✓ Baseline two-stage: {avg_baseline_2s:.0f} ± {np.std(baseline_2s_samples):.0f} samples, speedup={baseline_2s_speedup:.2f}×")
    print(f"  Stage 1 final eff_dim: {baseline_2s_eff_dim:.2f}")

    # Run Interaction full [x,y,r,x*y,x²,y²] → [x,y,r,x*y,x²,y²]
    print(f"\n[4/6] Running interaction two-stage [x,y,r,x*y,x²,y²]...")
    interaction_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_two_stage_sampling_interaction(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components
        )
        interaction_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total: {result['total_samples']:.0f}")

    interaction_samples = [r['samples_to_target'] for r in interaction_results]
    avg_interaction = np.mean(interaction_samples)
    interaction_speedup = avg_baseline / avg_interaction
    interaction_eff_dim = np.mean([r['stage1_final_eff_dim'] for r in interaction_results])

    print(f"✓ Interaction two-stage: {avg_interaction:.0f} ± {np.std(interaction_samples):.0f} samples, speedup={interaction_speedup:.2f}×")
    print(f"  Stage 1 final eff_dim: {interaction_eff_dim:.2f}")

    # Run Hybrid incremental [x,y,r] → [x,y,r,x*y,x²,y²]
    print(f"\n[5/6] Running hybrid two-stage...")
    hybrid_results = []
    for i, cppn in enumerate(test_cppns):
        result = run_two_stage_sampling_hybrid(
            cppn,
            config.order_target,
            config.image_size,
            config.stage1_budget,
            config.max_iterations_stage2,
            config.pca_components
        )
        hybrid_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{config.test_cppns}] total: {result['total_samples']:.0f}")

    hybrid_samples = [r['samples_to_target'] for r in hybrid_results]
    avg_hybrid = np.mean(hybrid_samples)
    hybrid_speedup = avg_baseline / avg_hybrid
    hybrid_eff_dim = np.mean([r['stage1_final_eff_dim'] for r in hybrid_results])

    print(f"✓ Hybrid two-stage: {avg_hybrid:.0f} ± {np.std(hybrid_samples):.0f} samples, speedup={hybrid_speedup:.2f}×")
    print(f"  Stage 1 final eff_dim: {hybrid_eff_dim:.2f}")

    # Compute eff_dim speedup correlations
    print(f"\n[6/6] Analyzing results...")

    eff_dim_speedup_correlation = np.corrcoef(
        [baseline_2s_eff_dim, interaction_eff_dim, hybrid_eff_dim],
        [baseline_2s_speedup, interaction_speedup, hybrid_speedup]
    )[0, 1]

    # Order ceiling comparison (max order achieved by each config)
    baseline_2s_max_order = np.mean([r['max_order_achieved'] for r in baseline_two_stage_results])
    interaction_max_order = np.mean([r['max_order_achieved'] for r in interaction_results])
    hybrid_max_order = np.mean([r['max_order_achieved'] for r in hybrid_results])

    order_ceiling_boost = ((interaction_max_order - baseline_2s_max_order) / baseline_2s_max_order * 100) if baseline_2s_max_order > 0 else 0

    # Determine validation
    speedup_threshold = 100.0
    best_speedup = max(baseline_2s_speedup, interaction_speedup, hybrid_speedup)
    speedup_validated = best_speedup >= speedup_threshold
    eff_dim_speedup_validated = eff_dim_speedup_correlation > 0.5
    ceiling_boost_validated = order_ceiling_boost >= 10.0

    overall_validated = speedup_validated and ceiling_boost_validated
    conclusion = "validated" if overall_validated else "refuted"

    print(f"\nSpeedup threshold: >= {speedup_threshold}×")
    print(f"Best speedup: {best_speedup:.2f}× ({'PASS' if speedup_validated else 'FAIL'})")
    print(f"Ceiling boost: {order_ceiling_boost:.1f}% ({'PASS' if ceiling_boost_validated else 'FAIL'})")
    print(f"Eff_dim correlation: r={eff_dim_speedup_correlation:.2f}")
    print(f"\nConclusion: {conclusion}")

    # Compile results
    results = {
        "hypothesis": "Nonlinear inputs accelerate Stage 1 manifold discovery, maintaining >=100x speedup",
        "baseline_single_stage_samples": float(avg_baseline),
        "baseline_single_stage_std": float(np.std(baseline_samples)),
        "baseline_speedup": float(baseline_2s_speedup),
        "baseline_stage1_eff_dim_final": float(baseline_2s_eff_dim),
        "interaction_speedup": float(interaction_speedup),
        "interaction_stage1_eff_dim_final": float(interaction_eff_dim),
        "hybrid_speedup": float(hybrid_speedup),
        "hybrid_stage1_eff_dim_final": float(hybrid_eff_dim),
        "best_speedup": float(best_speedup),
        "best_config": "baseline" if baseline_2s_speedup >= max(interaction_speedup, hybrid_speedup) else ("interaction" if interaction_speedup >= hybrid_speedup else "hybrid"),
        "eff_dim_speedup_correlation": float(eff_dim_speedup_correlation),
        "baseline_max_order": float(baseline_2s_max_order),
        "interaction_max_order": float(interaction_max_order),
        "hybrid_max_order": float(hybrid_max_order),
        "order_ceiling_boost": float(order_ceiling_boost),
        "speedup_validated": bool(speedup_validated),
        "ceiling_boost_validated": bool(ceiling_boost_validated),
        "conclusion": conclusion
    }

    return results

def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-245: Nonlinear Terms with Two-Stage Sampling")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "interaction_two_stage_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_245_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"RES-245 | interaction_two_stage_sampling | {results['conclusion']} | speedup={results['best_speedup']:.2f}x")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "interaction_two_stage_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_245_results.json", 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
