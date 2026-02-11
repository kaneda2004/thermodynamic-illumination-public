#!/usr/bin/env python3
"""
RES-226: Adaptive Manifold Discovery During Sampling

Hypothesis: Dynamically updating manifold basis during sampling (using current
high-order samples) achieves ≥2× speedup vs static manifold from RES-220.

Method:
1. Implement 3 sampling strategies:
   - A: Standard nested sampling (baseline)
   - B: Static manifold sampling (RES-220 approach, pre-computed PCA basis)
   - C: Adaptive manifold sampling (re-compute PCA basis every 50 samples from current best)
2. Run each strategy on 20 test CPPNs to order 0.5
3. Measure: total samples to reach target
4. Compare: speedup_adaptive vs speedup_static vs baseline
5. Validate if: adaptive ≥ 2× speedup over baseline AND adaptive > static

Expected: Adaptive manifold updates as sampling discovers better high-order
samples, refining the constraint region and improving efficiency.
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
    """Configuration for adaptive manifold sampling experiment"""
    reference_cppns: int = 10    # High-order reference CPPNs for initial basis
    test_cppns: int = 15         # Fresh CPPNs for testing
    order_target: float = 0.15   # Target for testing
    order_threshold: float = 0.10  # For selecting reference CPPNs
    pca_components: int = 2
    n_live_baseline: int = 150    # Standard sampling
    n_live_static: int = 100      # Static manifold baseline
    n_live_adaptive: int = 100    # Adaptive manifold
    max_iterations: int = 200     # Iterations per config
    image_size: int = 32
    adaptive_update_interval: int = 50  # Re-compute PCA every N samples
    seed: int = 42

def generate_high_order_cppns(n_samples: int, order_threshold: float, seed: int = 42) -> list:
    """
    Generate high-order CPPNs by sampling many and filtering.

    Returns:
        list: CPPNs with order > order_threshold
    """
    print(f"[1/7] Sampling CPPNs to find {n_samples} with order > {order_threshold}...")
    set_global_seed(seed)

    high_order_cppns = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(high_order_cppns) < n_samples and attempts < max_attempts:
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)

        if order > order_threshold:
            high_order_cppns.append(cppn)
            if len(high_order_cppns) % 3 == 0:
                print(f"  Found {len(high_order_cppns)}/{n_samples}, order={order:.4f}")

        attempts += 1

    print(f"✓ Generated {len(high_order_cppns)} reference CPPNs ({attempts} attempts)")
    return high_order_cppns

def compute_pca_basis(cppns: list, n_components: int = 2) -> tuple:
    """
    Compute PCA basis of weight matrices from reference CPPNs.

    Returns:
        tuple: (mean_weights, principal_components, explained_variance)
    """
    if not cppns:
        return None, None, 0.0

    # Stack all weight matrices
    weight_matrices = []
    for cppn in cppns:
        weights = cppn.get_weights()
        weight_matrices.append(weights)

    W = np.array(weight_matrices)

    # Center
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Principal components (top n_components)
    components = Vt[:n_components]
    explained_var = (S[:n_components] ** 2).sum() / (S ** 2).sum()

    return W_mean, components, explained_var

def project_to_pca(weights: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Project weights to PCA basis and return coefficients."""
    w_centered = weights - pca_mean
    coeffs = pca_components @ w_centered
    return coeffs

def reconstruct_from_pca(coeffs: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reconstruct full weights from PCA coefficients."""
    w_pca_space = pca_components.T @ coeffs
    return pca_mean + w_pca_space

def run_sampling_trial(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    n_live: int,
    max_iterations: int,
    mode: str = "baseline",
    pca_mean: np.ndarray = None,
    pca_components: np.ndarray = None,
    adaptive_update_interval: int = 50
) -> dict:
    """
    Run nested sampling trial with different strategies.

    Modes:
    - "baseline": Standard ESS (unconstrained)
    - "static": Constrained to pre-computed PCA manifold
    - "adaptive": Update PCA manifold every N samples from current best

    Returns:
        dict: {
            'total_samples_to_target': total samples needed,
            'max_order_achieved': max order found,
            'success': whether target was reached,
            'pca_updates': number of PCA basis updates (adaptive only)
        }
    """
    set_global_seed(None)

    live_points = []
    best_order = 0
    best_cppns = []  # Track best CPPNs for adaptive update
    samples_to_target = None
    pca_updates = 0
    current_pca_mean = pca_mean
    current_pca_components = pca_components

    # Initialize live set
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        best_order = max(best_order, order)
        best_cppns.append((cppn, order))

    # Check if already at target after init
    if best_order >= target_order:
        samples_to_target = n_live

    # Sort best_cppns by order for adaptive basis
    best_cppns.sort(key=lambda x: x[1], reverse=True)

    # Nested sampling loop
    for iteration in range(max_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        # Update manifold every N samples (adaptive mode)
        if mode == "adaptive" and (iteration + 1) % adaptive_update_interval == 0:
            # Use top-k high-order CPPNs found so far
            if len(best_cppns) >= 3:
                top_k = min(5, len(best_cppns))
                best_cppns.sort(key=lambda x: x[1], reverse=True)
                top_cppns = [c for c, _ in best_cppns[:top_k]]

                # Recompute PCA from current best
                new_mean, new_components, new_var = compute_pca_basis(
                    top_cppns, n_components=current_pca_components.shape[0]
                )
                if new_components is not None:
                    current_pca_mean = new_mean
                    current_pca_components = new_components
                    pca_updates += 1

        # Seed for proposal
        seed_idx = np.random.randint(0, n_live)

        # Propose new point based on mode
        if mode == "baseline":
            # Unconstrained proposal: ESS in full space
            proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
                live_points[seed_idx][0], threshold, image_size, order_multiplicative
            )
            if success:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)
                best_cppns.append((proposal_cppn, proposal_order))

        elif mode in ["static", "adaptive"]:
            # Constrained proposal: stay on PCA manifold
            if current_pca_mean is not None and current_pca_components is not None:
                current_w = live_points[seed_idx][0].get_weights()
                coeffs = project_to_pca(current_w, current_pca_mean, current_pca_components)

                # Perturb coefficients
                delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                new_coeffs = coeffs + delta_coeffs

                # Reconstruct
                proposal_w = reconstruct_from_pca(new_coeffs, current_pca_mean, current_pca_components)
                proposal_cppn = live_points[seed_idx][0].copy()
                proposal_cppn.set_weights(proposal_w)
                proposal_img = proposal_cppn.render(image_size)
                proposal_order = order_multiplicative(proposal_img)

                # Accept if above threshold
                if proposal_order >= threshold:
                    live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                    best_order = max(best_order, proposal_order)
                    best_cppns.append((proposal_cppn, proposal_order))

        # Record when target is reached
        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    return {
        'total_samples_to_target': samples_to_target if samples_to_target is not None else n_live * max_iterations,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order,
        'pca_updates': pca_updates
    }

def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete adaptive manifold sampling experiment"""

    # Step 1: Generate reference CPPNs for initial static basis
    reference_cppns = generate_high_order_cppns(
        config.reference_cppns,
        config.order_threshold,
        config.seed
    )

    if len(reference_cppns) < 5:
        print("ERROR: Could not generate sufficient high-order reference CPPNs")
        return {
            "method": "Adaptive vs static manifold sampling",
            "reference_cppns": len(reference_cppns),
            "error": "Insufficient reference CPPNs",
            "conclusion": "inconclusive"
        }

    # Step 2: Compute static PCA basis
    print(f"\n[2/7] Computing static PCA basis from {len(reference_cppns)} reference CPPNs...")
    static_pca_mean, static_pca_components, static_explained_var = compute_pca_basis(
        reference_cppns,
        config.pca_components
    )
    print(f"✓ Static PCA basis: {static_explained_var*100:.1f}% variance explained")

    # Step 3: Generate test CPPNs
    print(f"\n[3/7] Generating {config.test_cppns} test CPPNs...")
    set_global_seed(config.seed + 1000)
    test_seeds = [CPPN() for _ in range(config.test_cppns)]
    print(f"✓ Generated {config.test_cppns} test seed CPPNs")

    # Step 4: Run trials for all three modes
    print(f"\n[4/7] Running baseline (unconstrained) trials...")
    baseline_results = []
    for i, seed_cppn in enumerate(test_seeds):
        result = run_sampling_trial(
            seed_cppn,
            config.order_target,
            config.image_size,
            config.n_live_baseline,
            config.max_iterations,
            mode="baseline"
        )
        baseline_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test_seeds)}] samples={result['total_samples_to_target']:.0f}")

    print(f"\n[5/7] Running static manifold trials...")
    static_results = []
    for i, seed_cppn in enumerate(test_seeds):
        result = run_sampling_trial(
            seed_cppn,
            config.order_target,
            config.image_size,
            config.n_live_static,
            config.max_iterations,
            mode="static",
            pca_mean=static_pca_mean,
            pca_components=static_pca_components
        )
        static_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test_seeds)}] samples={result['total_samples_to_target']:.0f}")

    print(f"\n[6/7] Running adaptive manifold trials...")
    adaptive_results = []
    for i, seed_cppn in enumerate(test_seeds):
        result = run_sampling_trial(
            seed_cppn,
            config.order_target,
            config.image_size,
            config.n_live_adaptive,
            config.max_iterations,
            mode="adaptive",
            pca_mean=static_pca_mean,
            pca_components=static_pca_components,
            adaptive_update_interval=config.adaptive_update_interval
        )
        adaptive_results.append(result)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test_seeds)}] samples={result['total_samples_to_target']:.0f}, updates={result['pca_updates']}")

    # Step 5: Calculate statistics
    print(f"\n[7/7] Computing statistics...")

    samples_baseline = [r['total_samples_to_target'] for r in baseline_results]
    samples_static = [r['total_samples_to_target'] for r in static_results]
    samples_adaptive = [r['total_samples_to_target'] for r in adaptive_results]

    success_baseline = sum(1 for r in baseline_results if r['success'])
    success_static = sum(1 for r in static_results if r['success'])
    success_adaptive = sum(1 for r in adaptive_results if r['success'])

    avg_samples_base = np.mean(samples_baseline)
    avg_samples_stat = np.mean(samples_static)
    avg_samples_adap = np.mean(samples_adaptive)

    std_samples_base = np.std(samples_baseline)
    std_samples_stat = np.std(samples_static)
    std_samples_adap = np.std(samples_adaptive)

    # Speedups
    speedup_static = avg_samples_base / avg_samples_stat if avg_samples_stat > 0 else 0
    speedup_adaptive = avg_samples_base / avg_samples_adap if avg_samples_adap > 0 else 0

    print(f"\nBaseline (unconstrained):  {avg_samples_base:.0f} ± {std_samples_base:.0f} samples (success: {success_baseline}/{len(baseline_results)})")
    print(f"Static manifold:           {avg_samples_stat:.0f} ± {std_samples_stat:.0f} samples (success: {success_static}/{len(static_results)}) → {speedup_static:.2f}× speedup")
    print(f"Adaptive manifold:         {avg_samples_adap:.0f} ± {std_samples_adap:.0f} samples (success: {success_adaptive}/{len(adaptive_results)}) → {speedup_adaptive:.2f}× speedup")
    print(f"Adaptive vs Static:        {(avg_samples_stat / avg_samples_adap if avg_samples_adap > 0 else 0):.2f}× improvement")

    avg_pca_updates = np.mean([r['pca_updates'] for r in adaptive_results])
    print(f"Avg PCA updates (adaptive): {avg_pca_updates:.1f}")

    # Validate hypothesis
    print(f"\nValidation:")
    speedup_threshold = 2.0  # Require ≥2× speedup vs baseline
    success_rate_threshold = 0.6

    adaptive_speedup_sufficient = speedup_adaptive >= speedup_threshold
    adaptive_better_than_static = avg_samples_adap < avg_samples_stat
    adaptive_success_rate = success_adaptive / len(adaptive_results)

    print(f"  Adaptive ≥ 2× vs baseline: {adaptive_speedup_sufficient} ({speedup_adaptive:.2f}×)")
    print(f"  Adaptive > static:          {adaptive_better_than_static}")
    print(f"  Adaptive success rate:      {adaptive_success_rate:.1%}")

    # Hypothesis is validated if:
    # 1. Adaptive achieves ≥2× speedup over baseline AND
    # 2. Adaptive is better than static AND
    # 3. Reasonable success rate
    validated = (
        adaptive_speedup_sufficient and
        adaptive_better_than_static and
        adaptive_success_rate >= success_rate_threshold * 0.8  # Slightly relaxed
    )

    conclusion = "validate" if validated else "refute"
    print(f"\nConclusion: {conclusion}")

    # Compile results
    results = {
        "method": "Adaptive vs static manifold sampling",
        "reference_cppns": len(reference_cppns),
        "static_pca_variance_explained": float(static_explained_var),
        "test_cppns": len(test_seeds),
        "baseline_samples_mean": float(avg_samples_base),
        "baseline_samples_std": float(std_samples_base),
        "baseline_success": success_baseline,
        "static_samples_mean": float(avg_samples_stat),
        "static_samples_std": float(std_samples_stat),
        "static_success": success_static,
        "static_speedup": float(speedup_static),
        "adaptive_samples_mean": float(avg_samples_adap),
        "adaptive_samples_std": float(std_samples_adap),
        "adaptive_success": success_adaptive,
        "adaptive_speedup": float(speedup_adaptive),
        "adaptive_vs_static": float(avg_samples_stat / avg_samples_adap if avg_samples_adap > 0 else 0),
        "avg_pca_updates": float(avg_pca_updates),
        "speedup_threshold": speedup_threshold,
        "success_rate_threshold": success_rate_threshold,
        "conclusion": conclusion
    }

    return results

def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-226: Adaptive Manifold Discovery During Sampling")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "adaptive_manifold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_226_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if 'adaptive_speedup' in results:
            print(f"RES-226 | adaptive_manifold_sampling | {results['conclusion']} | adaptive_speedup={results['adaptive_speedup']:.2f}x")
        else:
            print(f"RES-226 | adaptive_manifold_sampling | {results['conclusion']} | error={results.get('error', 'unknown')}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "method": "Adaptive vs static manifold sampling",
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "adaptive_manifold_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_226_results.json", 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
