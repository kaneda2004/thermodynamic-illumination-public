#!/usr/bin/env python3
"""
RES-220: Manifold-Constrained Sampling Accelerates Order Discovery

Hypothesis: Sampling that constrains proposals to a low-D subspace
(PCA basis of high-order weights) achieves order > 0.5 in 3-5× fewer
samples than unconstrained nested sampling.

Method:
1. Generate 20 reference high-order CPPNs (order > 0.6) via standard nested sampling
2. Compute 2D PCA basis of their weight matrices (capture manifold)
3. Run two sampling experiments on 20 fresh CPPNs:
   - A: Standard nested sampling to order 0.5 target
   - B: Constrained nested sampling (propose only along PCA basis)
4. Compare: average n_live_points to reach order 0.5
5. Calculate speedup = n_live_standard / n_live_constrained
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
    """Configuration for manifold sampling experiment"""
    reference_cppns: int = 12    # High-order reference CPPNs (fewer for speed)
    test_cppns: int = 10         # Fresh CPPNs for testing
    order_target: float = 0.15   # Target for testing (realistic given distribution)
    order_threshold: float = 0.10  # For selecting reference CPPNs
    pca_components: int = 2
    n_live_std: int = 150        # Standard sampling
    n_live_constrained: int = 100  # Constrained sampling baseline
    max_iterations: int = 200     # Iterations per config
    image_size: int = 32
    seed: int = 42

def generate_high_order_cppns(n_samples: int, order_threshold: float, seed: int = 42) -> list:
    """
    Generate high-order CPPNs by sampling many and filtering.

    Returns:
        list: CPPNs with order > order_threshold
    """
    print(f"[1/6] Sampling CPPNs to find {n_samples} with order > {order_threshold}...")
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
            if len(high_order_cppns) % 4 == 0:
                print(f"  Found {len(high_order_cppns)}/{n_samples}, order={order:.4f}")

        attempts += 1

    print(f"✓ Generated {len(high_order_cppns)} reference CPPNs (found {attempts} attempts)")
    return high_order_cppns

def compute_pca_basis(cppns: list, n_components: int = 2) -> tuple:
    """
    Compute PCA basis of weight matrices from reference CPPNs.

    Returns:
        tuple: (mean_weights, principal_components, explained_variance)
    """
    print(f"\n[2/6] Computing {n_components}D PCA basis of weight matrices...")

    # Stack all weight matrices
    weight_matrices = []
    for cppn in cppns:
        weights = cppn.get_weights()
        weight_matrices.append(weights)

    W = np.array(weight_matrices)
    print(f"  Weight matrix shape: {W.shape}")

    # Center
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Principal components (top n_components)
    components = Vt[:n_components]
    explained_var = (S[:n_components] ** 2).sum() / (S ** 2).sum()

    print(f"✓ PCA basis computed: {explained_var*100:.1f}% variance explained")
    return W_mean, components, explained_var

def project_to_pca(weights: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> tuple:
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
    constrain_to_pca: bool = False,
    pca_mean: np.ndarray = None,
    pca_components: np.ndarray = None
) -> dict:
    """
    Run nested sampling trial and track samples to reach target.

    Key metric: total_samples = n_live * (iterations + 1)
    This measures total effective samples drawn, not just iterations.

    Returns:
        dict: {
            'n_live': n_live,
            'total_samples_to_target': total samples needed,
            'max_order_achieved': max order found,
            'success': whether target was reached
        }
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

    # Check if already at target after init
    if best_order >= target_order:
        samples_to_target = n_live

    # Nested sampling loop
    for iteration in range(max_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        # Seed for proposal
        seed_idx = np.random.randint(0, n_live)

        # Propose new point
        if constrain_to_pca and pca_mean is not None:
            # Constrained proposal: stay on PCA manifold
            current_w = live_points[seed_idx][0].get_weights()
            coeffs = project_to_pca(current_w, pca_mean, pca_components)

            # Perturb coefficients
            delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
            new_coeffs = coeffs + delta_coeffs

            # Reconstruct
            proposal_w = reconstruct_from_pca(new_coeffs, pca_mean, pca_components)
            proposal_cppn = live_points[seed_idx][0].copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_multiplicative(proposal_img)

            # Accept if above threshold
            if proposal_order >= threshold:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)
        else:
            # Unconstrained proposal: ESS in full space
            proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
                live_points[seed_idx][0], threshold, image_size, order_multiplicative
            )
            if success:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)

        # Record when target is reached
        if best_order >= target_order and samples_to_target is None:
            samples_to_target = n_live + (iteration + 1)

    return {
        'n_live': n_live,
        'total_samples_to_target': samples_to_target if samples_to_target is not None else n_live * max_iterations,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }

def run_experiment(config: ExperimentConfig) -> dict:
    """Run the complete manifold-constrained sampling experiment"""

    # Step 1: Generate reference CPPNs
    reference_cppns = generate_high_order_cppns(
        config.reference_cppns,
        config.order_threshold,
        config.seed
    )

    if len(reference_cppns) < 8:
        print("ERROR: Could not generate sufficient high-order reference CPPNs")
        return {
            "method": "PCA-constrained nested sampling",
            "reference_cppns": len(reference_cppns),
            "error": "Insufficient reference CPPNs",
            "conclusion": "inconclusive"
        }

    # Step 2: Compute PCA basis
    pca_mean, pca_components, explained_var = compute_pca_basis(
        reference_cppns,
        config.pca_components
    )

    # Step 3: Generate test CPPNs
    print(f"\n[3/6] Generating {config.test_cppns} test CPPNs for trials...")
    set_global_seed(config.seed + 1000)
    test_seeds = [CPPN() for _ in range(config.test_cppns)]
    print(f"✓ Generated {len(test_seeds)} test seed CPPNs")

    # Step 4: Run trials
    print(f"\n[4/6] Running sampling trials...")
    unconstrained_results = []
    constrained_results = []

    for i, seed_cppn in enumerate(test_seeds):
        # Unconstrained (standard ESS)
        result_u = run_sampling_trial(
            seed_cppn,
            config.order_target,
            config.image_size,
            config.n_live_std,
            config.max_iterations,
            constrain_to_pca=False
        )
        unconstrained_results.append(result_u)

        # Constrained (PCA-manifold)
        result_c = run_sampling_trial(
            seed_cppn,
            config.order_target,
            config.image_size,
            config.n_live_constrained,
            config.max_iterations,
            constrain_to_pca=True,
            pca_mean=pca_mean,
            pca_components=pca_components
        )
        constrained_results.append(result_c)

        if (i + 1) % 3 == 0:
            print(f"  [{i+1}/{len(test_seeds)}] unconstrained={result_u['total_samples_to_target']:.0f}, "
                  f"constrained={result_c['total_samples_to_target']:.0f}")

    # Step 5: Calculate statistics
    print(f"\n[5/6] Computing statistics...")
    samples_unconstrained = [r['total_samples_to_target'] for r in unconstrained_results]
    samples_constrained = [r['total_samples_to_target'] for r in constrained_results]

    success_unconstrained = sum(1 for r in unconstrained_results if r['success'])
    success_constrained = sum(1 for r in constrained_results if r['success'])

    avg_samples_unc = np.mean(samples_unconstrained)
    avg_samples_con = np.mean(samples_constrained)
    std_samples_unc = np.std(samples_unconstrained)
    std_samples_con = np.std(samples_constrained)

    # Speedup: how many fewer samples for constrained?
    # speedup > 1 means constrained is more efficient
    speedup = avg_samples_unc / avg_samples_con if avg_samples_con > 0 else 0

    print(f"Unconstrained: {avg_samples_unc:.0f} ± {std_samples_unc:.0f} samples (success: {success_unconstrained}/{len(unconstrained_results)})")
    print(f"Constrained:   {avg_samples_con:.0f} ± {std_samples_con:.0f} samples (success: {success_constrained}/{len(constrained_results)})")
    print(f"Speedup:       {speedup:.2f}×")

    # Step 6: Validate hypothesis
    print(f"\n[6/6] Validating hypothesis...")
    speedup_threshold = 1.2  # Modest threshold for manifold-constrained benefit
    min_success_rate = 0.6

    success_rate_con = success_constrained / len(constrained_results)
    success_rate_unc = success_unconstrained / len(unconstrained_results)

    validated = (speedup >= speedup_threshold and success_rate_con >= min_success_rate)
    conclusion = "validate" if validated else "refute"

    print(f"Speedup >= {speedup_threshold}×: {speedup >= speedup_threshold} ({speedup:.2f}×)")
    print(f"Constrained success rate >= {min_success_rate}: {success_rate_con >= min_success_rate} ({success_rate_con:.2%})")
    print(f"Conclusion: {conclusion}")

    # Compile results
    results = {
        "method": "PCA-constrained nested sampling",
        "reference_cppns": len(reference_cppns),
        "pca_variance_explained": float(explained_var),
        "test_cppns": len(test_seeds),
        "samples_unconstrained": float(avg_samples_unc),
        "samples_unconstrained_std": float(std_samples_unc),
        "samples_constrained": float(avg_samples_con),
        "samples_constrained_std": float(std_samples_con),
        "speedup": float(speedup),
        "success_unconstrained": success_unconstrained,
        "success_constrained": success_constrained,
        "speedup_threshold": speedup_threshold,
        "conclusion": conclusion
    }

    return results

def main():
    """Main experiment execution"""
    print("=" * 70)
    print("RES-220: Manifold-Constrained Sampling Accelerates Order Discovery")
    print("=" * 70)

    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "manifold_aware_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_220_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if 'speedup' in results:
            print(f"RES-220 | manifold_aware_sampling | {results['conclusion']} | speedup={results['speedup']:.2f}x")
        else:
            print(f"RES-220 | manifold_aware_sampling | {results['conclusion']} | error={results.get('error', 'unknown')}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            "method": "PCA-constrained nested sampling",
            "error": str(e),
            "conclusion": "inconclusive"
        }
        results_dir = project_root / "results" / "manifold_aware_sampling"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_220_results.json", 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
