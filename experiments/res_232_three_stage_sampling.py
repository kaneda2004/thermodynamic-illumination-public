#!/usr/bin/env python3
"""
RES-232: Three-Stage Progressive Manifold Sampling (Corrected)

Hypothesis: Progressive constraint tightening (5D→2D PCA basis) achieves ≥2× speedup vs baseline

Method:
1. Run LIVE baseline (single-stage) to get correct sample count.
2. Implement three-stage algorithm:
   - Stage 1: Unconstrained exploration
   - Stage 2: Refine with 5D PCA basis
   - Stage 3: Converge with 2D PCA basis
3. Test 4 variants: (50,50), (75,25), (25,75), (100,0)
4. Compute REAL speedup = baseline_samples / variant_samples
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
    """Configuration for three-stage progressive sampling experiment"""
    test_cppns: int = 20
    order_target: float = 0.50
    n_live_baseline: int = 100
    baseline_max_iter: int = 1000
    
    # Three-stage variant configs: (stage1_budget, stage2_budget)
    variants: list = None

    pca_components_stage2: int = 5
    pca_components_stage3: int = 2
    max_iterations: int = 300
    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.variants is None:
            self.variants = [
                (50, 50),
                (75, 25),
                (25, 75),
                (100, 0),
            ]

def generate_test_cppns(n_samples: int, seed: int = 42) -> list:
    print(f"[1/5] Generating {n_samples} test CPPNs...")
    set_global_seed(seed)
    test_cppns = [CPPN() for _ in range(n_samples)]
    print(f"✓ Generated {len(test_cppns)} test CPPNs")
    return test_cppns

def run_baseline_single_stage(cppn_seed, target_order, image_size, n_live, max_iter) -> dict:
    """Run standard single-stage nested sampling with CORRECT counting."""
    set_global_seed(None)
    
    # Re-initialize from seed for fairness
    # (In a real implementation we'd pass the seed_cppn, but here we just want a fresh run
    # typically from the same random state if set_global_seed was called before)
    # However, to match the structure of other tests, we'll just create new live points.
    
    live_points = []
    best_order = 0.0
    
    for _ in range(n_live):
        c = CPPN()
        img = c.render(image_size)
        o = order_multiplicative(img)
        live_points.append((c, img, o))
        best_order = max(best_order, o)
        
    samples = n_live
    
    if best_order < target_order:
        for i in range(max_iter):
            worst_idx = min(range(n_live), key=lambda x: live_points[x][2])
            threshold = live_points[worst_idx][2]
            seed_idx = np.random.randint(0, n_live)
            
            prop_cppn, prop_img, prop_order, _, _, success = elliptical_slice_sample(
                live_points[seed_idx][0], threshold, image_size, order_multiplicative
            )
            samples += 1
            
            if success:
                live_points[worst_idx] = (prop_cppn, prop_img, prop_order)
                best_order = max(best_order, prop_order)
                
            if best_order >= target_order:
                break
                
    return {
        'total_samples': samples,
        'success': best_order >= target_order
    }

def compute_pca_basis_from_samples(weights_samples: list, n_components: int) -> tuple:
    if len(weights_samples) < 2:
        return None, None, 0.0
    W = np.array(weights_samples)
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    n_comp = min(n_components, len(S))
    components = Vt[:n_comp]
    return W_mean, components, 0.0

def run_three_stage_sampling(
    seed_cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    stage2_budget: int,
    max_iterations: int = 300,
    pca_components_stage2: int = 5,
    pca_components_stage3: int = 2
) -> dict:
    set_global_seed(None)

    # Stage 1: Unconstrained
    n_live_s1 = 50
    live_points = []
    best_order = 0
    collected_weights = []
    
    for _ in range(n_live_s1):
        c = CPPN()
        img = c.render(image_size)
        o = order_multiplicative(img)
        live_points.append((c, img, o))
        collected_weights.append(c.get_weights())
        best_order = max(best_order, o)
        
    total_samples = n_live_s1
    samples_at_target = None
    if best_order >= target_order: samples_at_target = total_samples
    
    # Run Stage 1 budget
    extra_s1 = max(0, stage1_budget - n_live_s1)
    for _ in range(extra_s1):
        worst_idx = min(range(n_live_s1), key=lambda i: live_points[i][2])
        threshold = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live_s1)
        
        prop_c, prop_img, prop_o, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )
        total_samples += 1
        
        if success:
            live_points[worst_idx] = (prop_c, prop_img, prop_o)
            best_order = max(best_order, prop_o)
            collected_weights.append(prop_c.get_weights())
            
        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples

    # Stage 2: 5D PCA
    if samples_at_target is None and len(collected_weights) > 2:
        mean2, comps2, _ = compute_pca_basis_from_samples(collected_weights, pca_components_stage2)
        if mean2 is not None:
            for _ in range(stage2_budget):
                worst_idx = min(range(n_live_s1), key=lambda i: live_points[i][2])
                threshold = live_points[worst_idx][2]
                seed_idx = np.random.randint(0, n_live_s1)
                
                curr_w = live_points[seed_idx][0].get_weights()
                coeffs = comps2 @ (curr_w - mean2)
                delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                prop_w = mean2 + comps2.T @ (coeffs + delta)
                
                c = live_points[seed_idx][0].copy()
                c.set_weights(prop_w)
                img = c.render(image_size)
                o = order_multiplicative(img)
                total_samples += 1
                
                if o > threshold:
                    live_points[worst_idx] = (c, img, o)
                    best_order = max(best_order, o)
                    # Keep collecting weights for Stage 3? Or just use Stage 1 weights?
                    # Typically progressive implies refining the basis. Let's append successful ones.
                    collected_weights.append(prop_w)
                    
                if best_order >= target_order and samples_at_target is None:
                    samples_at_target = total_samples

    # Stage 3: 2D PCA (Tighter)
    if samples_at_target is None and len(collected_weights) > 2:
        mean3, comps3, _ = compute_pca_basis_from_samples(collected_weights, pca_components_stage3)
        if mean3 is not None:
            for _ in range(max_iterations):
                worst_idx = min(range(n_live_s1), key=lambda i: live_points[i][2])
                threshold = live_points[worst_idx][2]
                seed_idx = np.random.randint(0, n_live_s1)
                
                curr_w = live_points[seed_idx][0].get_weights()
                coeffs = comps3 @ (curr_w - mean3)
                delta = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.2  # Tighter noise
                prop_w = mean3 + comps3.T @ (coeffs + delta)
                
                c = live_points[seed_idx][0].copy()
                c.set_weights(prop_w)
                img = c.render(image_size)
                o = order_multiplicative(img)
                total_samples += 1
                
                if o > threshold:
                    live_points[worst_idx] = (c, img, o)
                    best_order = max(best_order, o)
                    
                if best_order >= target_order and samples_at_target is None:
                    samples_at_target = total_samples
                    break

    if samples_at_target is None: samples_at_target = total_samples
    
    return {
        'samples_to_target': samples_at_target,
        'max_order': float(best_order),
        'success': best_order >= target_order
    }

def run_experiment(config: ExperimentConfig) -> dict:
    test_cppns = generate_test_cppns(config.test_cppns, config.seed)

    # 1. Run Baseline
    print(f"\n[2/5] Running baseline single-stage sampling...")
    baseline_samples = []
    for i, cppn in enumerate(test_cppns):
        res = run_baseline_single_stage(cppn, config.order_target, config.image_size, 
                                      config.n_live_baseline, config.baseline_max_iter)
        baseline_samples.append(res['total_samples'])
        if (i+1) % 5 == 0: print(f"  [{i+1}/{config.test_cppns}] baseline: {res['total_samples']}")
        
avg_baseline = np.mean(baseline_samples)
    print(f"✓ Baseline: {avg_baseline:.0f} ± {np.std(baseline_samples):.0f} samples")

    # 2. Run Variants
    variant_results = {}
    for variant_idx, (stage1_N, stage2_N) in enumerate(config.variants):
        print(f"\n  [{variant_idx+1}/{len(config.variants)}] Testing ({stage1_N}, {stage2_N})...")
        variant_samples = []
        for i, cppn in enumerate(test_cppns):
            res = run_three_stage_sampling(cppn, config.order_target, config.image_size,
                                         stage1_N, stage2_N, config.max_iterations,
                                         config.pca_components_stage2, config.pca_components_stage3)
            variant_samples.append(res['samples_to_target'])
            
avg_var = np.mean(variant_samples)
speedup = avg_baseline / avg_var if avg_var > 0 else 0
        
        variant_results[f"{stage1_N}_{stage2_N}"] = {
            'avg_samples': float(avg_var),
            'speedup': float(speedup)
        }
        print(f"  ✓ Speedup: {speedup:.2f}x")

    # Best
    best_speedup = max(v['speedup'] for v in variant_results.values())
    best_key = [k for k,v in variant_results.items() if v['speedup'] == best_speedup][0]
    
    return {
        "best_speedup": float(best_speedup),
        "best_variant": best_key,
        "variants": variant_results,
        "baseline_samples": float(avg_baseline),
        "conclusion": "validate" if best_speedup >= 2.0 else "refute"
    }

def main():
    print("="*70)
    print("RES-232: Three-Stage Progressive (Corrected)")
    print("="*70)
    
    config = ExperimentConfig()
    results = run_experiment(config)
    
    # Save
    results_dir = project_root / "results" / "progressive_manifold_sampling"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "res_232_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✓ Results saved. Best speedup: {results['best_speedup']:.2f}x")

if __name__ == "__main__":
    main()