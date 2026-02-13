#!/usr/bin/env python3
"""
RES-283 MINIMAL TEST VERSION
- 10 CPPNs instead of 100
- 2 Stage 1 budgets instead of 3
- 1 strategy instead of 2
- Total: ~10 min runtime on e2-standard-4 ($0.01 cost)

Use this to verify the pipeline works before running full 100-CPPN version.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from multiprocessing import Pool

# ML for variance analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure project root is in path (works on both local and GCP)
local_path = Path('/Users/matt/Development/monochrome_noise_converger')
if local_path.exists():
    project_root = local_path
else:
    # On GCP, use current working directory (should be ~/repo)
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Project imports
from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    n_cppns: int = 10                       # REDUCED: 10 instead of 100
    order_target: float = 0.40              # FIXED: Achievable threshold for real speedup
    baseline_n_live: int = 100
    baseline_max_iterations: int = 1000     # FIXED: Enough iterations to reach target
    stage1_budgets: list = None             # [50, 100] instead of [50, 100, 150]
    strategies: list = None                 # ['uniform'] instead of ['uniform', 'decay']
    n_workers: int = 4
    checkpoint_interval: int = 5
    pca_components: int = 3
    max_iterations_stage2: int = 500        # FIXED: More Stage 2 iterations
    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.stage1_budgets is None:
            self.stage1_budgets = [50, 100]
        if self.strategies is None:
            self.strategies = ['uniform']


def detect_mps():
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("✓ MPS available - GPU acceleration enabled")
            return True
    except:
        pass
    logger.info("✗ MPS not available - using CPU")
    return False


def run_baseline_single_stage(target_order, image_size, n_live, max_iterations) -> Dict:
    # Create diverse live points by sampling from prior (new random CPPNs)
    # This is how nested sampling works - exploring the prior to find high-order regions
    live_points = []
    best_order = 0.0
    samples_to_target = None

    for _ in range(n_live):
        cppn_inst = CPPN()  # New random CPPN from prior - essential for NS diversity
        img = cppn_inst.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn_inst, img, order))
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_to_target = n_live

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


def run_two_stage_sampling(target_order, image_size, stage1_budget, max_iterations_stage2, strategy='uniform', pca_components=3) -> Dict:
    # Create diverse live points by sampling from prior (new random CPPNs)
    n_live_stage1 = 50
    live_points = []
    best_order = 0.0
    collected_weights = []
    samples_at_target = None

    for _ in range(n_live_stage1):
        cppn_inst = CPPN()  # New random CPPN from prior - essential for NS diversity
        img = cppn_inst.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn_inst, img, order))
        collected_weights.append(cppn_inst.get_weights())
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    stage1_samples = n_live_stage1
    # Stage 1 iterations: total budget minus initial live points
    # budget=50 → 0 extra iterations, budget=100 → 50 iterations
    for iteration in range(max(0, stage1_budget - n_live_stage1)):
        worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        seed_idx = np.random.randint(0, n_live_stage1)

        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], worst_order, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            stage1_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples

    # PCA
    if len(collected_weights) >= 2:
        W = np.array(collected_weights)
        W_mean = W.mean(axis=0)
        W_centered = W - W_mean
        U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
        n_comp = min(pca_components, len(S))
        pca_components_mat = Vt[:n_comp]
        pca_mean = W_mean
    else:
        pca_mean = pca_components_mat = None

    total_samples = stage1_samples

    if pca_mean is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            seed_idx = np.random.randint(0, n_live_stage1)

            current_w = live_points[seed_idx][0].get_weights()
            w_centered = current_w - pca_mean
            coeffs = pca_components_mat @ w_centered

            delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
            new_coeffs = coeffs + delta_coeffs
            proposal_w = pca_mean + (pca_components_mat.T @ new_coeffs)

            proposal_cppn = live_points[seed_idx][0].copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_multiplicative(proposal_img)

            if proposal_order >= worst_order:
                live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
                best_order = max(best_order, proposal_order)

            total_samples += 1

            if best_order >= target_order and samples_at_target is None:
                samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    return {
        'stage1_samples': stage1_samples,
        'stage2_samples': total_samples - stage1_samples,
        'total_samples': total_samples,
        'samples_to_target': samples_at_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }


def worker_task(args: Tuple) -> Dict:
    cppn_id, config = args

    try:
        set_global_seed(cppn_id)

        baseline = run_baseline_single_stage(
            config.order_target, config.image_size,
            config.baseline_n_live, config.baseline_max_iterations
        )

        variants_results = {}
        for budget in config.stage1_budgets:
            for strategy in config.strategies:
                key = f"budget_{budget}_strategy_{strategy}"
                result = run_two_stage_sampling(
                    config.order_target, config.image_size,
                    budget, config.max_iterations_stage2, strategy=strategy
                )
                result['speedup'] = baseline['total_samples'] / result['samples_to_target'] if result['samples_to_target'] > 0 else 0
                variants_results[key] = result

        return {
            'cppn_id': cppn_id,
            'baseline_samples': baseline['total_samples'],
            'variants': variants_results,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in CPPN {cppn_id}: {str(e)}")
        return {'cppn_id': cppn_id, 'error': str(e), 'success': False}


def run_parallel_experiments(config: ExperimentConfig) -> List[Dict]:
    logger.info(f"TEST RUN: {config.n_cppns} CPPNs, {len(config.stage1_budgets)} budgets, {len(config.strategies)} strategies")
    logger.info(f"Total combinations: ~{config.n_cppns * len(config.stage1_budgets) * len(config.strategies)}")

    jobs = [(cppn_id, config) for cppn_id in range(config.n_cppns)]
    results = []

    with Pool(config.n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_task, jobs)):
            results.append(result)
            if (i + 1) % config.checkpoint_interval == 0:
                logger.info(f"  Progress: {i+1}/{config.n_cppns} CPPNs")

    return results


def analyze_variance(results: List[Dict], config: ExperimentConfig) -> Dict:
    speedups = []
    stage1_budgets_data = []
    strategy_data = []
    cppn_ids = []
    best_speedups = []
    variant_details = []

    for result in results:
        if not result['success']:
            continue

        best_speedup = 0
        cppn_id = result['cppn_id']

        for variant_key, variant_result in result['variants'].items():
            speedup = variant_result['speedup']
            speedups.append(speedup)

            # Parse variant_key: "budget_50_strategy_uniform"
            parts = variant_key.split('_')
            budget = int(parts[1])
            strategy = parts[3]

            stage1_budgets_data.append(budget)
            strategy_data.append(1 if strategy == 'decay' else 0)  # 0=uniform, 1=decay
            cppn_ids.append(cppn_id)

            variant_details.append({
                'cppn_id': cppn_id,
                'budget': budget,
                'strategy': strategy,
                'speedup': speedup
            })

            best_speedup = max(best_speedup, speedup)

        best_speedups.append(best_speedup)

    # Build feature matrix with more features
    X = np.column_stack([
        stage1_budgets_data,    # Feature 1: Stage 1 budget
        strategy_data,          # Feature 2: Strategy (0=uniform, 1=decay)
        cppn_ids                # Feature 3: CPPN ID - captures intrinsic difficulty
    ])
    y = np.array(speedups)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)

    cv_scores = cross_val_score(rf, X_scaled, y, cv=3, scoring='r2')
    y_pred = rf.predict(X_scaled)
    r2_train = r2_score(y, y_pred)
    r2_cv = cv_scores.mean()

    return {
        'summary': {
            'n_cppns_tested': len([r for r in results if r['success']]),
            'speedup_mean': float(np.mean(best_speedups)),
            'speedup_std': float(np.std(best_speedups)) if len(best_speedups) > 1 else 0,
            'speedup_min': float(np.min(best_speedups)),
            'speedup_max': float(np.max(best_speedups))
        },
        'variance_explained': {
            'r_squared_train': float(r2_train),
            'r_squared_cv': float(r2_cv),
            'cv_std': float(cv_scores.std())
        },
        'feature_importance': {
            'stage1_budget': float(rf.feature_importances_[0]),
            'strategy': float(rf.feature_importances_[1]),
            'cppn_id': float(rf.feature_importances_[2])
        },
        'speedups': speedups,
        'best_speedups': best_speedups,
        'variant_details': variant_details
    }


def main():
    print("=" * 80)
    print("RES-283 MINIMAL TEST VERSION (10 CPPNs, ~10 min runtime)")
    print("=" * 80)

    detect_mps()

    config = ExperimentConfig()

    try:
        logger.info("[1/3] Running test experiments...")
        results = run_parallel_experiments(config)

        logger.info("[2/3] Analyzing results...")
        analysis = analyze_variance(results, config)

        logger.info("[3/3] Saving results...")
        results_dir = project_root / "results" / "variance_decomposition"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "res_283_test_results.json", 'w') as f:
            json.dump({'config': {'n_cppns': config.n_cppns}, 'analysis': analysis}, f, indent=2)

        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"CPPNs tested: {analysis['summary']['n_cppns_tested']}")
        print(f"Mean speedup: {analysis['summary']['speedup_mean']:.2f}×")
        print(f"R² (CV): {analysis['variance_explained']['r_squared_cv']:.4f}")
        print(f"\n✓ Test successful! Ready to run full 100-CPPN version")

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
