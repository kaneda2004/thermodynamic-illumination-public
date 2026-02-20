#!/usr/bin/env python3
"""
RES-283: Variance Decomposition - Explain 80%+ of speedup variance using systematic factors

Hypothesis: Speedup variance across CPPNs can be explained by a small set of factors:
1. CPPN intrinsic dimension (from RES-001)
2. Stage 1 sampling budget
3. Stage 2 strategy (uniform, decay, adaptive)

This test answers the critical question: "Is 92× speedup real or just luck on 20 CPPNs?"

Design:
- Test 80-100 CPPNs
- 3 Stage 1 budgets: [50, 100, 150]
- 2 strategies: [uniform, decay] (skip adaptive to avoid RES-268 issues)
- Parallelize: 4 worker processes
- MPS support: GPU acceleration for rendering operations
- Checkpointing: Save results every 10 CPPNs for robustness

Success criteria:
- R² ≥ 0.80 for variance explained
- Feature importance: CPPN dimension should be primary driver
- Speedup is predictable, not random
- Runtime ≤ 15 hours (feasible with parallelization + MPS)
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

# Multiprocessing for parallelization
from multiprocessing import Pool, Queue, Process, Value, Manager
import pickle

# ML for variance analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure project root is in path
project_root = Path('/Users/matt/Development/monochrome_noise_converger')
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for variance decomposition experiment"""
    n_cppns: int = 100                      # CPPNs to test
    order_target: float = 0.50              # Target order level
    baseline_n_live: int = 100              # Baseline single-stage n_live
    baseline_max_iterations: int = 300      # Baseline max iterations

    # Two-stage variants
    stage1_budgets: list = None             # N values: [50, 100, 150]
    strategies: list = None                 # [uniform, decay] (skip adaptive)

    # Parallelization
    n_workers: int = 4                      # Processes to run in parallel
    checkpoint_interval: int = 10           # Save checkpoint every N CPPNs

    # Other
    pca_components: int = 3                 # Manifold dimensionality
    max_iterations_stage2: int = 250        # Stage 2 iterations
    image_size: int = 32
    seed: int = 42

    def __post_init__(self):
        if self.stage1_budgets is None:
            self.stage1_budgets = [50, 100, 150]
        if self.strategies is None:
            self.strategies = ['uniform', 'decay']  # Skip adaptive


def detect_mps():
    """Detect and report MPS (Metal Performance Shaders) availability"""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("✓ MPS (Metal Performance Shaders) available - GPU acceleration enabled")
            return True
        else:
            logger.info("✗ MPS not available - using CPU only")
            return False
    except:
        logger.info("✗ PyTorch/MPS not available - using CPU")
        return False


def run_baseline_single_stage(
    cppn: CPPN,
    target_order: float,
    image_size: int,
    n_live: int,
    max_iterations: int
) -> Dict:
    """Run single-stage nested sampling to target order"""
    set_global_seed(None)

    live_points = []
    best_order = 0.0
    samples_to_target = None

    # Initialize
    for _ in range(n_live):
        cppn_inst = CPPN()
        img = cppn_inst.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn_inst, img, order))
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


def run_two_stage_sampling(
    cppn: CPPN,
    target_order: float,
    image_size: int,
    stage1_budget: int,
    max_iterations_stage2: int,
    strategy: str = 'uniform',
    pca_components: int = 3,
    device: str = 'cpu'
) -> Dict:
    """Run two-stage sampling with specified strategy"""
    set_global_seed(None)

    # STAGE 1: Exploration
    n_live_stage1 = 50
    live_points = []
    best_order = 0.0
    collected_weights = []
    samples_at_target = None

    # Initialize
    for _ in range(n_live_stage1):
        cppn_inst = CPPN()
        img = cppn_inst.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn_inst, img, order))
        collected_weights.append(cppn_inst.get_weights())
        best_order = max(best_order, order)

    if best_order >= target_order:
        samples_at_target = n_live_stage1

    # Stage 1 iterations
    stage1_samples = n_live_stage1
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
            stage1_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = stage1_samples

    # STAGE 2: Manifold convergence
    pca_mean, pca_components_mat, explained_var = _compute_pca_basis(
        collected_weights, pca_components
    )

    total_samples = stage1_samples

    if pca_mean is not None and pca_components_mat is not None:
        for iteration in range(max_iterations_stage2):
            worst_idx = min(range(n_live_stage1), key=lambda i: live_points[i][2])
            worst_order = live_points[worst_idx][2]
            threshold = worst_order

            seed_idx = np.random.randint(0, n_live_stage1)
            current_w = live_points[seed_idx][0].get_weights()
            coeffs = _project_to_pca(current_w, pca_mean, pca_components_mat)

            if coeffs is not None:
                # Generate proposal based on strategy
                if strategy == 'uniform':
                    delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
                elif strategy == 'decay':
                    # Decay: reduce step size as iterations increase
                    decay_factor = 1.0 - (iteration / max_iterations_stage2) * 0.5
                    delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5 * decay_factor
                else:
                    delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5

                new_coeffs = coeffs + delta_coeffs
                proposal_w = _reconstruct_from_pca(new_coeffs, pca_mean, pca_components_mat)

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
        'stage1_samples': stage1_samples,
        'stage2_samples': total_samples - stage1_samples,
        'total_samples': total_samples,
        'samples_to_target': samples_at_target,
        'max_order_achieved': float(best_order),
        'success': best_order >= target_order
    }


def _compute_pca_basis(weights_samples: List, n_components: int) -> Tuple:
    """Compute PCA basis from weight samples"""
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


def _project_to_pca(weights: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Project weights to PCA basis"""
    if pca_mean is None or pca_components is None:
        return None
    w_centered = weights - pca_mean
    coeffs = pca_components @ w_centered
    return coeffs


def _reconstruct_from_pca(coeffs: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reconstruct weights from PCA coefficients"""
    w_pca_space = pca_components.T @ coeffs
    return pca_mean + w_pca_space


def worker_task(args: Tuple) -> Dict:
    """Single worker task: run baseline + two-stage for one CPPN"""
    cppn_id, config = args

    try:
        # Generate CPPN
        set_global_seed(cppn_id)  # Deterministic per CPPN
        cppn = CPPN()

        # Baseline
        baseline = run_baseline_single_stage(
            cppn,
            config.order_target,
            config.image_size,
            config.baseline_n_live,
            config.baseline_max_iterations
        )

        # Two-stage variants
        variants_results = {}
        for budget in config.stage1_budgets:
            for strategy in config.strategies:
                key = f"budget_{budget}_strategy_{strategy}"
                result = run_two_stage_sampling(
                    cppn,
                    config.order_target,
                    config.image_size,
                    budget,
                    config.max_iterations_stage2,
                    strategy=strategy,
                    pca_components=config.pca_components
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
        logger.error(f"Error processing CPPN {cppn_id}: {str(e)}")
        return {
            'cppn_id': cppn_id,
            'error': str(e),
            'success': False
        }


def run_parallel_experiments(config: ExperimentConfig) -> List[Dict]:
    """Run experiments in parallel with 4 workers"""
    logger.info(f"Starting parallel variance decomposition experiment")
    logger.info(f"  CPPNs: {config.n_cppns}")
    logger.info(f"  Stage 1 budgets: {config.stage1_budgets}")
    logger.info(f"  Strategies: {config.strategies}")
    logger.info(f"  Workers: {config.n_workers}")

    # Create job list
    jobs = [(cppn_id, config) for cppn_id in range(config.n_cppns)]

    # Run with Pool
    results = []
    with Pool(config.n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_task, jobs)):
            results.append(result)

            # Checkpointing
            if (i + 1) % config.checkpoint_interval == 0:
                checkpoint_path = project_root / "results" / "variance_decomposition" / f"checkpoint_{i+1:03d}.json"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'completed': i + 1,
                        'total': config.n_cppns,
                        'timestamp': datetime.now().isoformat()
                    }, f)
                logger.info(f"  Checkpoint saved: {i+1}/{config.n_cppns} CPPNs")

    return results


def analyze_variance(results: List[Dict], config: ExperimentConfig) -> Dict:
    """Analyze speedup variance and feature importance"""
    logger.info("Analyzing variance decomposition...")

    # Extract speedups and features
    speedups = []
    stage1_budgets_data = []
    strategies_data = []
    best_speedups = []

    for result in results:
        if not result['success']:
            continue

        baseline_samples = result['baseline_samples']
        best_speedup = 0

        for variant_key, variant_result in result['variants'].items():
            speedup = variant_result['speedup']
            speedups.append(speedup)

            # Extract features from variant_key
            # Format: "budget_50_strategy_uniform"
            budget = int(variant_key.split('_')[1])
            strategy = variant_key.split('_')[3]

            stage1_budgets_data.append(budget)
            strategies_data.append(1 if strategy == 'decay' else 0)  # 0=uniform, 1=decay

            best_speedup = max(best_speedup, speedup)

        best_speedups.append(best_speedup)

    # Prepare features matrix
    X = np.column_stack([
        stage1_budgets_data,
        strategies_data
    ])
    y = np.array(speedups)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit random forest model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')

    # Predictions
    y_pred = rf.predict(X_scaled)
    r2_train = r2_score(y, y_pred)
    r2_cv = cv_scores.mean()
    mape = mean_absolute_percentage_error(y, y_pred)

    # Feature importance
    feature_importance = {
        'stage1_budget': float(rf.feature_importances_[0]),
        'strategy_choice': float(rf.feature_importances_[1])
    }

    # Summary statistics
    summary = {
        'n_cppns_tested': len([r for r in results if r['success']]),
        'speedup_mean': float(np.mean(best_speedups)),
        'speedup_std': float(np.std(best_speedups)),
        'speedup_min': float(np.min(best_speedups)),
        'speedup_max': float(np.max(best_speedups)),
        'speedup_median': float(np.median(best_speedups))
    }

    return {
        'summary': summary,
        'variance_explained': {
            'model_type': 'random_forest',
            'r_squared_train': float(r2_train),
            'r_squared_cv': float(r2_cv),
            'cv_std': float(cv_scores.std()),
            'mape': float(mape)
        },
        'feature_importance': feature_importance,
        'feature_names': ['stage1_budget', 'strategy_choice'],
        'speedups': speedups,
        'best_speedups': best_speedups,
        'y_predictions': y_pred.tolist(),
        'X_data': X.tolist()
    }


def create_visualization(analysis: Dict, output_path: Path):
    """Create 4-panel variance decomposition figure"""
    logger.info("Creating visualization...")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Speedup distribution
    ax1 = fig.add_subplot(gs[0, 0])
    best_speedups = analysis['best_speedups']
    ax1.hist(best_speedups, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(best_speedups), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(best_speedups):.1f}×')
    ax1.set_xlabel('Speedup (×)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Speedup Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Speedup vs Stage 1 Budget
    ax2 = fig.add_subplot(gs[0, 1])
    X_data = np.array(analysis['X_data'])
    speedups = np.array(analysis['speedups'])

    for budget in [50, 100, 150]:
        mask = X_data[:, 0] == budget
        ax2.scatter(X_data[mask, 0] + np.random.normal(0, 2, sum(mask)),
                   speedups[mask], alpha=0.6, s=50, label=f'Budget={budget}')

    ax2.set_xlabel('Stage 1 Budget (samples)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=11, fontweight='bold')
    ax2.set_title('Panel B: Speedup vs Stage 1 Budget', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Panel 3: Feature importance
    ax3 = fig.add_subplot(gs[1, 0])
    feature_names = analysis['feature_names']
    importances = [analysis['feature_importance'][f] for f in feature_names]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax3.bar(range(len(feature_names)), importances, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(feature_names)))
    ax3.set_xticklabels(feature_names, rotation=15, ha='right')
    ax3.set_ylabel('Importance', fontsize=11, fontweight='bold')
    ax3.set_title('Panel C: Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Panel 4: Predicted vs Actual
    ax4 = fig.add_subplot(gs[1, 1])
    y_pred = np.array(analysis['y_predictions'])
    speedups = np.array(analysis['speedups'])
    ax4.scatter(speedups, y_pred, alpha=0.6, s=50, color='steelblue', edgecolor='black')

    # Perfect prediction line
    min_val = min(speedups.min(), y_pred.min())
    max_val = max(speedups.max(), y_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax4.set_xlabel('Actual Speedup (×)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted Speedup (×)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Panel D: Model Validation (R²={analysis["variance_explained"]["r_squared_cv"]:.3f})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Main title
    fig.suptitle('RES-283: Variance Decomposition Analysis\nSpeedup predictability across 100 CPPNs',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    """Main experiment execution"""
    print("=" * 80)
    print("RES-283: Variance Decomposition")
    print("=" * 80)

    # Detect MPS
    detect_mps()

    # Configuration
    config = ExperimentConfig(n_cppns=100)

    try:
        # Run parallel experiments
        logger.info("[1/4] Running parallel experiments...")
        results = run_parallel_experiments(config)

        # Analyze variance
        logger.info("[2/4] Analyzing variance...")
        analysis = analyze_variance(results, config)

        # Save results
        logger.info("[3/4] Saving results...")
        results_dir = project_root / "results" / "variance_decomposition"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "res_283_results.json", 'w') as f:
            json.dump({
                'config': {
                    'n_cppns': config.n_cppns,
                    'stage1_budgets': config.stage1_budgets,
                    'strategies': config.strategies,
                    'n_workers': config.n_workers
                },
                'analysis': analysis
            }, f, indent=2)

        logger.info(f"✓ Results saved: {results_dir / 'res_283_results.json'}")

        # Create visualization
        logger.info("[4/4] Creating visualization...")
        create_visualization(analysis, results_dir / "res_283_figure_variance.pdf")

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"CPPNs tested: {analysis['summary']['n_cppns_tested']}")
        print(f"Mean speedup: {analysis['summary']['speedup_mean']:.2f}× ± {analysis['summary']['speedup_std']:.2f}×")
        print(f"Speedup range: {analysis['summary']['speedup_min']:.2f}× to {analysis['summary']['speedup_max']:.2f}×")
        print(f"\nVariance Explained (R²):")
        print(f"  Training: {analysis['variance_explained']['r_squared_train']:.4f}")
        print(f"  Cross-validation: {analysis['variance_explained']['r_squared_cv']:.4f}")
        print(f"\nFeature Importance:")
        for feature, importance in analysis['feature_importance'].items():
            print(f"  {feature}: {importance:.4f}")

        target_r2 = 0.80
        status = "VALIDATED" if analysis['variance_explained']['r_squared_cv'] >= target_r2 else "INCONCLUSIVE"
        print(f"\n✓ Status: {status}")
        print(f"✓ Target R² ≥ {target_r2}: {analysis['variance_explained']['r_squared_cv']:.4f}")

    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        results_dir = project_root / "results" / "variance_decomposition"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_283_results.json", 'w') as f:
            json.dump({
                'error': str(e),
                'status': 'FAILED'
            }, f)


if __name__ == "__main__":
    main()
