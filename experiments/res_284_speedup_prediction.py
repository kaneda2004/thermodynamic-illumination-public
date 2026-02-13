#!/usr/bin/env python3
"""
RES-284: Feature-Based Speedup Prediction (Parallelized)
Goal: Predict speedup using intrinsic CPPN features to solve the negative RÂ² issue in RES-283.

Hypothesis: Speedup is not random but determined by specific geometric and spectral properties
of the CPPN initialization (e.g., spectral slope, local roughness, gradient magnitude).

Method:
1. Generate 100 CPPNs.
2. Compute predictive features for each CPPN (before extensive sampling).
3. Run corrected Two-Stage Sampling to measure ACTUAL speedup.
4. Train Random Forest (Features -> Speedup).
5. Validate if RÂ² > 0.
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
from multiprocessing import Pool
from scipy import ndimage

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Ensure project root is in path (works on both local and GCP)
# Force CWD for batch execution to avoid any path confusion
project_root = Path.cwd()

sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Project imports
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample, set_global_seed, PRIOR_SIGMA
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    n_cppns: int = 100                      # 100 for robust statistics
    image_size: int = 32
    target_order: float = 0.40              # Fixed achievable target
    n_workers: int = 32                     # Optimized for n2-standard-32
    baseline_n_live: int = 100
    baseline_max_iter: int = 1000
    stage1_budget: int = 100
    stage2_iter: int = 500
    pca_components: int = 3
    checkpoint_interval: int = 10

def compute_features(cppn: CPPN, image_size: int) -> Dict[str, float]:
    """Compute intrinsic predictive features from CPPN initialization."""
    img = cppn.render(image_size)
    
    # 1. Basic Statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    # 2. Initial Order
    initial_order = order_multiplicative(img)
    
    # 3. Gradient Magnitude (Roughness)
    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)
    mean_grad = np.mean(grad_mag)
    
    # 4. Spectral Features (FFT)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    # Radial profile
    y, x = np.indices((image_size, image_size))
    center = np.array([image_size//2, image_size//2])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    # Simple spectral slope proxy (ratio of low freq to high freq energy)
    low_freq_energy = np.sum(radial_profile[:image_size//4])
    high_freq_energy = np.sum(radial_profile[image_size//4:])
    spectral_ratio = low_freq_energy / (high_freq_energy + 1e-8)
    
    return {
        'initial_mean': float(mean_val),
        'initial_std': float(std_val),
        'initial_order': float(initial_order),
        'mean_gradient': float(mean_grad),
        'spectral_ratio': float(spectral_ratio)
    }

def run_sampling_task(args):
    cppn_id, config = args
    try:
        set_global_seed(cppn_id)
        
        # Generate CPPN
        cppn = CPPN()
        
        # 1. Compute Features
        features = compute_features(cppn, config.image_size)
        
        # 2. Run Baseline (Corrected Counting)
        # ---------------------------------------------------------
        n_live = config.baseline_n_live
        live_points = []
        best_order = 0.0
        
        # Init live points
        for _ in range(n_live):
            c = CPPN()
            img = c.render(config.image_size)
            o = order_multiplicative(img)
            live_points.append((c, img, o))
            best_order = max(best_order, o)
            
        baseline_samples = n_live
        if best_order < config.target_order:
            for i in range(config.baseline_max_iter):
                worst_idx = min(range(n_live), key=lambda x: live_points[x][2])
                threshold = live_points[worst_idx][2]
                seed_idx = np.random.randint(0, n_live)
                
                prop_cppn, prop_img, prop_order, _, _, success = elliptical_slice_sample(
                    live_points[seed_idx][0], threshold, config.image_size, order_multiplicative
                )
                
                baseline_samples += 1  # Correct counting
                
                if success:
                    live_points[worst_idx] = (prop_cppn, prop_img, prop_order)
                    best_order = max(best_order, prop_order)
                    
                if best_order >= config.target_order:
                    break
        
        # 3. Run Two-Stage (Corrected Counting & Decay Strategy)
        # ---------------------------------------------------------
        # Re-seed for fair comparison
        set_global_seed(cppn_id) 
        cppn = CPPN() 
        
        # Stage 1: Exploration
        n_live_s1 = 50
        live_s1 = []
        collected_w = []
        best_order_s2 = 0.0
        
        for _ in range(n_live_s1):
            c = CPPN()
            img = c.render(config.image_size)
            o = order_multiplicative(img)
            live_s1.append((c, img, o))
            collected_w.append(c.get_weights())
            best_order_s2 = max(best_order_s2, o)
            
        stage1_samples = n_live_s1
        
        # Run Stage 1 budget
        extra_s1 = max(0, config.stage1_budget - n_live_s1)
        for _ in range(extra_s1):
            if best_order_s2 >= config.target_order: break
            
            worst_idx = min(range(n_live_s1), key=lambda x: live_s1[x][2])
            threshold = live_s1[worst_idx][2]
            seed_idx = np.random.randint(0, n_live_s1)
            
            prop_cppn, prop_img, prop_order, _, _, success = elliptical_slice_sample(
                live_s1[seed_idx][0], threshold, config.image_size, order_multiplicative
            )
            stage1_samples += 1
            
            if success:
                live_s1[worst_idx] = (prop_cppn, prop_img, prop_order)
                best_order_s2 = max(best_order_s2, prop_order)
                collected_w.append(prop_cppn.get_weights())

        # Stage 2: PCA Constrained
        stage2_samples = 0
        if best_order_s2 < config.target_order and len(collected_w) > 2:
            # Compute PCA
            W = np.array(collected_w)
            W_mean = W.mean(axis=0)
            W_centered = W - W_mean
            U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
            comps = Vt[:config.pca_components]
            
            for i in range(config.stage2_iter):
                if best_order_s2 >= config.target_order: break
                
                worst_idx = min(range(n_live_s1), key=lambda x: live_s1[x][2])
                threshold = live_s1[worst_idx][2]
                seed_idx = np.random.randint(0, n_live_s1)
                
                current_w = live_s1[seed_idx][0].get_weights()
                coeffs = comps @ (current_w - W_mean)
                
                # Decay strategy logic
                decay = 1.0 - (i / config.stage2_iter)
                sigma = PRIOR_SIGMA * 0.5 * max(0.1, decay)
                
                delta = np.random.randn(len(coeffs)) * sigma
                prop_w = W_mean + comps.T @ (coeffs + delta)
                
                c = live_s1[seed_idx][0].copy()
                c.set_weights(prop_w)
                img = c.render(config.image_size)
                o = order_multiplicative(img)
                
                stage2_samples += 1
                
                if o > threshold:
                    live_s1[worst_idx] = (c, img, o)
                    best_order_s2 = max(best_order_s2, o)

        total_s2_samples = stage1_samples + stage2_samples
        speedup = baseline_samples / total_s2_samples if total_s2_samples > 0 else 0.0
        
        return {
            'cppn_id': cppn_id,
            'features': features,
            'speedup': speedup,
            'baseline_samples': int(baseline_samples),
            'two_stage_samples': int(total_s2_samples),
            'success': bool(best_order_s2 >= config.target_order)
        }
    except Exception as e:
        return {'cppn_id': cppn_id, 'error': str(e), 'success': False}

def run_experiment():
    config = Config()
    
    # Ensure results directory exists
    results_dir = project_root / "results" / "variance_decomposition"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"RES-284: Feature-Based Speedup Prediction (n={config.n_cppns})")
    logger.info("="*60)
    
    jobs = [(i, config) for i in range(config.n_cppns)]
    results = []
    
    with Pool(config.n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_sampling_task, jobs)):
            results.append(result)
            if (i + 1) % config.checkpoint_interval == 0:
                logger.info(f"  Progress: {i+1}/{config.n_cppns} CPPNs")
        
    # Analyze
    valid_results = [r for r in results if r['success']]
    logger.info(f"Completed {len(valid_results)}/{config.n_cppns} successful runs")
    
    if not valid_results:
        logger.error("No successful runs")
        return

    # Prepare ML Data
    X = []
    y = []
    feature_names = list(valid_results[0]['features'].keys())
    
    for r in valid_results:
        feats = [r['features'][k] for k in feature_names]
        X.append(feats)
        y.append(r['speedup'])
        
    X = np.array(X)
    y = np.array(y)
    
    # Train Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # Use cross_val_score to get robust R2
    scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    rf.fit(X, y)
    
    logger.info("="*60)
    logger.info("RES-284 RESULTS")
    logger.info("="*60)
    logger.info(f"Mean Speedup: {np.mean(y):.2f}x (std: {np.std(y):.2f})")
    logger.info(f"Model RÂ² (CV): {scores.mean():.4f} (std: {scores.std():.4f})")
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nFeature Importance:")
    for f in range(X.shape[1]):
        logger.info(f"  {feature_names[indices[f]]:20s}: {importances[indices[f]]:.4f}")
        
    # Save Results
    output = {
        'summary': {
            'n_cppns': config.n_cppns,
            'mean_speedup': float(np.mean(y)),
            'r2_cv': float(scores.mean()),
            'feature_importance': {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        },
        'raw_data': results
    }
    
    results_file = results_dir / 'res_284_results.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    run_experiment()