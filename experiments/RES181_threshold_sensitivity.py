"""
RES-181: Order sensitivity to threshold perturbation peaks at 0.5 due to density gate design

Hypothesis: The order metric's sensitivity to small threshold changes (d(order)/d(threshold))
is maximal at threshold=0.5 because the density gate (Gaussian centered at 0.5) has its
steepest slope there.

Method:
1. Generate CPPN grayscale outputs (before thresholding)
2. For each grayscale image, compute order at thresholds [0.1, 0.2, ..., 0.9]
3. Compute local sensitivity = |order(t + delta) - order(t - delta)| / (2*delta)
4. Test if sensitivity at t=0.5 is significantly higher than at other thresholds

Success criteria: d>0.5, p<0.01
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats
import json

def compute_order_at_threshold(grayscale_img: np.ndarray, threshold: float) -> float:
    """Binarize grayscale image at threshold and compute order."""
    binary_img = (grayscale_img > threshold).astype(np.uint8)
    return order_multiplicative(binary_img)

def compute_sensitivity(grayscale_img: np.ndarray, threshold: float, delta: float = 0.05) -> float:
    """Compute order sensitivity at a given threshold using finite difference."""
    t_low = max(0.01, threshold - delta)
    t_high = min(0.99, threshold + delta)

    order_low = compute_order_at_threshold(grayscale_img, t_low)
    order_high = compute_order_at_threshold(grayscale_img, t_high)

    return abs(order_high - order_low) / (t_high - t_low)

def main():
    set_global_seed(42)

    # Generate CPPN grayscale outputs
    n_samples = 500
    size = 32

    print(f"Generating {n_samples} CPPN grayscale images...")
    grayscale_images = []

    for i in range(n_samples):
        cppn = CPPN()
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        grayscale = cppn.activate(x, y)
        grayscale_images.append(grayscale)

    # Test thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    delta = 0.03  # Small delta for sensitivity computation

    print("Computing sensitivity at each threshold...")
    sensitivity_by_threshold = {t: [] for t in thresholds}

    for img in grayscale_images:
        for t in thresholds:
            sens = compute_sensitivity(img, t, delta)
            sensitivity_by_threshold[t].append(sens)

    # Compute statistics
    mean_sensitivity = {t: np.mean(v) for t, v in sensitivity_by_threshold.items()}
    std_sensitivity = {t: np.std(v) for t, v in sensitivity_by_threshold.items()}

    print("\nSensitivity by threshold:")
    for t in thresholds:
        print(f"  t={t:.1f}: mean={mean_sensitivity[t]:.4f}, std={std_sensitivity[t]:.4f}")

    # Test hypothesis: sensitivity at 0.5 vs all other thresholds
    sens_05 = np.array(sensitivity_by_threshold[0.5])

    # Compare to average of non-0.5 thresholds
    non_05_thresholds = [t for t in thresholds if t != 0.5]
    sens_non05 = []
    for t in non_05_thresholds:
        sens_non05.extend(sensitivity_by_threshold[t])
    sens_non05 = np.array(sens_non05)

    # T-test
    t_stat, p_value = stats.ttest_ind(sens_05, sens_non05)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(sens_05)**2 + np.std(sens_non05)**2) / 2)
    effect_size = (np.mean(sens_05) - np.mean(sens_non05)) / pooled_std

    print(f"\nSensitivity at 0.5 vs other thresholds:")
    print(f"  Mean(0.5): {np.mean(sens_05):.4f}")
    print(f"  Mean(other): {np.mean(sens_non05):.4f}")
    print(f"  Cohen's d: {effect_size:.3f}")
    print(f"  p-value: {p_value:.2e}")

    # Also test vs specific thresholds
    print("\nPairwise comparisons (0.5 vs each):")
    pairwise_results = {}
    for t in non_05_thresholds:
        sens_t = np.array(sensitivity_by_threshold[t])
        t_stat_pair, p_val_pair = stats.ttest_ind(sens_05, sens_t)
        d_pair = (np.mean(sens_05) - np.mean(sens_t)) / np.sqrt((np.std(sens_05)**2 + np.std(sens_t)**2) / 2)
        pairwise_results[t] = {'d': d_pair, 'p': p_val_pair}
        print(f"  0.5 vs {t:.1f}: d={d_pair:.3f}, p={p_val_pair:.2e}")

    # Find which threshold has maximum sensitivity
    max_sens_threshold = max(mean_sensitivity.keys(), key=lambda t: mean_sensitivity[t])
    print(f"\nThreshold with max sensitivity: {max_sens_threshold}")

    # Determine outcome
    is_validated = (effect_size > 0.5 and p_value < 0.01 and max_sens_threshold == 0.5)
    is_refuted = (effect_size < 0.0) or (max_sens_threshold != 0.5 and abs(mean_sensitivity[max_sens_threshold] - mean_sensitivity[0.5]) > 0.01)

    if is_validated:
        status = "validated"
    elif is_refuted:
        status = "refuted"
    else:
        status = "inconclusive"

    print(f"\nStatus: {status}")

    # Save results
    results = {
        "experiment": "RES-181",
        "hypothesis": "Order sensitivity to threshold perturbation peaks at 0.5 due to density gate design",
        "domain": "threshold_behavior",
        "status": status,
        "n_samples": n_samples,
        "thresholds": thresholds,
        "mean_sensitivity": {str(k): v for k, v in mean_sensitivity.items()},
        "std_sensitivity": {str(k): v for k, v in std_sensitivity.items()},
        "max_sens_threshold": max_sens_threshold,
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "mean_at_05": float(np.mean(sens_05)),
        "mean_at_other": float(np.mean(sens_non05)),
        "pairwise_results": {str(k): {'d': float(v['d']), 'p': float(v['p'])} for k, v in pairwise_results.items()}
    }

    output_path = "results/threshold_sensitivity/results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results

if __name__ == "__main__":
    main()
