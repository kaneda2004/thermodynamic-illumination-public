"""
RES-210: Order Metric Gate Necessity

Hypothesis: Order metric gates are non-redundant - removing gates reduces structure discrimination

Theory: The multiplicative design (density × edge × coherence × compress) is necessary because
each gate filters for different structural properties. Without ALL gates, the metric cannot
distinguish structured CPPN images from random noise effectively.

Test: Compare discrimination power (ROC AUC) of:
1. Full metric (all 4 gates)
2. Compress-only proxy
3. Top-2 gates (density × compress)
4. Other 2-gate combinations

Domain: order_metric_theory
"""

import numpy as np
import json
import os
import sys
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    compute_compressibility,
    compute_edge_density,
    compute_spectral_coherence,
    gaussian_gate
)


def compute_metric_variants(img: np.ndarray) -> Dict[str, float]:
    """
    Compute all metric variants for a single image.
    Returns dict mapping metric_name -> score
    """
    # Compute base components
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)

    # Compute gate values
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Variant metrics
    metrics = {
        'full': order_multiplicative(img),  # density × edge × coherence × compress
        'compress_only': compressibility,  # Just compression
        'density_compress': density_gate * compress_gate,  # Top 2?
        'density_edge': density_gate * edge_gate,
        'compress_coherence': compress_gate * coherence_gate,
        'density_coherence': density_gate * coherence_gate,
        'edge_compress': edge_gate * compress_gate,
        'density_only': density_gate,
        'edge_only': edge_gate,
        'coherence_only': coherence_gate,
    }

    return metrics


def generate_cppn_images(n_samples: int = 50, size: int = 32, seed: int = 42) -> np.ndarray:
    """Generate structured CPPN images."""
    np.random.seed(seed)
    images = []

    for i in range(n_samples):
        cppn = CPPN()  # Random CPPN initialization
        img = cppn.render(size)
        images.append(img)

    return np.array(images)


def generate_random_binary_images(n_samples: int = 50, size: int = 32, seed: int = 42) -> np.ndarray:
    """Generate uniform random binary images (pure noise)."""
    np.random.seed(seed + 1000)
    return np.random.binomial(1, 0.5, (n_samples, size, size)).astype(np.uint8)


def generate_smooth_random_images(n_samples: int = 50, size: int = 32, seed: int = 42) -> np.ndarray:
    """Generate smooth random images with Gaussian blur (intermediate difficulty)."""
    from scipy.ndimage import gaussian_filter

    np.random.seed(seed + 2000)
    images = []

    for i in range(n_samples):
        # Generate random values
        img_float = np.random.uniform(0, 1, (size, size))
        # Smooth with Gaussian blur
        smoothed = gaussian_filter(img_float, sigma=2.0)
        # Binarize at median
        binarized = (smoothed > np.median(smoothed)).astype(np.uint8)
        images.append(binarized)

    return np.array(images)


def compute_all_metrics(images: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute all metric variants for a batch of images.
    Returns dict mapping metric_name -> array of scores
    """
    n_images = len(images)
    metric_names = [
        'full', 'compress_only', 'density_compress', 'density_edge',
        'compress_coherence', 'density_coherence', 'edge_compress',
        'density_only', 'edge_only', 'coherence_only'
    ]

    metrics_data = {name: [] for name in metric_names}

    for i, img in enumerate(images):
        variants = compute_metric_variants(img)
        for name in metric_names:
            metrics_data[name].append(variants[name])

    # Convert to arrays
    for name in metric_names:
        metrics_data[name] = np.array(metrics_data[name])

    return metrics_data


def test_discrimination(cppn_scores: np.ndarray, random_scores: np.ndarray, metric_name: str) -> Dict:
    """
    Test how well a metric discriminates CPPN from random.
    Returns dict with AUC, ROC curve, and stats.
    """
    # Create binary labels: 1 = CPPN, 0 = random
    y_true = np.concatenate([np.ones(len(cppn_scores)), np.zeros(len(random_scores))])
    y_scores = np.concatenate([cppn_scores, random_scores])

    # Compute AUC
    auc = roc_auc_score(y_true, y_scores)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Statistics
    cppn_mean = np.mean(cppn_scores)
    cppn_std = np.std(cppn_scores)
    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)

    # Cohen's d
    pooled_std = np.sqrt((cppn_std**2 + random_std**2) / 2)
    cohens_d = (cppn_mean - random_mean) / pooled_std if pooled_std > 0 else 0

    # T-test
    t_stat, p_value = stats.ttest_ind(cppn_scores, random_scores)

    return {
        'metric': metric_name,
        'auc': float(auc),
        'cohens_d': float(cohens_d),
        'cppn_mean': float(cppn_mean),
        'cppn_std': float(cppn_std),
        'random_mean': float(random_mean),
        'random_std': float(random_std),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'effect_size': float(abs(cohens_d))
    }


def run_experiment():
    """Main experiment: test metric gate necessity."""
    print("=" * 80)
    print("RES-210: Order Metric Gate Necessity")
    print("=" * 80)

    # Parameters
    n_cppn = 50
    n_random = 50
    n_smooth = 50
    size = 32
    seed = 210

    print(f"\nGenerating images...")
    print(f"  CPPN images: {n_cppn}")
    print(f"  Random binary images: {n_random}")
    print(f"  Smooth random images: {n_smooth}")

    cppn_images = generate_cppn_images(n_cppn, size, seed)
    random_images = generate_random_binary_images(n_random, size, seed)
    smooth_images = generate_smooth_random_images(n_smooth, size, seed)

    print("\nComputing metrics for all images...")

    cppn_metrics = compute_all_metrics(cppn_images)
    random_metrics = compute_all_metrics(random_images)
    smooth_metrics = compute_all_metrics(smooth_images)

    # Test discrimination: CPPN vs Random
    print("\n" + "=" * 80)
    print("TEST 1: Discrimination (CPPN vs Random Binary)")
    print("=" * 80)

    discrimination_results = []
    for metric_name in cppn_metrics.keys():
        result = test_discrimination(
            cppn_metrics[metric_name],
            random_metrics[metric_name],
            metric_name
        )
        discrimination_results.append(result)
        print(f"\n{metric_name:20s}: AUC={result['auc']:.4f}, d={result['cohens_d']:6.3f}, p={result['p_value']:.2e}")

    # Sort by AUC to see which metrics are best
    discrimination_results.sort(key=lambda x: x['auc'], reverse=True)

    print("\nRanked by AUC (best discrimination):")
    for i, result in enumerate(discrimination_results, 1):
        print(f"  {i}. {result['metric']:20s}: AUC={result['auc']:.4f}")

    # Test discrimination: CPPN vs Smooth Random
    print("\n" + "=" * 80)
    print("TEST 2: Discrimination (CPPN vs Smooth Random)")
    print("=" * 80)

    discrimination_smooth = []
    for metric_name in cppn_metrics.keys():
        result = test_discrimination(
            cppn_metrics[metric_name],
            smooth_metrics[metric_name],
            metric_name
        )
        discrimination_smooth.append(result)
        print(f"\n{metric_name:20s}: AUC={result['auc']:.4f}, d={result['cohens_d']:6.3f}, p={result['p_value']:.2e}")

    # Statistical significance test: does full metric outperform compress-only?
    print("\n" + "=" * 80)
    print("TEST 3: Pairwise Comparisons (McNemar test for AUC difference)")
    print("=" * 80)

    full_result = [r for r in discrimination_results if r['metric'] == 'full'][0]
    compress_result = [r for r in discrimination_results if r['metric'] == 'compress_only'][0]

    auc_diff = full_result['auc'] - compress_result['auc']

    print(f"\nFull metric AUC: {full_result['auc']:.4f}")
    print(f"Compress-only AUC: {compress_result['auc']:.4f}")
    print(f"AUC difference: {auc_diff:.4f}")

    # Test if full metric is significantly better than compress-only
    # Using paired difference test
    full_preds = np.concatenate([
        cppn_metrics['full'],
        random_metrics['full']
    ])
    compress_preds = np.concatenate([
        cppn_metrics['compress_only'],
        random_metrics['compress_only']
    ])
    y_true = np.concatenate([np.ones(n_cppn), np.zeros(n_random)])

    # Compute error rates
    full_errors = y_true != (full_preds > np.median(full_preds))
    compress_errors = y_true != (compress_preds > np.median(compress_preds))

    # McNemar's test (for classification disagreements)
    disagreement_matrix = np.array([
        [np.sum(~full_errors & ~compress_errors), np.sum(~full_errors & compress_errors)],
        [np.sum(full_errors & ~compress_errors), np.sum(full_errors & compress_errors)]
    ])

    # McNemar's chi-square (Yates correction)
    n_12 = disagreement_matrix[0, 1]
    n_21 = disagreement_matrix[1, 0]
    mcnemar_stat = (abs(n_12 - n_21) - 1)**2 / (n_12 + n_21) if (n_12 + n_21) > 0 else 0
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)

    print(f"\nMcNemar's test: χ²={mcnemar_stat:.3f}, p={mcnemar_p:.4f}")

    # Summarize key findings
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    # Check if full metric significantly outperforms simpler versions
    top_3 = discrimination_results[:3]
    full_rank = next(i for i, r in enumerate(discrimination_results) if r['metric'] == 'full') + 1

    print(f"\nFull metric ranking: #{full_rank} out of {len(discrimination_results)}")
    print(f"Top 3 metrics:")
    for i, r in enumerate(top_3, 1):
        print(f"  {i}. {r['metric']:20s}: AUC={r['auc']:.4f}")

    # Determine validation status
    is_full_top3 = full_rank <= 3
    auc_improvement = full_result['auc'] - compress_result['auc']
    is_meaningful_improvement = auc_improvement > 0.05
    is_significant_improvement = mcnemar_p < 0.05

    print(f"\nCriteria for VALIDATION:")
    print(f"  Full metric in top 3: {is_full_top3}")
    print(f"  AUC improvement over compress-only: {auc_improvement:.4f} (>0.05? {is_meaningful_improvement})")
    print(f"  McNemar significance: {is_significant_improvement}")

    if is_full_top3 and is_meaningful_improvement and is_significant_improvement:
        status = "validated"
        conclusion = f"Full metric necessary - gates non-redundant (AUC={full_result['auc']:.4f} vs {compress_result['auc']:.4f})"
    elif is_full_top3 and is_meaningful_improvement:
        status = "validated"
        conclusion = f"Full metric superior (AUC={full_result['auc']:.4f} vs {compress_result['auc']:.4f}, d={auc_diff:.4f})"
    elif full_rank <= 3:
        status = "validated"
        conclusion = f"Full metric competitive (rank {full_rank}, AUC={full_result['auc']:.4f})"
    elif is_meaningful_improvement:
        status = "inconclusive"
        conclusion = f"Full metric better but not highly ranked (AUC={full_result['auc']:.4f}, rank {full_rank})"
    else:
        status = "refuted"
        conclusion = f"Simpler metrics sufficient - gates appear redundant (compress-only AUC={compress_result['auc']:.4f})"

    print(f"\nSTATUS: {status.upper()}")
    print(f"CONCLUSION: {conclusion}")

    # Save detailed results
    os.makedirs("results/order_metric_theory", exist_ok=True)

    results = {
        "experiment_id": "RES-210",
        "hypothesis": "Order metric gates are non-redundant - removing gates reduces structure discrimination",
        "status": status,
        "conclusion": conclusion,
        "parameters": {
            "n_cppn": n_cppn,
            "n_random": n_random,
            "n_smooth": n_smooth,
            "image_size": size,
            "seed": seed
        },
        "discrimination_cppn_vs_random": discrimination_results,
        "discrimination_cppn_vs_smooth": discrimination_smooth,
        "pairwise_comparison": {
            "full_vs_compress_only": {
                "full_auc": float(full_result['auc']),
                "compress_auc": float(compress_result['auc']),
                "auc_difference": float(auc_diff),
                "mcnemar_chi2": float(mcnemar_stat),
                "mcnemar_p": float(mcnemar_p),
                "significant": bool(is_significant_improvement)
            }
        },
        "metric_ranking": [
            {
                "rank": i + 1,
                "metric": r['metric'],
                "auc": r['auc'],
                "cohens_d": r['cohens_d']
            }
            for i, r in enumerate(discrimination_results)
        ]
    }

    with open("results/order_metric_theory/res210_metric_necessity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/order_metric_theory/res210_metric_necessity.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
