"""
RES-123: CPPN Expressivity Experiment

Hypothesis: CPPNs can only represent a sparse subset of binary images -
the distance from random binary images to nearest CPPN-representable images
is significantly larger than zero.

Method:
1. Generate random binary target images
2. Use gradient-free optimization to find CPPN that minimizes Hamming distance
3. Measure the minimum achievable distance
4. Compare to baseline (random CPPN distance to random image)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN

@dataclass
class ExprResult:
    target_distances: list  # Min distances achieved for each target
    baseline_distances: list  # Random CPPN distances
    n_targets: int
    n_optimization_samples: int

def hamming_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized Hamming distance between two binary images."""
    return np.mean(img1 != img2)

def optimize_cppn_to_target(target: np.ndarray, n_samples: int = 1000, size: int = 32) -> float:
    """
    Find CPPN that minimizes distance to target using random search.
    Returns minimum normalized Hamming distance found.
    """
    best_dist = 1.0

    for _ in range(n_samples):
        cppn = CPPN()  # New random CPPN
        rendered = cppn.render(size)
        dist = hamming_distance(rendered, target)
        if dist < best_dist:
            best_dist = dist

    return best_dist

def run_expressivity_experiment(
    n_targets: int = 50,
    n_optimization_samples: int = 2000,
    n_baseline_samples: int = 200,
    size: int = 32,
    seed: int = 42
) -> ExprResult:
    """
    Main experiment: measure how close CPPNs can get to random binary images.
    """
    np.random.seed(seed)

    target_distances = []
    baseline_distances = []

    print(f"Testing CPPN expressivity with {n_targets} random targets...")
    print(f"Optimization samples per target: {n_optimization_samples}")

    for i in range(n_targets):
        # Generate random binary target
        target = np.random.randint(0, 2, size=(size, size), dtype=np.uint8)

        # Find best CPPN match
        min_dist = optimize_cppn_to_target(target, n_optimization_samples, size)
        target_distances.append(min_dist)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_targets}: mean min distance so far = {np.mean(target_distances):.4f}")

    # Baseline: distance between random CPPNs and random images
    print(f"\nComputing baseline ({n_baseline_samples} random pairs)...")
    for _ in range(n_baseline_samples):
        target = np.random.randint(0, 2, size=(size, size), dtype=np.uint8)
        cppn = CPPN()
        rendered = cppn.render(size)
        baseline_distances.append(hamming_distance(rendered, target))

    return ExprResult(
        target_distances=target_distances,
        baseline_distances=baseline_distances,
        n_targets=n_targets,
        n_optimization_samples=n_optimization_samples
    )

def analyze_results(result: ExprResult) -> dict:
    """Statistical analysis of expressivity results."""

    target_dists = np.array(result.target_distances)
    baseline_dists = np.array(result.baseline_distances)

    # Key question: Is optimized distance significantly > 0?
    # This would indicate CPPNs can't perfectly represent arbitrary images

    # One-sample t-test: is mean optimized distance > 0?
    t_stat, p_one = stats.ttest_1samp(target_dists, 0)
    p_greater_than_zero = p_one / 2 if t_stat > 0 else 1 - p_one / 2

    # Effect size: how much worse than perfect (0) is the optimization?
    # Cohen's d relative to zero
    effect_size_vs_zero = np.mean(target_dists) / np.std(target_dists)

    # Comparison: optimized vs baseline (random CPPN)
    t_vs_baseline, p_vs_baseline = stats.ttest_ind(target_dists, baseline_dists)
    improvement = (np.mean(baseline_dists) - np.mean(target_dists)) / np.std(baseline_dists)

    # Expected random distance should be ~0.5 for random binary images
    expected_random = 0.5

    return {
        'mean_optimized_dist': np.mean(target_dists),
        'std_optimized_dist': np.std(target_dists),
        'min_optimized_dist': np.min(target_dists),
        'max_optimized_dist': np.max(target_dists),
        'mean_baseline_dist': np.mean(baseline_dists),
        'std_baseline_dist': np.std(baseline_dists),
        'effect_size_vs_zero': effect_size_vs_zero,
        'p_greater_than_zero': p_greater_than_zero,
        'improvement_over_baseline': improvement,
        'p_vs_baseline': p_vs_baseline,
        't_vs_baseline': t_vs_baseline,
        'gap_from_perfect': np.mean(target_dists),  # How far from 0
        'expressivity_coverage': 1 - np.mean(target_dists)  # Fraction of pixels matchable
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RES-123: CPPN Expressivity Experiment")
    print("=" * 60)

    # Run main experiment
    result = run_expressivity_experiment(
        n_targets=50,
        n_optimization_samples=2000,
        n_baseline_samples=200,
        size=32,
        seed=42
    )

    # Analyze
    analysis = analyze_results(result)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOptimized CPPN → Random Target Distance:")
    print(f"  Mean: {analysis['mean_optimized_dist']:.4f} ± {analysis['std_optimized_dist']:.4f}")
    print(f"  Range: [{analysis['min_optimized_dist']:.4f}, {analysis['max_optimized_dist']:.4f}]")

    print(f"\nBaseline (Random CPPN) Distance:")
    print(f"  Mean: {analysis['mean_baseline_dist']:.4f} ± {analysis['std_baseline_dist']:.4f}")

    print(f"\nKey Metrics:")
    print(f"  Effect size (vs zero): {analysis['effect_size_vs_zero']:.2f}")
    print(f"  p-value (dist > 0): {analysis['p_greater_than_zero']:.2e}")
    print(f"  Improvement over baseline: {analysis['improvement_over_baseline']:.2f} Cohen's d")
    print(f"  p-value (vs baseline): {analysis['p_vs_baseline']:.2e}")

    print(f"\nInterpretation:")
    print(f"  Expressivity coverage: {analysis['expressivity_coverage']*100:.1f}% of pixels matchable")
    print(f"  Gap from perfect representation: {analysis['gap_from_perfect']*100:.1f}%")

    # Validation check
    validated = (analysis['p_greater_than_zero'] < 0.01 and
                 analysis['effect_size_vs_zero'] > 0.5 and
                 analysis['mean_optimized_dist'] > 0.3)  # Significant gap from perfect

    print(f"\n" + "=" * 60)
    print(f"VALIDATION: {'VALIDATED' if validated else 'REFUTED'}")
    print("=" * 60)

    if validated:
        print("CPPNs CAN'T perfectly represent arbitrary binary images.")
        print(f"Minimum achievable distance: {analysis['mean_optimized_dist']:.3f} (significant gap from 0)")
    else:
        print("Result inconclusive or CPPNs have high expressivity.")
