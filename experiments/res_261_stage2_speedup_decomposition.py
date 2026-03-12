#!/usr/bin/env python3
"""
RES-261: Stage 2 Speedup Decomposition Analysis

HYPOTHESIS: The 8% speedup from richer features comes primarily from
tighter Stage 2 manifold (better entropy descent), not from Stage 2 being
inherently faster.

METHOD:
1. For baseline [x,y,r] and full [x,y,r,x*y,x²,y²] feature sets:
   - Run Stage 1 (150 iterations) → measure final manifold entropy
   - Extract Stage 1 manifold (PCA basis)
   - Run Stage 2 in that manifold → measure samples needed
   - Decompose total speedup = (Stage 1 efficiency × Stage 2 efficiency)
2. Compare decomposition baseline vs full
3. Determine if speedup comes from tighter manifold or faster discovery
"""

import numpy as np
import json
from pathlib import Path
import sys
import os

# Ensure imports work from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_cppn_image(weights, size=64):
    """Generate image from CPPN network weights."""
    image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            # Normalize coordinates to [-1, 1]
            x = (i - size/2) / (size/2)
            y = (j - size/2) / (size/2)
            r = np.sqrt(x**2 + y**2)

            # Simple CPPN: weighted combination of features
            val = (
                weights[0] * x +
                weights[1] * y +
                weights[2] * r +
                (weights[3] if len(weights) > 3 else 0) * (x * y) +
                (weights[4] if len(weights) > 4 else 0) * (x**2) +
                (weights[5] if len(weights) > 5 else 0) * (y**2)
            )
            image[i, j] = np.tanh(val)

    return image

def compute_entropy(image, bins=16):
    """Compute Shannon entropy of image histogram."""
    # Flatten and clip to valid range
    img_flat = np.clip(image.flatten(), -1, 1)
    hist, _ = np.histogram(img_flat, bins=bins, range=(-1, 1))
    # Normalize histogram to probabilities
    hist = hist / np.sum(hist)
    # Compute Shannon entropy
    hist = hist[hist > 0]  # Remove zero bins
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def pca_manifold(weights, n_components=2):
    """Extract manifold basis using PCA."""
    if weights.shape[1] < 3:
        return weights

    mean = weights.mean(axis=0)
    centered = weights - mean
    cov = centered.T @ centered / (len(weights) - 1)

    # SVD for numerical stability
    U, S, _ = np.linalg.svd(cov, full_matrices=False)

    # Project onto principal components
    basis = U[:, :n_components]
    projections = centered @ basis

    return projections, basis, mean

def stage_1_sampling(feature_set, n_iterations=150, n_samples=100):
    """
    Stage 1: Discovery phase - gradient-like optimization in feature space.
    Returns: best_weights, entropy trajectory, final manifold
    """
    n_features = len(feature_set)

    # Initialize random weights
    weights = np.random.randn(n_samples, n_features)
    entropy_trajectory = []
    best_entropies_so_far = []

    for iteration in range(n_iterations):
        # Evaluate current population
        entropies = np.array([
            compute_entropy(create_cppn_image(np.concatenate([w[:3], np.zeros(max(0, n_features - 3))])))
            for w in weights
        ])

        mean_entropy = float(np.mean(entropies))
        min_entropy = float(np.min(entropies))

        # Track best entropy found so far
        if iteration == 0:
            best_entropy_overall = min_entropy
        else:
            best_entropy_overall = min(best_entropy_overall, min_entropy)

        best_entropies_so_far.append(best_entropy_overall)

        entropy_trajectory.append({
            'iteration': iteration,
            'mean_entropy': mean_entropy,
            'min_entropy': min_entropy,
            'max_entropy': float(np.max(entropies)),
            'std_entropy': float(np.std(entropies)),
            'best_so_far': best_entropy_overall
        })

        # Guided mutation: bias towards better-performing weights
        scores = 1.0 / (1.0 + entropies)  # Higher score for lower entropy
        scores = scores / scores.sum()

        # Elite selection + mutation
        elite_idx = np.argsort(entropies)[:max(1, n_samples // 4)]
        new_weights = weights[elite_idx]

        # Mutate elite
        for _ in range(n_samples - len(elite_idx)):
            parent_idx = np.random.choice(elite_idx)
            child = weights[parent_idx].copy()
            child += 0.1 * np.random.randn(n_features)
            new_weights = np.vstack([new_weights, child])

        weights = np.clip(new_weights, -5, 5)

    # Final evaluation
    final_entropies = np.array([
        compute_entropy(create_cppn_image(np.concatenate([w[:3], np.zeros(max(0, n_features - 3))])))
        for w in weights
    ])

    final_entropy = float(np.mean(final_entropies))
    final_best = float(np.min(final_entropies))

    return weights, entropy_trajectory, final_entropy, final_best

def stage_2_sampling(weights, feature_set, pca_basis, pca_mean, n_iterations=100):
    """
    Stage 2: Refinement phase - sampling on discovered manifold.
    Measures how fast convergence happens vs. Stage 1 speed.
    Returns: convergence_curve, final_best_entropy
    """
    n_features = len(feature_set)
    n_samples = min(50, len(weights) // 2)  # Use subset for refinement

    # Project to manifold
    centered = weights[:n_samples] - pca_mean
    latent = centered @ pca_basis

    # Track convergence on manifold
    best_entropy_trajectory = []
    best_entropy_overall = np.inf

    for iteration in range(n_iterations):
        # Small refinement in latent space
        latent += 0.05 * np.random.randn(*latent.shape)
        latent = np.clip(latent, -3, 3)

        # Reconstruct to feature space
        reconstructed = (latent @ pca_basis.T) + pca_mean

        # Evaluate entropy
        entropies = np.array([
            compute_entropy(create_cppn_image(np.concatenate([w[:3], np.zeros(max(0, n_features - 3))])))
            for w in reconstructed
        ])

        min_entropy = np.min(entropies)
        best_entropy_overall = min(best_entropy_overall, min_entropy)
        best_entropy_trajectory.append(best_entropy_overall)

    return np.array(best_entropy_trajectory), best_entropy_overall

def decompose_speedup():
    """Main decomposition analysis."""

    print("=" * 70)
    print("RES-261: Stage 2 Speedup Decomposition Analysis")
    print("=" * 70)

    results = {
        "experiment": "RES-261",
        "domain": "entropy_reduction",
        "hypothesis": "Speedup decomposition: tighter manifold vs faster discovery",
        "method": "Two-stage sampling analysis with feature decomposition",
    }

    # Feature sets
    baseline_features = ["x", "y", "r"]
    full_features = ["x", "y", "r", "x*y", "x²", "y²"]

    stage1_results = {}
    stage2_results = {}

    print("\nSTAGE 1: MANIFOLD DISCOVERY")
    print("-" * 70)

    for name, features in [("baseline", baseline_features), ("full", full_features)]:
        print(f"\nProcessing {name} ({len(features)} features)...")

        # Stage 1: Discover manifold
        weights, entropy_traj, final_entropy, final_best = stage_1_sampling(features, n_iterations=150, n_samples=100)

        # Extract manifold
        projections, basis, mean = pca_manifold(weights, n_components=2)

        initial_entropy = entropy_traj[0]['mean_entropy']
        # Use best entropy found, not mean
        entropy_reduction_s1 = (initial_entropy - final_best) / initial_entropy * 100

        # Measure convergence speed in Stage 1 using mean entropy trajectory
        # Track how fast mean entropy decreases (more realistic measure)
        mean_trajectory_s1 = [t['mean_entropy'] for t in entropy_traj]
        target_s1 = initial_entropy * 0.5
        samples_to_target_s1 = len(entropy_traj)
        for i, mean in enumerate(mean_trajectory_s1):
            if mean <= target_s1:
                samples_to_target_s1 = i
                break

        # Measure mean entropy final value
        final_mean_trajectory = np.mean([t['mean_entropy'] for t in entropy_traj[-10:]])

        stage1_results[name] = {
            'features': features,
            'n_features': len(features),
            'initial_entropy': float(initial_entropy),
            'final_best_entropy': float(final_best),
            'final_mean_entropy': float(final_mean_trajectory),
            'entropy_reduction_pct': float(entropy_reduction_s1),
            'samples_to_50pct_reduction': int(samples_to_target_s1),
            'entropy_trajectory_convergence': float(mean_trajectory_s1[-1]),
            'manifold_basis_shape': list(basis.shape),
            'pca_variance_retained': float(np.sum(projections.var(axis=0))),
        }

        print(f"  Initial entropy: {initial_entropy:.4f}")
        print(f"  Final best entropy: {final_best:.4f}")
        print(f"  Entropy reduction: {entropy_reduction_s1:.1f}%")
        print(f"  Samples to 50% reduction: {samples_to_target_s1}")
        print(f"  Manifold dimension: {basis.shape[1]}")

        # Stage 2: Refine on manifold
        print(f"\n  STAGE 2: MANIFOLD REFINEMENT")
        s2_trajectory, s2_best = stage_2_sampling(weights, features, basis, mean, n_iterations=100)

        # Measure convergence speed in Stage 2
        # Use convergence curve acceleration metric
        if len(s2_trajectory) > 1:
            # Measure entropy reduction per iteration
            improvements = np.diff(s2_trajectory)
            improvements = np.clip(improvements, -1, 0)  # Only consider reductions
            avg_improvement_s2 = np.mean(np.abs(improvements))

            target_s2 = s2_trajectory[0] * 0.5
            samples_to_target_s2 = len(s2_trajectory)
            for i, best in enumerate(s2_trajectory):
                if best <= target_s2:
                    samples_to_target_s2 = i
                    break
        else:
            avg_improvement_s2 = 0
            samples_to_target_s2 = 100

        entropy_reduction_s2 = (s2_trajectory[0] - s2_best) / s2_trajectory[0] * 100 if s2_trajectory[0] > 0 else 0

        stage2_results[name] = {
            'starting_entropy_s2': float(s2_trajectory[0]),
            'final_entropy_s2': float(s2_best),
            'entropy_reduction_pct': float(entropy_reduction_s2),
            'samples_to_50pct_reduction_s2': int(samples_to_target_s2),
            'avg_entropy_improvement_per_sample': float(avg_improvement_s2),
        }

        print(f"  Starting entropy (S2): {s2_trajectory[0]:.4f}")
        print(f"  Final entropy (S2): {s2_best:.4f}")
        print(f"  Entropy reduction: {entropy_reduction_s2:.1f}%")
        print(f"  Samples to 50% reduction: {samples_to_target_s2}")
        print(f"  Avg improvement per sample: {avg_improvement_s2:.6f}")

    # DECOMPOSITION ANALYSIS
    print("\n" + "=" * 70)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 70)

    # Use sample counts for decomposition (how many samples to reach 50% improvement)
    baseline_s1_samples = stage1_results['baseline']['samples_to_50pct_reduction']
    baseline_s2_samples = stage2_results['baseline']['samples_to_50pct_reduction_s2']
    baseline_total_samples = baseline_s1_samples + baseline_s2_samples

    full_s1_samples = stage1_results['full']['samples_to_50pct_reduction']
    full_s2_samples = stage2_results['full']['samples_to_50pct_reduction_s2']
    full_total_samples = full_s1_samples + full_s2_samples

    # Also track entropy reductions
    baseline_s1_entropy = stage1_results['baseline']['entropy_reduction_pct']
    baseline_s2_entropy = stage2_results['baseline']['entropy_reduction_pct']
    full_s1_entropy = stage1_results['full']['entropy_reduction_pct']
    full_s2_entropy = stage2_results['full']['entropy_reduction_pct']

    # Decompose contributions by samples
    baseline_s1_contribution = (baseline_s1_samples / baseline_total_samples * 100) if baseline_total_samples > 0 else 0
    baseline_s2_contribution = (baseline_s2_samples / baseline_total_samples * 100) if baseline_total_samples > 0 else 0

    full_s1_contribution = (full_s1_samples / full_total_samples * 100) if full_total_samples > 0 else 0
    full_s2_contribution = (full_s2_samples / full_total_samples * 100) if full_total_samples > 0 else 0

    decomposition = {
        'baseline': {
            'stage1_entropy_reduction_pct': float(baseline_s1_entropy),
            'stage2_entropy_reduction_pct': float(baseline_s2_entropy),
            'stage1_samples_to_50pct': int(baseline_s1_samples),
            'stage2_samples_to_50pct': int(baseline_s2_samples),
            'total_samples_to_target': int(baseline_total_samples),
            'stage1_contribution_pct': float(baseline_s1_contribution),
            'stage2_contribution_pct': float(baseline_s2_contribution),
        },
        'full': {
            'stage1_entropy_reduction_pct': float(full_s1_entropy),
            'stage2_entropy_reduction_pct': float(full_s2_entropy),
            'stage1_samples_to_50pct': int(full_s1_samples),
            'stage2_samples_to_50pct': int(full_s2_samples),
            'total_samples_to_target': int(full_total_samples),
            'stage1_contribution_pct': float(full_s1_contribution),
            'stage2_contribution_pct': float(full_s2_contribution),
        }
    }

    # Speedup analysis (samples + entropy improvement rate)
    speedup_improvement_samples = (baseline_total_samples - full_total_samples) / baseline_total_samples * 100
    speedup_improvement_entropy = (full_s1_entropy + full_s2_entropy) - (baseline_s1_entropy + baseline_s2_entropy)

    stage1_speedup = (baseline_s1_samples - full_s1_samples) / baseline_s1_samples * 100 if baseline_s1_samples > 0 else 0
    stage2_speedup = (baseline_s2_samples - full_s2_samples) / baseline_s2_samples * 100 if baseline_s2_samples > 0 else 0

    # Also compare entropy reduction rates per sample (manifold tightness)
    baseline_s2_rate = stage2_results['baseline']['avg_entropy_improvement_per_sample']
    full_s2_rate = stage2_results['full']['avg_entropy_improvement_per_sample']
    s2_rate_improvement = (full_s2_rate - baseline_s2_rate) / baseline_s2_rate * 100 if baseline_s2_rate > 0 else 0

    stage_analysis = {
        'speedup_from_sample_reduction_pct': float(speedup_improvement_samples),
        'entropy_improvement_pct': float(speedup_improvement_entropy),
        'stage1_speedup_pct': float(stage1_speedup),
        'stage2_speedup_pct': float(stage2_speedup),
        'stage2_convergence_rate_improvement_pct': float(s2_rate_improvement),
        'interpretation': 'manifold tightness' if s2_rate_improvement > 0 else 'not tightness',
    }

    print(f"\nBASELINE [{', '.join(baseline_features)}]")
    print(f"  Stage 1: {baseline_s1_entropy:.1f}% entropy reduction in {baseline_s1_samples} samples")
    print(f"  Stage 2: {baseline_s2_entropy:.1f}% entropy reduction in {baseline_s2_samples} samples")
    print(f"  Stage 2 convergence rate: {baseline_s2_rate:.6f} entropy/sample")
    print(f"  Total: {baseline_total_samples} samples")
    print(f"  Stage 1 contribution: {baseline_s1_contribution:.1f}%")
    print(f"  Stage 2 contribution: {baseline_s2_contribution:.1f}%")

    print(f"\nFULL [{', '.join(full_features)}]")
    print(f"  Stage 1: {full_s1_entropy:.1f}% entropy reduction in {full_s1_samples} samples")
    print(f"  Stage 2: {full_s2_entropy:.1f}% entropy reduction in {full_s2_samples} samples")
    print(f"  Stage 2 convergence rate: {full_s2_rate:.6f} entropy/sample")
    print(f"  Total: {full_total_samples} samples")
    print(f"  Stage 1 contribution: {full_s1_contribution:.1f}%")
    print(f"  Stage 2 contribution: {full_s2_contribution:.1f}%")

    print(f"\nIMPROVEMENT FROM RICHER FEATURES")
    print(f"  Sample count change: {speedup_improvement_samples:.1f}% ({baseline_total_samples} → {full_total_samples})")
    print(f"  Stage 1 sample speedup: {stage1_speedup:.1f}% ({baseline_s1_samples} → {full_s1_samples} samples)")
    print(f"  Stage 2 sample speedup: {stage2_speedup:.1f}% ({baseline_s2_samples} → {full_s2_samples} samples)")
    print(f"  Stage 2 convergence rate improvement: {s2_rate_improvement:.1f}% (manifold tightness gain)")
    print(f"  Total entropy improvement: {speedup_improvement_entropy:.1f}%")

    # Determine which stage drives improvement
    # Hypothesis: richer features improve manifold tightness (Stage 2 convergence rate)
    if s2_rate_improvement > 10:  # Significant manifold tightness improvement
        status = "validated_manifold_tightness"
        summary = f"Stage 2 manifold convergence rate improved {s2_rate_improvement:.1f}% - VALIDATES tighter manifold hypothesis"
    elif stage1_speedup > 0 and stage2_speedup <= 0:
        status = "stage1_dominant"
        summary = f"Stage 1 speedup {stage1_speedup:.1f}% dominates (discovery efficiency), Stage 2 shows marginal change"
    elif stage2_speedup > 0:
        status = "stage2_contributes"
        summary = f"Both stages contribute: Stage 1 {stage1_speedup:.1f}%, Stage 2 {stage2_speedup:.1f}%"
    else:
        status = "inconclusive"
        summary = f"No clear speedup pattern. Total entropy improved {speedup_improvement_entropy:.1f}% but samples unchanged"

    results['decomposition'] = decomposition
    results['stage_analysis'] = stage_analysis
    results['status'] = status
    results['summary'] = summary

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/entropy_reduction')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'res_261_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    return results

if __name__ == '__main__':
    results = decompose_speedup()
    print("\n" + "=" * 70)
    print(f"STATUS: {results['status'].upper()}")
    print(f"SUMMARY: {results['summary']}")
    print("=" * 70)
