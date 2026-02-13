#!/usr/bin/env python3
"""
EXPERIMENT: Prior Feature Divergence

Hypothesis: Different structured priors (CPPN, sinusoidal sums, Perlin noise)
achieve similar order values through fundamentally different structural mechanisms,
as measured by feature vector divergence.

Null Hypothesis: Feature distributions are identical across priors when matched
on order threshold.

This explores the generative-discriminative tradeoff: if different priors reach
the same order through different mechanisms, then "order" is a lossy discriminative
measure that compresses away generative diversity.

Builds on: RES-007 (feature correlations with order)
Domain: generative_discriminative_tradeoff, prior_comparison
"""

import sys
import os
import numpy as np
from scipy.stats import f as f_dist
from scipy.spatial.distance import pdist
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    order_multiplicative,
    compute_symmetry,
    compute_edge_density,
    compute_spectral_coherence,
    compute_compressibility,
    CPPN,
)


# ============================================================================
# ALTERNATIVE PRIORS
# ============================================================================

class SinusoidalPrior:
    """
    Sum of sinusoidal waves with random frequencies and phases.
    Creates periodic, wave-like patterns.
    """
    def __init__(self, n_waves: int = 5):
        self.n_waves = n_waves
        self.frequencies_x = np.random.uniform(0.5, 4.0, n_waves)
        self.frequencies_y = np.random.uniform(0.5, 4.0, n_waves)
        self.phases_x = np.random.uniform(0, 2*np.pi, n_waves)
        self.phases_y = np.random.uniform(0, 2*np.pi, n_waves)
        self.weights = np.random.randn(n_waves)
        self.threshold = np.random.uniform(-0.5, 0.5)

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-np.pi, np.pi, size)
        x, y = np.meshgrid(coords, coords)

        result = np.zeros_like(x)
        for i in range(self.n_waves):
            result += self.weights[i] * (
                np.sin(self.frequencies_x[i] * x + self.phases_x[i]) +
                np.sin(self.frequencies_y[i] * y + self.phases_y[i])
            )

        return (result > self.threshold).astype(np.uint8)


class PerlinNoisePrior:
    """
    Simplified Perlin-like noise using interpolated random grid.
    Creates smooth, cloud-like patterns.
    """
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        # Random gradients at grid points
        self.grid = np.random.randn(grid_size + 1, grid_size + 1, 2)
        self.grid = self.grid / np.linalg.norm(self.grid, axis=2, keepdims=True)
        self.threshold = np.random.uniform(-0.3, 0.3)

    def _smoothstep(self, t):
        """Smooth interpolation function."""
        return t * t * (3 - 2 * t)

    def render(self, size: int = 32) -> np.ndarray:
        result = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                # Normalize to grid coordinates
                x = j / size * self.grid_size
                y = i / size * self.grid_size

                # Grid cell coordinates
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, self.grid_size), min(y0 + 1, self.grid_size)

                # Local coordinates within cell
                sx = x - x0
                sy = y - y0

                # Dot products with gradients
                n00 = np.dot(self.grid[y0, x0], [sx, sy])
                n10 = np.dot(self.grid[y0, x1], [sx - 1, sy])
                n01 = np.dot(self.grid[y1, x0], [sx, sy - 1])
                n11 = np.dot(self.grid[y1, x1], [sx - 1, sy - 1])

                # Interpolate
                u = self._smoothstep(sx)
                v = self._smoothstep(sy)

                nx0 = n00 * (1 - u) + n10 * u
                nx1 = n01 * (1 - u) + n11 * u
                result[i, j] = nx0 * (1 - v) + nx1 * v

        return (result > self.threshold).astype(np.uint8)


class GaborPrior:
    """
    Sum of Gabor filters with random orientations and scales.
    Creates texture-like patterns.
    """
    def __init__(self, n_gabors: int = 4):
        self.n_gabors = n_gabors
        self.orientations = np.random.uniform(0, np.pi, n_gabors)
        self.frequencies = np.random.uniform(0.1, 0.4, n_gabors)
        self.scales = np.random.uniform(0.1, 0.3, n_gabors)
        self.centers_x = np.random.uniform(-0.5, 0.5, n_gabors)
        self.centers_y = np.random.uniform(-0.5, 0.5, n_gabors)
        self.weights = np.random.randn(n_gabors)
        self.threshold = np.random.uniform(-0.3, 0.3)

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)

        result = np.zeros_like(x)
        for i in range(self.n_gabors):
            # Rotate coordinates
            theta = self.orientations[i]
            x_rot = (x - self.centers_x[i]) * np.cos(theta) + (y - self.centers_y[i]) * np.sin(theta)
            y_rot = -(x - self.centers_x[i]) * np.sin(theta) + (y - self.centers_y[i]) * np.cos(theta)

            # Gaussian envelope
            envelope = np.exp(-(x_rot**2 + y_rot**2) / (2 * self.scales[i]**2))

            # Sinusoidal carrier
            carrier = np.cos(2 * np.pi * self.frequencies[i] * x_rot * size)

            result += self.weights[i] * envelope * carrier

        return (result > self.threshold).astype(np.uint8)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(img: np.ndarray) -> dict:
    """Extract feature vector from image."""
    return {
        'edge_density': compute_edge_density(img),
        'symmetry': compute_symmetry(img),
        'spectral_coherence': compute_spectral_coherence(img),
        'compressibility': compute_compressibility(img),
        'density': np.mean(img),
    }


def feature_vector(img: np.ndarray) -> np.ndarray:
    """Extract feature vector as numpy array."""
    features = extract_features(img)
    return np.array([
        features['edge_density'],
        features['symmetry'],
        features['spectral_coherence'],
        features['compressibility'],
    ])


# ============================================================================
# SAMPLE GENERATION AT ORDER THRESHOLD
# ============================================================================

def sample_from_prior(prior_class, n_samples: int, order_min: float, order_max: float,
                     image_size: int = 32, max_attempts: int = 10000, **kwargs) -> list:
    """
    Generate samples from a prior within an order band.

    Returns list of (image, order, features) tuples.
    """
    samples = []
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        prior = prior_class(**kwargs)
        img = prior.render(image_size)
        order = order_multiplicative(img)

        if order_min <= order <= order_max:
            features = feature_vector(img)
            samples.append((img, order, features))

    return samples


def sample_cppn(n_samples: int, order_min: float, order_max: float,
               image_size: int = 32, max_attempts: int = 10000) -> list:
    """Generate CPPN samples within order band."""
    samples = []
    attempts = 0

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        if order_min <= order <= order_max:
            features = feature_vector(img)
            samples.append((img, order, features))

    return samples


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def manova_two_groups(X1: np.ndarray, X2: np.ndarray) -> dict:
    """
    Two-sample MANOVA using Pillai's trace.

    Returns test statistic, F-approximation, and p-value.
    """
    n1, p = X1.shape
    n2 = X2.shape[0]
    N = n1 + n2

    # Group means
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    grand_mean = (n1 * mean1 + n2 * mean2) / N

    # Between-group scatter matrix
    B = (n1 * np.outer(mean1 - grand_mean, mean1 - grand_mean) +
         n2 * np.outer(mean2 - grand_mean, mean2 - grand_mean))

    # Within-group scatter matrix
    W1 = (X1 - mean1).T @ (X1 - mean1)
    W2 = (X2 - mean2).T @ (X2 - mean2)
    W = W1 + W2

    # Total scatter
    T = B + W

    # Pillai's trace: tr(B * inv(T))
    try:
        # Add regularization for numerical stability
        T_reg = T + 1e-8 * np.eye(p)
        T_inv = np.linalg.inv(T_reg)
        pillai = np.trace(B @ T_inv)
    except np.linalg.LinAlgError:
        return {'error': 'singular matrix'}

    # F-approximation for two groups
    s = 1  # min(p, g-1) where g=2
    m = (abs(p - s) - 1) / 2
    n = (N - p - 2) / 2

    df1 = p
    df2 = N - p - 1

    if df2 <= 0:
        return {'error': 'insufficient df'}

    # F statistic
    if pillai < 1:
        F = (pillai / (1 - pillai)) * (df2 / df1)
    else:
        F = np.inf

    # P-value
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        'pillai_trace': float(pillai),
        'F_statistic': float(F),
        'df1': int(df1),
        'df2': int(df2),
        'p_value': float(p_value),
    }


def manova_multi_groups(groups: list[np.ndarray]) -> dict:
    """
    Multi-group MANOVA using Pillai's trace.

    groups: list of (n_i, p) arrays
    """
    g = len(groups)
    ns = [X.shape[0] for X in groups]
    N = sum(ns)
    p = groups[0].shape[1]

    # Grand mean
    grand_mean = sum(n * np.mean(X, axis=0) for n, X in zip(ns, groups)) / N

    # Between-group scatter
    B = np.zeros((p, p))
    for n, X in zip(ns, groups):
        diff = np.mean(X, axis=0) - grand_mean
        B += n * np.outer(diff, diff)

    # Within-group scatter
    W = np.zeros((p, p))
    for X in groups:
        X_centered = X - np.mean(X, axis=0)
        W += X_centered.T @ X_centered

    # Total scatter
    T = B + W

    # Pillai's trace
    try:
        T_reg = T + 1e-8 * np.eye(p)
        T_inv = np.linalg.inv(T_reg)
        pillai = np.trace(B @ T_inv)
    except np.linalg.LinAlgError:
        return {'error': 'singular matrix'}

    # F-approximation (Pillai-Bartlett)
    s = min(p, g - 1)
    m = (abs(p - g + 1) - 1) / 2
    n_stat = (N - g - p) / 2

    # Approximation for Pillai
    df1 = s * p
    df2 = s * (N - g - p + s)

    if df2 <= 0 or s == 0:
        return {'error': 'insufficient df'}

    # F from Pillai
    V = pillai  # Sum over eigenvalues
    F = (V / s) / ((1 - V / s) * (df2 / (s * (N - g - p + s))))

    # Simpler approximation
    F = (pillai / (s * (1 - pillai/s))) * (df2 / df1) if pillai < s else np.inf

    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        'pillai_trace': float(pillai),
        'F_statistic': float(F),
        'df1': int(df1),
        'df2': int(df2),
        'p_value': float(p_value),
        'n_groups': g,
    }


def cohens_d_multivariate(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Multivariate Cohen's d using pooled covariance.
    Also known as Mahalanobis distance.
    """
    n1, p = X1.shape
    n2 = X2.shape[0]

    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)

    # Pooled covariance
    S1 = np.cov(X1.T)
    S2 = np.cov(X2.T)
    S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # Mahalanobis distance
    try:
        S_inv = np.linalg.inv(S_pooled + 1e-8 * np.eye(p))
        diff = mean1 - mean2
        d = np.sqrt(diff @ S_inv @ diff)
    except np.linalg.LinAlgError:
        d = np.nan

    return float(d)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """
    Main experiment: Compare feature distributions across priors at matched order.
    """
    print("=" * 70)
    print("EXPERIMENT: PRIOR FEATURE DIVERGENCE")
    print("=" * 70)
    print()
    print("Hypothesis: Different structured priors achieve similar order values")
    print("through fundamentally different structural mechanisms.")
    print()
    print("H0: Feature distributions identical across priors at matched order")
    print("H1: Feature distributions differ significantly")
    print()

    # Parameters
    n_samples = 100  # Per prior
    image_size = 32
    order_min = 0.05
    order_max = 0.20  # Order band to match on

    np.random.seed(42)

    print(f"Parameters:")
    print(f"  Samples per prior: {n_samples}")
    print(f"  Order band: [{order_min}, {order_max}]")
    print(f"  Image size: {image_size}x{image_size}")
    print()

    # Generate samples from each prior
    priors = {
        'CPPN': lambda: sample_cppn(n_samples, order_min, order_max, image_size),
        'Sinusoidal': lambda: sample_from_prior(SinusoidalPrior, n_samples, order_min, order_max, image_size),
        'Perlin': lambda: sample_from_prior(PerlinNoisePrior, n_samples, order_min, order_max, image_size),
        'Gabor': lambda: sample_from_prior(GaborPrior, n_samples, order_min, order_max, image_size),
    }

    samples = {}
    for name, sampler in priors.items():
        print(f"Sampling from {name} prior...")
        samples[name] = sampler()
        print(f"  Got {len(samples[name])} samples")

    # Check if we have enough samples
    min_samples = min(len(s) for s in samples.values())
    if min_samples < 20:
        print(f"\nWARNING: Only {min_samples} matched samples - may lack statistical power")

    print()

    # Extract feature matrices
    feature_matrices = {}
    order_values = {}
    for name, s in samples.items():
        if len(s) > 0:
            feature_matrices[name] = np.array([x[2] for x in s])
            order_values[name] = np.array([x[1] for x in s])

    # Report feature means per prior
    print("=" * 70)
    print("FEATURE MEANS BY PRIOR (at matched order)")
    print("=" * 70)
    print()
    print(f"{'Prior':<12} {'Order':<8} {'Edge':<8} {'Symm':<8} {'Spec':<8} {'Comp':<8}")
    print("-" * 60)

    for name in feature_matrices:
        X = feature_matrices[name]
        order_mean = np.mean(order_values[name])
        edge_mean = np.mean(X[:, 0])
        symm_mean = np.mean(X[:, 1])
        spec_mean = np.mean(X[:, 2])
        comp_mean = np.mean(X[:, 3])
        print(f"{name:<12} {order_mean:<8.3f} {edge_mean:<8.3f} {symm_mean:<8.3f} {spec_mean:<8.3f} {comp_mean:<8.3f}")

    print()

    # Statistical tests
    print("=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    print()

    results = {
        'parameters': {
            'n_samples_target': n_samples,
            'order_band': [order_min, order_max],
            'image_size': image_size,
        },
        'sample_counts': {name: len(s) for name, s in samples.items()},
        'pairwise_tests': {},
        'overall_test': None,
    }

    # Pairwise MANOVA tests
    print("PAIRWISE MANOVA (CPPN vs each alternative):")
    print("-" * 60)

    names = list(feature_matrices.keys())

    for name in names:
        if name != 'CPPN' and name in feature_matrices and 'CPPN' in feature_matrices:
            X1 = feature_matrices['CPPN']
            X2 = feature_matrices[name]

            test_result = manova_two_groups(X1, X2)
            cohens_d = cohens_d_multivariate(X1, X2)

            results['pairwise_tests'][f'CPPN_vs_{name}'] = {
                **test_result,
                'cohens_d': cohens_d,
            }

            if 'error' not in test_result:
                print(f"\nCPPN vs {name}:")
                print(f"  Pillai's trace: {test_result['pillai_trace']:.4f}")
                print(f"  F({test_result['df1']}, {test_result['df2']}): {test_result['F_statistic']:.2f}")
                print(f"  P-value: {test_result['p_value']:.6f}")
                print(f"  Cohen's d: {cohens_d:.2f}")

                if test_result['p_value'] < 0.01 and cohens_d > 0.5:
                    print(f"  -> SIGNIFICANT DIFFERENCE (p < 0.01, d > 0.5)")

    # Overall MANOVA
    print("\n" + "-" * 60)
    print("OVERALL MANOVA (all priors simultaneously):")

    groups = [feature_matrices[name] for name in names if name in feature_matrices]
    if len(groups) >= 2:
        overall = manova_multi_groups(groups)
        results['overall_test'] = overall

        if 'error' not in overall:
            print(f"  Pillai's trace: {overall['pillai_trace']:.4f}")
            print(f"  F({overall['df1']}, {overall['df2']}): {overall['F_statistic']:.2f}")
            print(f"  P-value: {overall['p_value']:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count significant pairwise differences
    sig_pairs = []
    max_d = 0
    for pair, res in results['pairwise_tests'].items():
        if 'error' not in res and res['p_value'] < 0.01 and res['cohens_d'] > 0.5:
            sig_pairs.append(pair)
            max_d = max(max_d, res['cohens_d'])

    print(f"\nSignificant pairwise differences: {len(sig_pairs)} / {len(results['pairwise_tests'])}")
    if sig_pairs:
        print(f"  {', '.join(sig_pairs)}")
        print(f"  Maximum effect size: d = {max_d:.2f}")

    if results['overall_test'] and 'error' not in results['overall_test']:
        overall_sig = results['overall_test']['p_value'] < 0.01
        print(f"\nOverall MANOVA significant (p < 0.01): {overall_sig}")
        print(f"  Pillai's trace: {results['overall_test']['pillai_trace']:.4f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if len(sig_pairs) >= 2 and results['overall_test'] and results['overall_test'].get('p_value', 1) < 0.01:
        print("  -> HYPOTHESIS SUPPORTED")
        print("  Different priors achieve similar order through DIFFERENT mechanisms.")
        print("  This reveals the generative-discriminative tradeoff:")
        print("  'Order' is a lossy discriminative measure that compresses away")
        print("  the generative diversity of different structural mechanisms.")
        results['status'] = 'validated'
        results['confidence'] = 'high'
    elif len(sig_pairs) >= 1:
        print("  -> PARTIAL SUPPORT")
        print("  Some priors show different mechanisms but not all.")
        results['status'] = 'inconclusive'
        results['confidence'] = 'medium'
    else:
        print("  -> HYPOTHESIS NOT SUPPORTED")
        print("  Priors appear to generate similar structures at matched order.")
        results['status'] = 'refuted'
        results['confidence'] = 'medium'

    # Save results
    output_dir = Path("results/prior_feature_divergence")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_dir / "divergence_results.json", 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to {output_dir}/divergence_results.json")

    return results


if __name__ == "__main__":
    run_experiment()
