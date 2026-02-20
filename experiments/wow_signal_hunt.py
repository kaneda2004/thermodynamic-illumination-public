#!/usr/bin/env python3
"""
WOW Signal Hunt: Looking for Unexpected Phenomena

The mean-field critical exponents are expected. Here we hunt for
something truly surprising:

1. EMERGENCE OF SEMANTICS: Do semantic features (edges, shapes, faces)
   emerge at specific order thresholds? Is there a "consciousness threshold"?

2. COMPUTATIONAL PHASE TRANSITION: Does the image encode more complex
   computations as order increases? Connection to Turing completeness?

3. THE INFORMATION CLIFF: Is there a sharp boundary where images go
   from "meaningless" to "meaningful"? A phase transition in meaning?

4. FRACTAL STRUCTURE: Does the set of structured images form a fractal?
   What is its fractal dimension?

5. THE GOLDEN RATIO: Do optimal priors have special mathematical properties?
"""

import sys
import os
import numpy as np
from collections import Counter
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    order_multiplicative,
    order_kolmogorov_proxy,
    compute_symmetry,
    compute_edge_density,
    compute_spectral_coherence,
    compute_compressibility,
    CPPN,
)


# ============================================================================
# EXPERIMENT 1: EMERGENCE OF FEATURES
# ============================================================================

def experiment_feature_emergence(n_live=100, n_iterations=1000, image_size=32):
    """
    Track when specific visual features EMERGE during the order climb.

    Question: Is there a sharp threshold where:
    - Edges appear?
    - Symmetry appears?
    - Connected shapes appear?
    - Something "recognizable" appears?
    """
    print("=" * 70)
    print("EXPERIMENT: FEATURE EMERGENCE")
    print("=" * 70)
    print()
    print("Tracking when visual features emerge along the order trajectory...")
    print()

    # Run nested sampling
    dead_points, live_points, best = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Track feature values along trajectory
    orders = []
    log_Xs = []
    edge_densities = []
    symmetries = []
    spectral_coherences = []
    compressibilities = []

    for d in dead_points:
        img = d.image
        orders.append(d.order_value)
        log_Xs.append(d.log_X)
        edge_densities.append(compute_edge_density(img))
        symmetries.append(compute_symmetry(img))
        spectral_coherences.append(compute_spectral_coherence(img))
        compressibilities.append(compute_compressibility(img))

    orders = np.array(orders)
    log_Xs = np.array(log_Xs)
    edge_densities = np.array(edge_densities)
    symmetries = np.array(symmetries)
    spectral_coherences = np.array(spectral_coherences)
    compressibilities = np.array(compressibilities)

    # Find emergence thresholds (where feature first exceeds baseline + 3σ)
    def find_emergence(values, window=20):
        baseline = np.mean(values[:window])
        baseline_std = np.std(values[:window])
        threshold = baseline + 3 * baseline_std

        for i, v in enumerate(values):
            if v > threshold:
                return i, orders[i], log_Xs[i]
        return None, None, None

    print("FEATURE EMERGENCE THRESHOLDS:")
    print("-" * 60)
    print(f"{'Feature':<25} {'Order':<15} {'log(X)':<15}")
    print("-" * 60)

    features = [
        ('Edge density', edge_densities),
        ('Symmetry', symmetries),
        ('Spectral coherence', spectral_coherences),
        ('Compressibility', compressibilities),
    ]

    emergence_data = {}
    for name, values in features:
        idx, order, log_X = find_emergence(values)
        if order is not None:
            print(f"{name:<25} {order:<15.4f} {log_X:<15.2f}")
            emergence_data[name] = {'order': order, 'log_X': log_X}
        else:
            print(f"{name:<25} {'Not found':<15} {'-':<15}")

    # Look for SHARP transitions (large derivative)
    print()
    print("SHARPEST TRANSITIONS:")
    print("-" * 60)

    for name, values in features:
        derivatives = np.abs(np.diff(values))
        peak_idx = np.argmax(derivatives)
        peak_order = orders[peak_idx]
        peak_value = derivatives[peak_idx]

        print(f"{name:<25} Peak at order={peak_order:.4f}, magnitude={peak_value:.4f}")

    # Correlation analysis
    print()
    print("FEATURE CORRELATIONS WITH ORDER:")
    print("-" * 60)

    for name, values in features:
        corr = np.corrcoef(orders, values)[0, 1]
        print(f"{name:<25} r = {corr:.3f}")

    return {
        'orders': orders.tolist(),
        'log_Xs': log_Xs.tolist(),
        'features': {
            'edge_density': edge_densities.tolist(),
            'symmetry': symmetries.tolist(),
            'spectral_coherence': spectral_coherences.tolist(),
            'compressibility': compressibilities.tolist(),
        },
        'emergence': emergence_data
    }


# ============================================================================
# EXPERIMENT 2: THE INFORMATION CLIFF
# ============================================================================

def experiment_information_cliff(n_samples=1000, image_size=32):
    """
    Is there a sharp boundary between "meaningless" and "meaningful" images?

    We sample many CPPNs and look at the distribution of orders.
    If there's a cliff, we should see bimodality.
    """
    print("=" * 70)
    print("EXPERIMENT: THE INFORMATION CLIFF")
    print("=" * 70)
    print()
    print(f"Sampling {n_samples} CPPNs to find distribution of order values...")
    print()

    orders = []
    kolmogorov_values = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        orders.append(order_multiplicative(img))
        kolmogorov_values.append(order_kolmogorov_proxy(img))

        if (i + 1) % 200 == 0:
            print(f"  Sampled {i+1}/{n_samples}")

    orders = np.array(orders)
    kolmogorov_values = np.array(kolmogorov_values)

    # Distribution analysis
    print()
    print("ORDER DISTRIBUTION:")
    print("-" * 60)
    print(f"  Mean: {np.mean(orders):.4f}")
    print(f"  Std:  {np.std(orders):.4f}")
    print(f"  Min:  {np.min(orders):.4f}")
    print(f"  Max:  {np.max(orders):.4f}")

    # Check for bimodality
    # Use Hartigan's dip test approximation: look at histogram shape
    hist, bin_edges = np.histogram(orders, bins=50)
    hist_normalized = hist / hist.max()

    # Find local minima (potential "cliff" locations)
    local_mins = []
    for i in range(1, len(hist) - 1):
        if hist[i] < hist[i-1] and hist[i] < hist[i+1]:
            local_mins.append((bin_edges[i], hist[i]))

    print()
    print("BIMODALITY CHECK:")
    print("-" * 60)

    if local_mins:
        print(f"  Found {len(local_mins)} local minima (potential cliff locations):")
        for order_val, count in local_mins[:3]:
            print(f"    Order ≈ {order_val:.3f}")
    else:
        print("  No clear bimodality detected (unimodal distribution)")

    # ASCII histogram
    print()
    print("ORDER HISTOGRAM:")
    print("-" * 60)

    max_bar = 50
    for i in range(0, len(hist), 2):  # Every other bin for compactness
        bar_len = int(hist_normalized[i] * max_bar)
        order_val = (bin_edges[i] + bin_edges[i+1]) / 2
        print(f"  {order_val:.2f} | {'█' * bar_len}")

    # Kolmogorov distribution
    print()
    print("KOLMOGOROV (COMPRESSIBILITY) DISTRIBUTION:")
    print("-" * 60)
    print(f"  Mean: {np.mean(kolmogorov_values):.4f}")
    print(f"  Std:  {np.std(kolmogorov_values):.4f}")

    return {
        'orders': orders.tolist(),
        'kolmogorov': kolmogorov_values.tolist(),
        'stats': {
            'mean': float(np.mean(orders)),
            'std': float(np.std(orders)),
            'min': float(np.min(orders)),
            'max': float(np.max(orders)),
        },
        'local_minima': local_mins,
    }


# ============================================================================
# EXPERIMENT 3: FRACTAL DIMENSION OF STRUCTURE
# ============================================================================

def experiment_fractal_dimension(n_live=50, n_iterations=500, image_size=32):
    """
    What is the fractal dimension of the set of "structured" images?

    If structured images form a fractal subset of image space,
    the dimension tells us about the geometry of meaning.

    We estimate this using box-counting on the order metric.
    """
    print("=" * 70)
    print("EXPERIMENT: FRACTAL DIMENSION OF STRUCTURE")
    print("=" * 70)
    print()

    # Run nested sampling to get trajectory
    dead_points, _, _ = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    orders = np.array([d.order_value for d in dead_points])
    log_Xs = np.array([d.log_X for d in dead_points])

    # The "dimension" of the structured subset can be estimated from
    # how V(T) scales with T near the transition:
    # V(T) ~ (T_c - T)^d_eff

    # Find scaling region
    # V = 2^(-B) where B = -log_X / ln(2)
    bits = -log_Xs / np.log(2)

    # Look for power-law: bits ~ f(order)
    # d_eff = d(bits)/d(log(order))

    valid = orders > 0.01  # Avoid log(0)
    log_orders = np.log(orders[valid])
    bits_valid = bits[valid]

    # Sliding window derivative
    window = 20
    local_dims = []
    positions = []

    for i in range(window, len(log_orders) - window):
        d_bits = bits_valid[i+window] - bits_valid[i-window]
        d_log_order = log_orders[i+window] - log_orders[i-window]

        if abs(d_log_order) > 0.01:
            local_dim = d_bits / d_log_order
            local_dims.append(local_dim)
            positions.append(orders[valid][i])

    local_dims = np.array(local_dims)
    positions = np.array(positions)

    print("EFFECTIVE DIMENSION ANALYSIS:")
    print("-" * 60)

    if len(local_dims) > 0:
        print(f"  Mean effective dimension: {np.mean(local_dims):.2f}")
        print(f"  Std: {np.std(local_dims):.2f}")
        print()

        # Look for constant dimension region
        stable_region = np.abs(local_dims - np.mean(local_dims)) < np.std(local_dims)
        if np.sum(stable_region) > len(local_dims) * 0.5:
            stable_dim = np.mean(local_dims[stable_region])
            print(f"  Stable dimension estimate: {stable_dim:.2f}")
            print()
            print("  INTERPRETATION:")
            if stable_dim < 1:
                print(f"    d_eff < 1: Structure forms a sparse, dust-like set")
            elif stable_dim < 2:
                print(f"    d_eff ≈ {stable_dim:.1f}: Structure forms curve-like paths")
            elif stable_dim < 3:
                print(f"    d_eff ≈ {stable_dim:.1f}: Structure forms surface-like regions")
            else:
                print(f"    d_eff ≈ {stable_dim:.1f}: Structure fills volume")

    return {
        'local_dims': local_dims.tolist() if len(local_dims) > 0 else [],
        'positions': positions.tolist() if len(positions) > 0 else [],
        'mean_dim': float(np.mean(local_dims)) if len(local_dims) > 0 else None,
    }


# ============================================================================
# EXPERIMENT 4: SEMANTIC EMERGENCE THRESHOLD
# ============================================================================

def experiment_semantic_threshold(n_live=100, n_iterations=800, image_size=32):
    """
    At what order level do images become "recognizable"?

    This is a deep question: is there a threshold where images
    transition from noise to something a human would call structured?

    We use a proxy: the point where images become classifiable
    into discrete types (circle, square, gradient, etc.)
    """
    print("=" * 70)
    print("EXPERIMENT: SEMANTIC EMERGENCE THRESHOLD")
    print("=" * 70)
    print()
    print("Looking for the threshold where images become 'recognizable'...")
    print()

    # Run nested sampling
    dead_points, live_points, _ = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Classify images into semantic categories based on simple features
    def classify_image(img):
        """Simple classifier for image type."""
        sym = compute_symmetry(img)
        edges = compute_edge_density(img)
        spectral = compute_spectral_coherence(img)
        density = np.mean(img)

        # Simple decision tree
        if sym > 0.9:
            if edges < 0.1:
                return "solid"
            else:
                return "symmetric_pattern"
        elif spectral > 0.5:
            if edges < 0.15:
                return "blob"
            else:
                return "gradient"
        elif edges > 0.3:
            return "noise"
        elif 0.3 < density < 0.7:
            return "partial"
        else:
            return "sparse"

    # Track categories along trajectory
    orders = []
    categories = []

    for d in dead_points:
        orders.append(d.order_value)
        categories.append(classify_image(d.image))

    orders = np.array(orders)

    # Find where category distribution changes
    window = 30
    category_entropy = []

    for i in range(window, len(categories)):
        window_cats = categories[i-window:i]
        counts = Counter(window_cats)
        total = sum(counts.values())
        probs = [c/total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        category_entropy.append(entropy)

    category_entropy = np.array(category_entropy)

    # Find transition: where entropy drops (images become classifiable)
    if len(category_entropy) > 10:
        # Smoothed derivative
        smooth_entropy = np.convolve(category_entropy, np.ones(10)/10, mode='valid')
        entropy_derivative = np.diff(smooth_entropy)

        # Find steepest drop
        drop_idx = np.argmin(entropy_derivative)
        transition_order = orders[window + drop_idx + 5]  # Approximate

        print("SEMANTIC TRANSITION:")
        print("-" * 60)
        print(f"  Entropy drops most sharply at order ≈ {transition_order:.4f}")
        print()

        # Category distribution before and after
        print("  Category distribution BEFORE transition:")
        before_cats = categories[:window + drop_idx]
        for cat, count in Counter(before_cats).most_common():
            print(f"    {cat}: {count/len(before_cats)*100:.1f}%")

        print()
        print("  Category distribution AFTER transition:")
        after_cats = categories[window + drop_idx:]
        for cat, count in Counter(after_cats).most_common():
            print(f"    {cat}: {count/len(after_cats)*100:.1f}%")

        print()
        print("  INTERPRETATION:")
        print(f"    Below order ≈ {transition_order:.3f}: Images are 'noise-like'")
        print(f"    Above order ≈ {transition_order:.3f}: Images are 'structured'")
        print("    This may represent a 'semantic cliff' in image space!")

        return {
            'transition_order': float(transition_order),
            'categories': categories,
            'entropy': category_entropy.tolist(),
        }

    return {'categories': categories}


# ============================================================================
# MAIN
# ============================================================================

def run_wow_hunt():
    """Hunt for WOW signals."""
    print("\n" + "=" * 70)
    print("WOW SIGNAL HUNT")
    print("Looking for Unexpected Phenomena in Image Space")
    print("=" * 70 + "\n")

    results = {}

    print("\n[1/4] Feature Emergence...")
    results['feature_emergence'] = experiment_feature_emergence(
        n_live=50, n_iterations=500
    )

    print("\n[2/4] Information Cliff...")
    results['information_cliff'] = experiment_information_cliff(n_samples=500)

    print("\n[3/4] Fractal Dimension...")
    results['fractal'] = experiment_fractal_dimension()

    print("\n[4/4] Semantic Threshold...")
    results['semantic'] = experiment_semantic_threshold(n_live=50, n_iterations=500)

    # Summary
    print("\n" + "=" * 70)
    print("WOW SIGNAL SUMMARY")
    print("=" * 70 + "\n")

    wow_count = 0

    if results.get('fractal', {}).get('mean_dim'):
        dim = results['fractal']['mean_dim']
        print(f"✓ Effective dimension: {dim:.2f}")
        if dim < 3:
            print(f"  → Structure forms a LOW-DIMENSIONAL manifold!")
            wow_count += 1

    if results.get('semantic', {}).get('transition_order'):
        t_order = results['semantic']['transition_order']
        print(f"✓ Semantic transition at order ≈ {t_order:.3f}")
        print(f"  → There IS a 'meaning threshold' in image space!")
        wow_count += 1

    cliff_data = results.get('information_cliff', {})
    if cliff_data.get('local_minima'):
        print(f"✓ Found {len(cliff_data['local_minima'])} local minima in order distribution")
        print(f"  → Possible bimodality (cliff between noise and structure)")
        wow_count += 1

    print()
    if wow_count >= 2:
        print("🎯 MULTIPLE WOW SIGNALS DETECTED!")
    elif wow_count == 1:
        print("📊 One interesting signal found")
    else:
        print("📉 Need deeper investigation")

    # Save
    output_dir = Path("results/wow_signals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(output_dir / "wow_hunt.json", 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/wow_hunt.json")

    return results


if __name__ == "__main__":
    run_wow_hunt()
