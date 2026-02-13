"""
Edge Orientation Experiment (RES-023)

Hypothesis: High-order CPPN images exhibit more PEAKED edge orientation
distributions (lower entropy in orientation histogram), indicating coherent
directional structure, while low-order images have more UNIFORM distributions
(high entropy) characteristic of noise.

Null hypothesis: Edge orientation entropy is independent of order level.

Novelty:
- RES-007 measured edge DENSITY correlation with order
- RES-017 measured spectral DECAY (frequency distribution)
- This measures edge ORIENTATION distribution - directional structure
  independent of density or frequency content.

Method:
1. Generate N CPPN images with varying order
2. Compute Sobel gradients (Gx, Gy) to get edge orientations
3. Compute orientation histogram (binned angles from arctan2)
4. Calculate histogram entropy as measure of distribution uniformity
5. Correlate entropy with order using Spearman rho
6. Compare low-order vs high-order entropy using Mann-Whitney U

Prediction: High-order images have LOW entropy (peaked histogram with
dominant orientations), low-order images have HIGH entropy (uniform random).
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
from scipy.ndimage import sobel
from typing import Tuple
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_sobel_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Sobel gradients for edge detection.

    Returns:
        Gx: horizontal gradient
        Gy: vertical gradient
    """
    img_float = img.astype(float)
    Gx = sobel(img_float, axis=1)  # Horizontal edges
    Gy = sobel(img_float, axis=0)  # Vertical edges
    return Gx, Gy


def compute_edge_magnitude_and_orientation(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute edge magnitude and orientation using Sobel gradients.

    Returns:
        magnitude: edge strength at each pixel
        orientation: edge angle in radians [-pi, pi]
    """
    Gx, Gy = compute_sobel_gradients(img)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx)  # Range [-pi, pi]
    return magnitude, orientation


def compute_orientation_histogram(orientation: np.ndarray, magnitude: np.ndarray,
                                   n_bins: int = 36, magnitude_threshold: float = 0.1
                                   ) -> np.ndarray:
    """
    Compute histogram of edge orientations weighted by magnitude.

    Args:
        orientation: edge angles in radians [-pi, pi]
        magnitude: edge magnitudes (weights)
        n_bins: number of histogram bins (default 36 = 10 degree bins)
        magnitude_threshold: minimum magnitude to include (filters noise)

    Returns:
        histogram: normalized orientation histogram
    """
    # Filter by magnitude threshold (only count significant edges)
    mask = magnitude > magnitude_threshold

    if not np.any(mask):
        # No significant edges - return uniform histogram
        return np.ones(n_bins) / n_bins

    # Get orientations of significant edges
    valid_orientations = orientation[mask]
    valid_magnitudes = magnitude[mask]

    # Bin edges from -pi to pi
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Compute weighted histogram
    hist, _ = np.histogram(valid_orientations, bins=bin_edges, weights=valid_magnitudes)

    # Normalize to probability distribution
    total = hist.sum()
    if total > 0:
        hist = hist / total
    else:
        hist = np.ones(n_bins) / n_bins

    return hist


def compute_histogram_entropy(hist: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Higher entropy = more uniform distribution
    Lower entropy = more peaked distribution

    Max entropy for n bins = log2(n)
    """
    # Filter out zero probabilities
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(hist * np.log2(hist))

    return entropy


def compute_histogram_peak_ratio(hist: np.ndarray, top_k: int = 4) -> float:
    """
    Compute ratio of top-k bin probabilities to total.

    Higher ratio = more peaked (dominant orientations)
    Lower ratio = more uniform (no dominant orientations)
    """
    sorted_hist = np.sort(hist)[::-1]  # Descending
    return np.sum(sorted_hist[:top_k]) / np.sum(hist) if np.sum(hist) > 0 else 0.0


def compute_dominant_orientation_strength(hist: np.ndarray) -> float:
    """
    Compute strength of dominant orientation.

    Returns ratio of max bin to mean bin.
    Uniform distribution = 1.0, peaked = > 1.0
    """
    if np.sum(hist) == 0:
        return 1.0

    mean_bin = np.mean(hist)
    max_bin = np.max(hist)

    return max_bin / mean_bin if mean_bin > 0 else 1.0


def run_experiment(n_samples: int = 500, image_size: int = 32,
                   n_bins: int = 36, seed: int = 42):
    """
    Run the edge orientation experiment.
    """
    set_global_seed(seed)

    print(f"=== Edge Orientation Experiment ===")
    print(f"N samples: {n_samples}")
    print(f"Image size: {image_size}")
    print(f"Orientation bins: {n_bins} ({360/n_bins:.1f} degrees each)")
    print()

    # Maximum possible entropy for this number of bins
    max_entropy = np.log2(n_bins)
    print(f"Maximum possible entropy: {max_entropy:.3f} bits")
    print()

    # Collect data
    orders = []
    entropies = []
    peak_ratios = []
    dominant_strengths = []
    edge_counts = []  # Number of significant edges

    print("Generating CPPN samples and computing edge orientations...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        # Compute edge orientation analysis
        magnitude, orientation = compute_edge_magnitude_and_orientation(img)

        # Count significant edges (magnitude > threshold)
        n_sig_edges = np.sum(magnitude > 0.1)

        # Compute orientation histogram
        hist = compute_orientation_histogram(orientation, magnitude, n_bins=n_bins)

        # Compute metrics
        entropy = compute_histogram_entropy(hist)
        peak_ratio = compute_histogram_peak_ratio(hist, top_k=4)
        dom_strength = compute_dominant_orientation_strength(hist)

        orders.append(order)
        entropies.append(entropy)
        peak_ratios.append(peak_ratio)
        dominant_strengths.append(dom_strength)
        edge_counts.append(n_sig_edges)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    orders = np.array(orders)
    entropies = np.array(entropies)
    peak_ratios = np.array(peak_ratios)
    dominant_strengths = np.array(dominant_strengths)
    edge_counts = np.array(edge_counts)

    # Normalize entropy to [0, 1] by dividing by max
    normalized_entropies = entropies / max_entropy

    print(f"\nOrder range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"Entropy range: [{entropies.min():.3f}, {entropies.max():.3f}] bits")
    print(f"Normalized entropy range: [{normalized_entropies.min():.3f}, {normalized_entropies.max():.3f}]")
    print(f"Peak ratio range: [{peak_ratios.min():.3f}, {peak_ratios.max():.3f}]")
    print(f"Edge count range: [{edge_counts.min()}, {edge_counts.max()}]")

    # Primary analysis: Spearman correlation between order and entropy
    rho_entropy, p_entropy = stats.spearmanr(orders, entropies)
    print(f"\n--- Primary Analysis: Order vs Orientation Entropy ---")
    print(f"Spearman rho: {rho_entropy:.4f}")
    print(f"Spearman p-value: {p_entropy:.2e}")

    # Also test peak ratio (should anti-correlate with entropy)
    rho_peak, p_peak = stats.spearmanr(orders, peak_ratios)
    print(f"\n--- Secondary Analysis: Order vs Peak Ratio ---")
    print(f"Spearman rho: {rho_peak:.4f}")
    print(f"Spearman p-value: {p_peak:.2e}")

    # Also test dominant strength
    rho_dom, p_dom = stats.spearmanr(orders, dominant_strengths)
    print(f"\n--- Tertiary Analysis: Order vs Dominant Strength ---")
    print(f"Spearman rho: {rho_dom:.4f}")
    print(f"Spearman p-value: {p_dom:.2e}")

    # Split at median for group comparison
    median_order = np.median(orders)
    low_mask = orders < median_order
    high_mask = orders >= median_order

    entropy_low = entropies[low_mask]
    entropy_high = entropies[high_mask]

    print(f"\n--- Low vs High Order Comparison ---")
    print(f"Low order (n={len(entropy_low)}): entropy = {entropy_low.mean():.3f} +/- {entropy_low.std():.3f}")
    print(f"High order (n={len(entropy_high)}): entropy = {entropy_high.mean():.3f} +/- {entropy_high.std():.3f}")

    # Mann-Whitney U test
    U, p_mann_whitney = stats.mannwhitneyu(entropy_low, entropy_high, alternative='two-sided')
    print(f"Mann-Whitney U: {U:.1f}")
    print(f"Mann-Whitney p-value: {p_mann_whitney:.2e}")

    # Cohen's d effect size
    pooled_std = np.sqrt((entropy_low.std()**2 + entropy_high.std()**2) / 2)
    cohens_d = (entropy_high.mean() - entropy_low.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.3f}")

    # Kruskal-Wallis across quartiles
    q1, q2, q3 = np.percentile(orders, [25, 50, 75])
    q_masks = [
        (orders >= 0) & (orders < q1),
        (orders >= q1) & (orders < q2),
        (orders >= q2) & (orders < q3),
        (orders >= q3)
    ]
    q_groups = [entropies[m] for m in q_masks if np.sum(m) > 0]
    H, p_kruskal = stats.kruskal(*q_groups)
    print(f"\nKruskal-Wallis H: {H:.2f}")
    print(f"Kruskal-Wallis p: {p_kruskal:.2e}")

    # Quartile breakdown
    print(f"\n--- Quartile Analysis ---")
    quartile_entropies = []
    for idx, (low, high, name) in enumerate([(0, q1, "Q1 (lowest order)"),
                                              (q1, q2, "Q2"),
                                              (q2, q3, "Q3"),
                                              (q3, 1.1, "Q4 (highest order)")]):
        mask = (orders >= low) & (orders < high)
        if np.sum(mask) > 0:
            qe = entropies[mask].mean()
            quartile_entropies.append(qe)
            print(f"  {name}: entropy = {qe:.3f} bits (n={np.sum(mask)})")

    # Determine status
    # Criteria: p < 0.01, |effect size| > 0.5
    is_significant = p_entropy < 0.01 and np.abs(cohens_d) > 0.5

    # Determine direction
    if cohens_d < 0:
        direction = "LOWER entropy (more peaked)"
        hypothesis_matches = True  # High order has more coherent orientations
    else:
        direction = "HIGHER entropy (more uniform)"
        hypothesis_matches = False

    print(f"\n=== CONCLUSION ===")
    if is_significant:
        print(f"Effect is SIGNIFICANT (p={p_entropy:.2e}, |d|={np.abs(cohens_d):.3f})")
        print(f"High-order images have {direction} orientation distributions.")

        if hypothesis_matches:
            print("VALIDATED: High-order images have more coherent directional structure.")
            status = "validated"
        else:
            print("REFUTED (direction): High-order images have MORE random orientations.")
            status = "refuted"
    else:
        print(f"INCONCLUSIVE: Effect not significant (p={p_entropy:.2e}, d={cohens_d:.3f})")
        status = "inconclusive"

    # Save results
    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'n_bins': n_bins,
        'seed': seed,
        'max_entropy': float(max_entropy),
        'order_range': [float(orders.min()), float(orders.max())],
        'entropy_range': [float(entropies.min()), float(entropies.max())],

        # Primary results
        'spearman_rho_entropy': float(rho_entropy),
        'spearman_p_entropy': float(p_entropy),
        'spearman_rho_peak': float(rho_peak),
        'spearman_p_peak': float(p_peak),
        'spearman_rho_dominant': float(rho_dom),
        'spearman_p_dominant': float(p_dom),

        # Group comparison
        'entropy_low_mean': float(entropy_low.mean()),
        'entropy_low_std': float(entropy_low.std()),
        'entropy_high_mean': float(entropy_high.mean()),
        'entropy_high_std': float(entropy_high.std()),
        'mann_whitney_U': float(U),
        'mann_whitney_p': float(p_mann_whitney),
        'cohens_d': float(cohens_d),

        # Kruskal-Wallis
        'kruskal_H': float(H),
        'kruskal_p': float(p_kruskal),
        'quartile_entropies': [float(q) for q in quartile_entropies],

        # Additional metrics
        'peak_ratio_low_mean': float(peak_ratios[low_mask].mean()),
        'peak_ratio_high_mean': float(peak_ratios[high_mask].mean()),
        'dominant_strength_low_mean': float(dominant_strengths[low_mask].mean()),
        'dominant_strength_high_mean': float(dominant_strengths[high_mask].mean()),

        'status': status
    }

    # Save to results directory
    results_dir = Path(__file__).parent.parent / "results" / "edge_orientation"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "edge_orientation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'edge_orientation_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
