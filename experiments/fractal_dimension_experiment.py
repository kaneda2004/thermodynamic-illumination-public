"""
Fractal Dimension Experiment (RES-023)

Hypothesis: High-order CPPN images exhibit fractal dimension properties approaching
natural image characteristics (D ~ 2.3 for binary images). The fractal dimension
should correlate positively with order level.

Null hypothesis: Box-counting fractal dimension is independent of order level.

Background:
- Natural images often exhibit fractal characteristics with dimension ~2.3
- Box-counting dimension measures how detail scales with resolution
- For a 2D binary image: D_box = lim(log(N(s))/log(1/s)) where N(s) = number of boxes
  of size s needed to cover the foreground
- Random noise has D ~ 2.0 (fills the plane)
- Smooth curves have D ~ 1.0
- Structured images with edges may have D ~ 1.3-1.8

Method:
1. Generate N CPPN images with varying order
2. Compute box-counting fractal dimension using multiple box sizes
3. Correlate D_box with order using Spearman rho
4. Compare low-order vs high-order D_box using Mann-Whitney U

This is a NEW domain (fractal_dimension) not previously explored in the research log.
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
from typing import Tuple, List
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_box_count(binary_img: np.ndarray, box_size: int) -> int:
    """
    Count the number of boxes of given size that contain at least one foreground pixel.

    Args:
        binary_img: Binary image (0/1)
        box_size: Size of the box in pixels

    Returns:
        Number of non-empty boxes
    """
    h, w = binary_img.shape
    count = 0

    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            # Get the box region (handle edge cases)
            box = binary_img[i:min(i+box_size, h), j:min(j+box_size, w)]
            if np.any(box > 0):
                count += 1

    return count


def compute_fractal_dimension(img: np.ndarray, min_box_size: int = 2, max_box_size: int = None) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Compute box-counting fractal dimension of a binary image.

    Uses linear regression on log-log plot of box size vs box count.
    D_box = -slope of log(N) vs log(s)

    Args:
        img: Binary image (will be converted to 0/1)
        min_box_size: Minimum box size to use
        max_box_size: Maximum box size (defaults to image_size // 2)

    Returns:
        D_box: Estimated fractal dimension
        r_squared: Goodness of fit
        std_err: Standard error of the estimate
        box_sizes: List of box sizes used
        box_counts: List of corresponding box counts
    """
    binary = (img > 0).astype(np.uint8)
    h, w = binary.shape

    if max_box_size is None:
        max_box_size = min(h, w) // 2

    # Generate box sizes (powers of 2 work well)
    box_sizes = []
    size = min_box_size
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    # Also add some intermediate sizes for better fit
    for size in [3, 5, 6, 10, 12]:
        if min_box_size < size < max_box_size and size not in box_sizes:
            box_sizes.append(size)

    box_sizes = sorted(box_sizes)

    if len(box_sizes) < 3:
        return 0.0, 0.0, 0.0, [], []

    # Compute box counts
    box_counts = []
    for size in box_sizes:
        count = compute_box_count(binary, size)
        box_counts.append(count)

    # Filter out zero counts
    valid = [(s, c) for s, c in zip(box_sizes, box_counts) if c > 0]
    if len(valid) < 3:
        return 0.0, 0.0, 0.0, box_sizes, box_counts

    valid_sizes, valid_counts = zip(*valid)

    # Linear regression on log-log plot
    # D = -slope because N(s) ~ s^(-D)
    log_sizes = np.log(np.array(valid_sizes))
    log_counts = np.log(np.array(valid_counts))

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)

    # Fractal dimension is the negative of the slope
    D_box = -slope

    return D_box, r_value ** 2, std_err, list(valid_sizes), list(valid_counts)


def run_experiment(n_samples: int = 500, image_size: int = 32, seed: int = 42, density_band: Tuple[float, float] = (0.1, 0.9)):
    """
    Run the fractal dimension experiment.

    Args:
        n_samples: Number of samples to generate
        image_size: Size of generated images
        seed: Random seed
        density_band: Only include images with density in this range (to exclude trivial all-black/all-white)
    """
    set_global_seed(seed)

    print(f"=== Fractal Dimension Experiment ===")
    print(f"N samples: {n_samples}")
    print(f"Image size: {image_size}")
    print(f"Density band: {density_band}")
    print()

    # Collect data
    orders = []
    fractal_dims = []
    r_squareds = []
    densities = []
    edge_densities = []

    print("Generating CPPN samples and computing fractal dimensions...")
    generated = 0
    attempts = 0
    max_attempts = n_samples * 5  # Avoid infinite loop

    while generated < n_samples and attempts < max_attempts:
        cppn = CPPN()
        img = cppn.render(image_size)
        attempts += 1

        # Check density is in valid band
        density = np.mean(img)
        if density < density_band[0] or density > density_band[1]:
            continue

        order = order_multiplicative(img)
        D_box, r_sq, std_err, _, _ = compute_fractal_dimension(img)

        # Simple edge density
        h_edges = np.sum(img[:, :-1] != img[:, 1:])
        v_edges = np.sum(img[:-1, :] != img[1:, :])
        edge_density = (h_edges + v_edges) / (2 * img.size)

        orders.append(order)
        fractal_dims.append(D_box)
        r_squareds.append(r_sq)
        densities.append(density)
        edge_densities.append(edge_density)
        generated += 1

        if generated % 100 == 0:
            print(f"  Processed {generated}/{n_samples} (attempts: {attempts})")

    orders = np.array(orders)
    fractal_dims = np.array(fractal_dims)
    r_squareds = np.array(r_squareds)
    densities = np.array(densities)
    edge_densities = np.array(edge_densities)

    print(f"\nOrder range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"Fractal dimension range: [{fractal_dims.min():.3f}, {fractal_dims.max():.3f}]")
    print(f"Mean fractal dimension: {fractal_dims.mean():.3f} +/- {fractal_dims.std():.3f}")
    print(f"Mean R^2 of power law fits: {r_squareds.mean():.3f}")

    # Reference values
    print(f"\n--- Reference Values ---")
    print(f"Natural image fractal dimension (typical): ~2.3")
    print(f"Random noise fractal dimension (theoretical): ~2.0")
    print(f"Smooth curve fractal dimension: ~1.0")

    # Primary analysis: Spearman correlation between order and fractal dimension
    rho, p_spearman = stats.spearmanr(orders, fractal_dims)
    print(f"\n--- Primary Analysis: Order vs Fractal Dimension Correlation ---")
    print(f"Spearman rho: {rho:.4f}")
    print(f"Spearman p-value: {p_spearman:.2e}")

    # Pearson for comparison
    r_pearson, p_pearson = stats.pearsonr(orders, fractal_dims)
    print(f"Pearson r: {r_pearson:.4f}")
    print(f"Pearson p-value: {p_pearson:.2e}")

    # Secondary analysis: Low vs High order comparison
    median_order = np.median(orders)
    low_mask = orders < median_order
    high_mask = orders >= median_order

    D_low = fractal_dims[low_mask]
    D_high = fractal_dims[high_mask]

    print(f"\n--- Secondary Analysis: Low vs High Order ---")
    print(f"Low order (n={len(D_low)}): D_box = {D_low.mean():.3f} +/- {D_low.std():.3f}")
    print(f"High order (n={len(D_high)}): D_box = {D_high.mean():.3f} +/- {D_high.std():.3f}")

    # Mann-Whitney U test
    U, p_mann_whitney = stats.mannwhitneyu(D_low, D_high, alternative='two-sided')
    print(f"Mann-Whitney U: {U:.1f}")
    print(f"Mann-Whitney p-value: {p_mann_whitney:.2e}")

    # Cohen's d effect size
    pooled_std = np.sqrt((D_low.std()**2 + D_high.std()**2) / 2)
    cohens_d = (D_high.mean() - D_low.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.3f}")

    # Tertiary analysis: Distance from natural image characteristics
    natural_D = 2.3
    dist_low = np.abs(D_low.mean() - natural_D)
    dist_high = np.abs(D_high.mean() - natural_D)

    print(f"\n--- Tertiary Analysis: Distance from Natural Image D (~2.3) ---")
    print(f"Low order distance from natural: {dist_low:.3f}")
    print(f"High order distance from natural: {dist_high:.3f}")
    print(f"High order closer to natural? {dist_high < dist_low}")

    # Quartile analysis for gradient
    q1, q2, q3 = np.percentile(orders, [25, 50, 75])
    quartile_Ds = []
    print(f"\n--- Quartile Analysis ---")
    for low, high, name in [(0, q1, "Q1 (lowest order)"), (q1, q2, "Q2"), (q2, q3, "Q3"), (q3, 1.1, "Q4 (highest order)")]:
        mask = (orders >= low) & (orders < high)
        if np.sum(mask) > 0:
            qD = fractal_dims[mask].mean()
            qD_std = fractal_dims[mask].std()
            quartile_Ds.append(qD)
            print(f"  {name}: D_box = {qD:.3f} +/- {qD_std:.3f} (n={np.sum(mask)})")

    # Kruskal-Wallis for monotonic trend
    q_masks = [
        (orders >= 0) & (orders < q1),
        (orders >= q1) & (orders < q2),
        (orders >= q2) & (orders < q3),
        (orders >= q3)
    ]
    q_groups = [fractal_dims[m] for m in q_masks if np.sum(m) > 0]
    H, p_kruskal = stats.kruskal(*q_groups)
    print(f"\nKruskal-Wallis H: {H:.2f}")
    print(f"Kruskal-Wallis p: {p_kruskal:.2e}")

    # Additional analysis: correlation with other features
    print(f"\n--- Feature Correlations ---")
    rho_density, p_density = stats.spearmanr(fractal_dims, densities)
    rho_edges, p_edges = stats.spearmanr(fractal_dims, edge_densities)
    print(f"D_box vs density: rho={rho_density:.3f}, p={p_density:.2e}")
    print(f"D_box vs edge_density: rho={rho_edges:.3f}, p={p_edges:.2e}")

    # Partial correlation: D_box vs order controlling for edge_density
    # Use residuals method
    slope_D_e, _, _, _, _ = stats.linregress(edge_densities, fractal_dims)
    slope_O_e, _, _, _, _ = stats.linregress(edge_densities, orders)
    resid_D = fractal_dims - slope_D_e * edge_densities
    resid_O = orders - slope_O_e * edge_densities
    rho_partial, p_partial = stats.spearmanr(resid_D, resid_O)
    print(f"Partial correlation (D_box vs order | edge_density): rho={rho_partial:.3f}, p={p_partial:.2e}")

    # Determine status
    is_significant = p_spearman < 0.01 and np.abs(cohens_d) > 0.5

    print(f"\n=== CONCLUSION ===")
    if is_significant:
        direction = "higher" if cohens_d > 0 else "lower"
        print(f"VALIDATED: High-order images have {direction} fractal dimension.")
        print(f"Spearman rho={rho:.3f}, Cohen's d={cohens_d:.3f}")
        status = "validated"
    else:
        if p_spearman < 0.01:
            print(f"INCONCLUSIVE: Significant correlation (p={p_spearman:.2e}) but small effect (d={cohens_d:.3f})")
        elif np.abs(cohens_d) > 0.5:
            print(f"INCONCLUSIVE: Large effect (d={cohens_d:.3f}) but not significant (p={p_spearman:.2e})")
        else:
            print(f"REFUTED: No significant relationship (p={p_spearman:.2e}, d={cohens_d:.3f})")
        status = "inconclusive" if p_spearman < 0.05 or np.abs(cohens_d) > 0.3 else "refuted"

    # Summary
    print(f"\nKey finding: Mean D_box = {fractal_dims.mean():.2f} (range {fractal_dims.min():.2f}-{fractal_dims.max():.2f})")
    print(f"This is {'close to' if np.abs(fractal_dims.mean() - natural_D) < 0.5 else 'far from'} natural image D (~2.3)")

    # Save results
    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'seed': seed,
        'density_band': list(density_band),
        'order_range': [float(orders.min()), float(orders.max())],
        'fractal_dim_range': [float(fractal_dims.min()), float(fractal_dims.max())],
        'fractal_dim_mean': float(fractal_dims.mean()),
        'fractal_dim_std': float(fractal_dims.std()),
        'mean_r_squared': float(r_squareds.mean()),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'D_low_mean': float(D_low.mean()),
        'D_low_std': float(D_low.std()),
        'D_high_mean': float(D_high.mean()),
        'D_high_std': float(D_high.std()),
        'mann_whitney_U': float(U),
        'mann_whitney_p': float(p_mann_whitney),
        'cohens_d': float(cohens_d),
        'natural_D': natural_D,
        'dist_low_from_natural': float(dist_low),
        'dist_high_from_natural': float(dist_high),
        'high_order_closer_to_natural': bool(dist_high < dist_low),
        'kruskal_H': float(H),
        'kruskal_p': float(p_kruskal),
        'quartile_Ds': quartile_Ds,
        'partial_correlation_rho': float(rho_partial),
        'partial_correlation_p': float(p_partial),
        'status': status
    }

    # Save to results directory
    results_dir = Path(__file__).parent.parent / "results" / "fractal_dimension"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "fractal_dimension_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'fractal_dimension_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
