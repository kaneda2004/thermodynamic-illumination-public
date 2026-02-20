"""
RES-150: Scale invariance analysis of CPPN images.

Hypothesis: High-order CPPN images exhibit scale invariance with similar structure
at multiple zoom levels.

Method:
1. Generate CPPN images at multiple resolutions (64, 128, 256)
2. Compute structural similarity between downsampled high-res and native low-res
3. Measure fractal dimension via box-counting
4. Test if high-order images have higher scale invariance correlation
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats
from scipy.ndimage import zoom

def render_at_resolution(cppn: CPPN, size: int) -> np.ndarray:
    """Render CPPN at specified resolution."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return (cppn.activate(x, y) > 0.5).astype(np.uint8)

def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity between two images (same size)."""
    # Simple pixel-wise correlation for binary images
    if img1.shape != img2.shape:
        # Resize img2 to match img1
        img2 = zoom(img2, (img1.shape[0]/img2.shape[0], img1.shape[1]/img2.shape[1]), order=0)
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging blocks."""
    h, w = img.shape
    new_h, new_w = h // factor, w // factor
    reshaped = img[:new_h*factor, :new_w*factor].reshape(new_h, factor, new_w, factor)
    return (reshaped.mean(axis=(1, 3)) > 0.5).astype(np.uint8)

def box_counting_dimension(img: np.ndarray) -> float:
    """Estimate fractal dimension via box-counting."""
    sizes = [2, 4, 8, 16]
    counts = []

    for size in sizes:
        if size > min(img.shape):
            continue
        # Count non-empty boxes
        h, w = img.shape
        count = 0
        for i in range(0, h, size):
            for j in range(0, w, size):
                box = img[i:min(i+size, h), j:min(j+size, w)]
                if box.sum() > 0:
                    count += 1
        counts.append(count)

    # Fit log-log line
    if len(counts) < 2:
        return 2.0
    valid_sizes = [s for s in sizes if s <= min(img.shape)][:len(counts)]
    log_sizes = np.log(valid_sizes)
    log_counts = np.log(counts)
    slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
    return -slope

def compute_scale_invariance(cppn: CPPN) -> dict:
    """Compute scale invariance metrics for a CPPN."""
    # Render at multiple resolutions
    img_64 = render_at_resolution(cppn, 64)
    img_128 = render_at_resolution(cppn, 128)
    img_256 = render_at_resolution(cppn, 256)

    # Downsample high-res and compare to low-res
    img_128_to_64 = downsample(img_128, 2)
    img_256_to_64 = downsample(img_256, 4)
    img_256_to_128 = downsample(img_256, 2)

    # Structural similarity across scales
    sim_64_128 = structural_similarity(img_64, img_128_to_64)
    sim_64_256 = structural_similarity(img_64, img_256_to_64)
    sim_128_256 = structural_similarity(img_128, img_256_to_128)

    # Average scale invariance
    scale_invariance = np.nanmean([sim_64_128, sim_64_256, sim_128_256])

    # Fractal dimension
    fractal_dim = box_counting_dimension(img_64)

    # Order at base resolution
    order = order_multiplicative(img_64)

    return {
        'order': order,
        'scale_invariance': scale_invariance,
        'fractal_dim': fractal_dim,
        'sim_64_128': sim_64_128,
        'sim_64_256': sim_64_256,
        'sim_128_256': sim_128_256
    }

def main():
    set_global_seed(42)
    n_samples = 500

    results = []
    for i in range(n_samples):
        cppn = CPPN()  # Random initialization
        metrics = compute_scale_invariance(cppn)
        results.append(metrics)
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{n_samples}")

    # Extract arrays
    orders = np.array([r['order'] for r in results])
    scale_inv = np.array([r['scale_invariance'] for r in results])
    fractal_dims = np.array([r['fractal_dim'] for r in results])

    # Filter out NaN values
    valid_mask = ~np.isnan(scale_inv)
    orders_valid = orders[valid_mask]
    scale_inv_valid = scale_inv[valid_mask]
    fractal_dims_valid = fractal_dims[valid_mask]

    print(f"\n=== RES-150: Scale Invariance Analysis ===")
    print(f"Valid samples: {len(orders_valid)}/{n_samples}")

    # Correlation: order vs scale invariance
    r_scale, p_scale = stats.pearsonr(orders_valid, scale_inv_valid)
    print(f"\nOrder vs Scale Invariance:")
    print(f"  r = {r_scale:.4f}, p = {p_scale:.2e}")

    # Correlation: order vs fractal dimension
    r_fractal, p_fractal = stats.pearsonr(orders_valid, fractal_dims_valid)
    print(f"\nOrder vs Fractal Dimension:")
    print(f"  r = {r_fractal:.4f}, p = {p_fractal:.2e}")

    # Split into high/low order groups
    median_order = np.median(orders_valid)
    high_order = scale_inv_valid[orders_valid > median_order]
    low_order = scale_inv_valid[orders_valid <= median_order]

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_order.var() + low_order.var()) / 2)
    cohens_d = (high_order.mean() - low_order.mean()) / (pooled_std + 1e-10)

    # t-test
    t_stat, p_ttest = stats.ttest_ind(high_order, low_order)

    print(f"\nHigh-order vs Low-order Scale Invariance:")
    print(f"  High mean: {high_order.mean():.4f}")
    print(f"  Low mean: {low_order.mean():.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  t-test p: {p_ttest:.2e}")

    # Conclusion
    print("\n=== CONCLUSION ===")
    if abs(cohens_d) >= 0.5 and p_scale < 0.01:
        status = "VALIDATED"
        direction = "higher" if cohens_d > 0 else "lower"
        print(f"VALIDATED: High-order CPPNs have {direction} scale invariance")
    else:
        status = "REFUTED"
        print(f"REFUTED: No significant relationship between order and scale invariance")

    print(f"\nMetrics for log:")
    print(f"  correlation_r: {r_scale:.4f}")
    print(f"  correlation_p: {p_scale:.2e}")
    print(f"  cohens_d: {cohens_d:.4f}")
    print(f"  mean_high_order: {high_order.mean():.4f}")
    print(f"  mean_low_order: {low_order.mean():.4f}")

    return {
        'status': status,
        'r': r_scale,
        'p': p_scale,
        'd': cohens_d,
        'fractal_r': r_fractal,
        'fractal_p': p_fractal
    }

if __name__ == "__main__":
    main()
