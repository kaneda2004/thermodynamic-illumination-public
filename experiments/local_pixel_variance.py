"""
RES-146: CPPN images have lower local pixel variance in neighborhoods than random images

Hypothesis: CPPN images produce smooth regions with consistent pixel values,
leading to lower variance within local neighborhoods (e.g., 3x3 patches).
Random images have independent pixels, so local variance should be high.
"""

import numpy as np
from scipy import ndimage
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, set_global_seed

def compute_local_variance(img, window_size=3):
    """Compute mean local variance using sliding window."""
    # Local mean
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = ndimage.convolve(img, kernel, mode='reflect')
    # Local variance = E[X^2] - E[X]^2
    local_mean_sq = ndimage.convolve(img**2, kernel, mode='reflect')
    local_var = local_mean_sq - local_mean**2
    return np.mean(local_var)

def main():
    set_global_seed(42)
    n_samples = 500
    resolution = 32

    cppn_local_vars = []
    random_local_vars = []

    # Generate CPPN images
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(resolution)
        local_var = compute_local_variance(img)
        cppn_local_vars.append(local_var)

    # Generate random images
    for i in range(n_samples):
        img = np.random.rand(resolution, resolution)
        local_var = compute_local_variance(img)
        random_local_vars.append(local_var)

    cppn_arr = np.array(cppn_local_vars)
    random_arr = np.array(random_local_vars)

    # Statistics
    cppn_mean = np.mean(cppn_arr)
    random_mean = np.mean(random_arr)

    # Cohen's d
    pooled_std = np.sqrt((np.var(cppn_arr) + np.var(random_arr)) / 2)
    cohens_d = (cppn_mean - random_mean) / pooled_std

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(cppn_arr, random_arr, alternative='two-sided')

    # Effect size ratio
    ratio = random_mean / cppn_mean if cppn_mean > 0 else float('inf')

    print(f"=== RES-146: Local Pixel Variance ===")
    print(f"CPPN local variance:   {cppn_mean:.6f} ± {np.std(cppn_arr):.6f}")
    print(f"Random local variance: {random_mean:.6f} ± {np.std(random_arr):.6f}")
    print(f"Ratio (random/CPPN):   {ratio:.2f}x")
    print(f"Cohen's d:             {cohens_d:.3f}")
    print(f"Mann-Whitney p-value:  {p_value:.2e}")
    print(f"")
    print(f"CPPN min/max: {np.min(cppn_arr):.6f} / {np.max(cppn_arr):.6f}")
    print(f"Random min/max: {np.min(random_arr):.6f} / {np.max(random_arr):.6f}")

    # Also test different window sizes
    print(f"\n=== Window Size Sensitivity ===")
    for ws in [3, 5, 7]:
        cppn_vars = [compute_local_variance(CPPN().render(resolution), ws) for _ in range(100)]
        rand_vars = [compute_local_variance(np.random.rand(resolution, resolution), ws) for _ in range(100)]
        c_mean, r_mean = np.mean(cppn_vars), np.mean(rand_vars)
        d = (c_mean - r_mean) / np.sqrt((np.var(cppn_vars) + np.var(rand_vars)) / 2)
        print(f"Window {ws}x{ws}: CPPN={c_mean:.4f}, Random={r_mean:.4f}, d={d:.2f}")

    # Investigate if CPPN images are binary (explaining zero variance)
    print(f"\n=== Image Properties Check ===")
    test_imgs = [CPPN().render(resolution) for _ in range(50)]
    unique_vals = [len(np.unique(img)) for img in test_imgs]
    print(f"Unique pixel values in CPPN images: mean={np.mean(unique_vals):.1f}, range=[{min(unique_vals)}, {max(unique_vals)}]")

    # Check if images are mostly flat
    def edge_frac(img):
        h_edges = np.abs(np.diff(img.astype(float), axis=0))
        v_edges = np.abs(np.diff(img.astype(float), axis=1))
        return (np.sum(h_edges) + np.sum(v_edges)) / (2 * img.size)
    edge_fraction = [edge_frac(img) for img in test_imgs]
    print(f"Edge fraction (transition rate): {np.mean(edge_fraction):.4f}")

    # Compare with grayscale CPPN output (before binarization)
    print(f"\n=== Grayscale Analysis (pre-binarization) ===")
    cppn_gray_vars = []
    for _ in range(200):
        cppn = CPPN()
        # Get raw grayscale output
        y_coords = np.linspace(-1, 1, resolution)
        x_coords = np.linspace(-1, 1, resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)
        raw_img = cppn.activate(xx.flatten(), yy.flatten()).reshape(resolution, resolution)
        cppn_gray_vars.append(compute_local_variance(raw_img))

    rand_gray = [compute_local_variance(np.random.rand(resolution, resolution)) for _ in range(200)]

    gray_cppn_mean = np.mean(cppn_gray_vars)
    gray_rand_mean = np.mean(rand_gray)
    gray_d = (gray_cppn_mean - gray_rand_mean) / np.sqrt((np.var(cppn_gray_vars) + np.var(rand_gray)) / 2)
    _, gray_p = stats.mannwhitneyu(cppn_gray_vars, rand_gray)

    print(f"CPPN grayscale local variance: {gray_cppn_mean:.6f}")
    print(f"Random grayscale local variance: {gray_rand_mean:.6f}")
    print(f"Cohen's d (grayscale): {gray_d:.3f}")
    print(f"p-value (grayscale): {gray_p:.2e}")

    # Verdict
    print(f"\n=== VERDICT ===")
    if cohens_d < -0.5 and p_value < 0.01:
        print("VALIDATED: CPPN images have significantly lower local variance")
        print(f"Binary images: d={cohens_d:.2f}")
        print(f"Grayscale images: d={gray_d:.2f}")
    elif cohens_d > 0.5 and p_value < 0.01:
        print("REFUTED: CPPN images have HIGHER local variance (opposite direction)")
    else:
        print(f"INCONCLUSIVE: Effect size |d|={abs(cohens_d):.2f} < 0.5 or p={p_value:.2e} >= 0.01")

if __name__ == "__main__":
    main()
