"""
RES-154: Order metric gradient magnitude varies >3x across input image space regions

Hypothesis: The order_multiplicative metric has non-uniform sensitivity - gradient
magnitude varies significantly depending on where in image space we are.

Method:
1. Generate diverse images spanning low/mid/high order regions
2. Compute numerical gradient (finite differences) at each image
3. Compare gradient magnitudes across order regions
4. Test if variance is >3x between regions

This matters because non-uniform gradients affect optimization/sampling dynamics.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats

np.random.seed(42)

def compute_gradient_magnitude(img: np.ndarray, eps: float = 0.01) -> float:
    """Compute gradient magnitude via finite differences on random pixel perturbations."""
    base_order = order_multiplicative(img)
    gradients = []

    # Sample random pixel perturbations
    for _ in range(50):
        perturbed = img.copy()
        i, j = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
        perturbed[i, j] = 1 - perturbed[i, j]  # Flip pixel
        new_order = order_multiplicative(perturbed)
        gradients.append(abs(new_order - base_order))

    return np.mean(gradients)

# Generate diverse images
print("Generating images across order space...")
n_samples = 300
images = []
orders = []
gradient_mags = []

# Mix of random and CPPN images for diversity
for i in range(n_samples):
    if i < n_samples // 3:
        # Random images (typically low order)
        img = (np.random.rand(32, 32) > 0.5).astype(np.uint8)
    elif i < 2 * n_samples // 3:
        # CPPN images (variable order)
        cppn = CPPN()
        img = cppn.render(32)
    else:
        # Gaussian-blurred random (mid-order)
        from scipy.ndimage import gaussian_filter
        raw = np.random.rand(32, 32)
        blurred = gaussian_filter(raw, sigma=np.random.uniform(1, 4))
        img = (blurred > 0.5).astype(np.uint8)

    order = order_multiplicative(img)
    images.append(img)
    orders.append(order)

print(f"Order distribution: min={min(orders):.4f}, max={max(orders):.4f}, mean={np.mean(orders):.4f}")

# Compute gradients
print("Computing gradient magnitudes...")
for img in images:
    gradient_mags.append(compute_gradient_magnitude(img))

orders = np.array(orders)
gradient_mags = np.array(gradient_mags)

# Bin by order regions
low_mask = orders < 0.1
mid_mask = (orders >= 0.1) & (orders < 0.4)
high_mask = orders >= 0.4

print(f"\nRegion counts: low={np.sum(low_mask)}, mid={np.sum(mid_mask)}, high={np.sum(high_mask)}")

# Compare gradient magnitudes across regions
low_grad = gradient_mags[low_mask] if np.any(low_mask) else np.array([0])
mid_grad = gradient_mags[mid_mask] if np.any(mid_mask) else np.array([0])
high_grad = gradient_mags[high_mask] if np.any(high_mask) else np.array([0])

print(f"\nGradient magnitudes by region:")
print(f"  Low order (<0.1):  mean={np.mean(low_grad):.6f}, std={np.std(low_grad):.6f}")
print(f"  Mid order (0.1-0.4): mean={np.mean(mid_grad):.6f}, std={np.std(mid_grad):.6f}")
print(f"  High order (>0.4): mean={np.mean(high_grad):.6f}, std={np.std(high_grad):.6f}")

# Ratio test
max_mean = max(np.mean(low_grad), np.mean(mid_grad), np.mean(high_grad))
min_mean = min(np.mean(low_grad), np.mean(mid_grad), np.mean(high_grad))
ratio = max_mean / (min_mean + 1e-10)
print(f"\nGradient ratio (max/min): {ratio:.2f}x")

# Statistical tests
# Correlation between order and gradient
corr, p_corr = stats.pearsonr(orders, gradient_mags)
print(f"\nOrder-gradient correlation: r={corr:.4f}, p={p_corr:.2e}")

# ANOVA across regions
if np.sum(low_mask) > 5 and np.sum(mid_mask) > 5 and np.sum(high_mask) > 5:
    f_stat, p_anova = stats.f_oneway(low_grad, mid_grad, high_grad)
    print(f"ANOVA across regions: F={f_stat:.2f}, p={p_anova:.2e}")

# Effect size (Cohen's d) between extreme regions
if len(high_grad) > 5 and len(low_grad) > 5:
    pooled_std = np.sqrt((np.var(high_grad) + np.var(low_grad)) / 2)
    cohens_d = (np.mean(high_grad) - np.mean(low_grad)) / (pooled_std + 1e-10)
    print(f"Cohen's d (high vs low): {cohens_d:.2f}")
else:
    cohens_d = 0

# Final verdict
hypothesis_supported = ratio > 3.0 and p_anova < 0.01

print(f"\n{'='*60}")
print(f"RESULT: {'VALIDATED' if hypothesis_supported else 'REFUTED'}")
print(f"  - Gradient ratio: {ratio:.2f}x (threshold: >3x)")
print(f"  - ANOVA p-value: {p_anova:.2e} (threshold: <0.01)")
print(f"  - Effect size d: {cohens_d:.2f}")
print(f"{'='*60}")
