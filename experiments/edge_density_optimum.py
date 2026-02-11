"""
RES-078: Test if edge_gate center=0.15 is optimal for maximizing order.

Approach:
1. Generate diverse CPPN images and measure their edge density vs order
2. Scan edge_gate centers from 0.05 to 0.35 and evaluate order distribution
3. Use independent ground truth (compressibility, coherence) to validate
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, compute_edge_density, compute_compressibility,
    compute_spectral_coherence, gaussian_gate, order_multiplicative
)

def generate_cppn_image(size=64, seed=None):
    """Generate a CPPN image."""
    if seed is not None:
        np.random.seed(seed)
    cppn = CPPN()
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    vals = cppn.activate(X, Y)
    return (vals > 0.5).astype(int)

def compute_order_with_edge_center(img, edge_center):
    """Compute order using a specific edge_gate center."""
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)

    # Gates
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)
    edge_gate = gaussian_gate(edge_density, center=edge_center, sigma=0.08)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    return density_gate * edge_gate * coherence_gate * compress_gate

def main():
    np.random.seed(42)
    n_samples = 500

    # Generate diverse CPPN images
    print(f"Generating {n_samples} CPPN images...")
    images = [generate_cppn_image(64, seed=i) for i in range(n_samples)]

    # Measure edge density distribution for CPPNs
    edge_densities = [compute_edge_density(img) for img in images]
    print(f"\nCPPN edge density: mean={np.mean(edge_densities):.3f}, std={np.std(edge_densities):.3f}")
    print(f"  Range: [{np.min(edge_densities):.3f}, {np.max(edge_densities):.3f}]")
    print(f"  Median: {np.median(edge_densities):.3f}")

    # Test different edge_gate centers
    centers = np.linspace(0.05, 0.35, 31)
    mean_orders = []
    high_order_counts = []  # Images with order > 0.3

    for center in centers:
        orders = [compute_order_with_edge_center(img, center) for img in images]
        mean_orders.append(np.mean(orders))
        high_order_counts.append(np.sum(np.array(orders) > 0.3))

    # Find optimal center
    optimal_idx = np.argmax(mean_orders)
    optimal_center = centers[optimal_idx]
    optimal_mean = mean_orders[optimal_idx]

    print(f"\n=== EDGE GATE CENTER ANALYSIS ===")
    print(f"Optimal center: {optimal_center:.3f} (mean order: {optimal_mean:.4f})")
    print(f"Current default (0.15): mean order = {mean_orders[10]:.4f}")

    # Statistical comparison: is 0.15 significantly worse than optimal?
    # Use bootstrap to get confidence interval on optimal
    n_bootstrap = 1000
    bootstrap_optima = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        boot_images = [images[i] for i in boot_idx]
        boot_means = []
        for center in centers:
            orders = [compute_order_with_edge_center(img, center) for img in boot_images]
            boot_means.append(np.mean(orders))
        bootstrap_optima.append(centers[np.argmax(boot_means)])

    opt_ci = np.percentile(bootstrap_optima, [2.5, 97.5])
    print(f"Bootstrap 95% CI for optimal center: [{opt_ci[0]:.3f}, {opt_ci[1]:.3f}]")

    in_ci = opt_ci[0] <= 0.15 <= opt_ci[1]
    print(f"0.15 within 95% CI: {in_ci}")

    # Ground truth validation: correlation with independent metrics
    print(f"\n=== GROUND TRUTH VALIDATION ===")
    coherences = [compute_spectral_coherence(img) for img in images]
    compressibilities = [compute_compressibility(img) for img in images]

    # What edge density maximizes coherence?
    bins = np.linspace(0.05, 0.35, 7)
    for i in range(len(bins)-1):
        mask = [(bins[i] <= e < bins[i+1]) for e in edge_densities]
        if sum(mask) > 5:
            coh_in_bin = [c for c, m in zip(coherences, mask) if m]
            print(f"  Edge [{bins[i]:.2f}-{bins[i+1]:.2f}]: n={sum(mask)}, coherence={np.mean(coh_in_bin):.3f}")

    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"CPPN natural edge density mode: ~{np.median(edge_densities):.3f}")
    print(f"Empirically optimal edge_gate center: {optimal_center:.3f}")
    print(f"Current implementation uses: 0.15")
    print(f"Difference: {abs(optimal_center - 0.15):.3f}")

    # Effect size: how much worse is 0.15 vs optimal?
    effect = (optimal_mean - mean_orders[10]) / np.std(mean_orders)
    print(f"Effect size (Cohen's d): {effect:.3f}")

    # Verdict
    is_optimal = abs(optimal_center - 0.15) < 0.03  # Within 0.03 tolerance
    print(f"\n=== VERDICT ===")
    if is_optimal and in_ci:
        print("VALIDATED: 0.15 is near-optimal for CPPN edge density")
    elif in_ci:
        print("VALIDATED: 0.15 is within statistical uncertainty of optimal")
    else:
        print(f"REFUTED: Optimal center is {optimal_center:.3f}, not 0.15")

    return {
        'optimal_center': float(optimal_center),
        'current_default': 0.15,
        'cppn_median_edge': float(np.median(edge_densities)),
        'optimal_in_ci': bool(in_ci),
        'effect_size': float(effect),
        'ci_95': [float(opt_ci[0]), float(opt_ci[1])]
    }

if __name__ == "__main__":
    results = main()
    print(f"\n{results}")
