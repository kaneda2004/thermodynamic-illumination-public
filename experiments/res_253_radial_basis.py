#!/usr/bin/env python3
"""
RES-253: Radial-Basis Coordinate Systems and Symmetry
Hypothesis: Polar coordinate inputs [distance, angle] create more symmetric
weight space structure (lower eff_dim but higher rotational order) compared
to Cartesian [x, y, r], achieving ≥1.3× symmetry improvement.

Method:
1. Implement coordinate system variants:
   - Cartesian baseline: [x, y, r]
   - Polar: [sqrt(x²+y²), atan2(y,x)]
   - Hybrid: [r, theta, r*cos(theta)]
2. For each:
   - Measure effective dimensionality (MLE via Levina & Bickel)
   - Compute symmetry scores (rotational mutual information, Fourier analysis)
   - Run nested sampling on 30 CPPNs per system to order ≥0.5
3. Test: Does polar achieve higher rotational symmetry?
4. Validate: polar eff_dim ≤ 3D AND symmetry_score ≥ 1.3× baseline
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.ndimage import rotate
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_system.log_manager import ResearchLogManager

warnings.filterwarnings('ignore')

# Hardware config for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IMAGE_SIZE = 64
N_CPNS = 30
K_NEIGHBORS = [5, 10, 15]
THRESHOLD_ORDER = 0.5


class CoordinateTransformer:
    """Compute coordinate system inputs for CPPN."""

    @staticmethod
    def cartesian(x, y):
        """Cartesian: [x, y, r]"""
        r = np.sqrt(x**2 + y**2)
        return np.stack([x, y, r], axis=-1)

    @staticmethod
    def polar(x, y):
        """Polar: [r, theta, r_normalized]"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        r_norm = r / np.sqrt(2)  # Normalize to [0,1]
        return np.stack([r_norm, theta / np.pi, r_norm], axis=-1)

    @staticmethod
    def hybrid(x, y):
        """Hybrid: [r, theta, r*cos(theta)]"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        r_cos = r * np.cos(theta) / np.sqrt(2)
        r_norm = r / np.sqrt(2)
        return np.stack([r_norm, theta / np.pi, r_cos], axis=-1)


class SimpleCPPN:
    """Minimal CPPN using just dense layers and activations."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.w1 = np.random.randn(3, 16) * 0.5
        self.b1 = np.zeros(16)
        self.w2 = np.random.randn(16, 16) * 0.5
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(16, 1) * 0.5
        self.b3 = np.zeros(1)

    def forward(self, coords):
        """coords shape: (..., 3)"""
        h1 = np.tanh(np.dot(coords, self.w1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.w2) + self.b2)
        out = np.tanh(np.dot(h2, self.w3) + self.b3)
        return out.squeeze(-1)


def generate_image(cppn, coord_func, size=IMAGE_SIZE):
    """Generate CPPN image using given coordinate system."""
    # Create coordinate grid
    grid = np.linspace(-1, 1, size)
    x, y = np.meshgrid(grid, grid)

    # Transform to desired coordinate system
    coords = coord_func(x, y)

    # Evaluate CPPN
    image = cppn.forward(coords)

    # Normalize to [0, 1]
    image = (image + 1) / 2
    return image


def estimate_intrinsic_dimension(images, k_values=[5, 10, 15]):
    """Estimate intrinsic dimension using MLE (Levina & Bickel 2004)."""
    # Flatten images
    flat_images = images.reshape(images.shape[0], -1)
    n_samples = flat_images.shape[0]

    # Compute pairwise distances
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances[i] = np.linalg.norm(flat_images - flat_images[i], axis=1)

    # MLE dimension estimation
    dimensions = []
    for k in k_values:
        # Find k-nearest neighbor distances
        sorted_dist = np.sort(distances, axis=1)
        d_k = sorted_dist[:, k]
        d_k1 = sorted_dist[:, k+1]

        # MLE formula
        ratios = d_k / d_k1
        m = np.mean(1.0 / np.log(ratios[ratios > 0]))
        dimensions.append(m)

    return np.mean(dimensions), np.std(dimensions)


def compute_rotational_symmetry(image):
    """Compute rotational symmetry via mutual information with 90-degree rotations."""
    sym_scores = []

    # Check 90, 180, 270 degree rotations
    for angle in [90, 180, 270]:
        rotated = rotate(image, angle, reshape=False, order=1)
        # Mutual information (approximated via correlation)
        correlation = np.corrcoef(image.flatten(), rotated.flatten())[0, 1]
        sym_scores.append(max(0, correlation))  # Clamp to [0, 1]

    return np.mean(sym_scores)


def compute_reflection_symmetry(image):
    """Compute reflection symmetry (horizontal + vertical)."""
    h_flip = np.fliplr(image)
    v_flip = np.flipud(image)

    h_corr = np.corrcoef(image.flatten(), h_flip.flatten())[0, 1]
    v_corr = np.corrcoef(image.flatten(), v_flip.flatten())[0, 1]

    return np.mean([max(0, h_corr), max(0, v_corr)])


def compute_fourier_symmetry(image):
    """Compute symmetry in Fourier domain (spectral analysis)."""
    # FFT magnitude spectrum
    fft = np.abs(np.fft.fft2(image))
    # Shift zero-frequency to center
    fft_shifted = np.fft.fftshift(fft)

    # Check radial symmetry in log-polar coords
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Compute variance in different directions
    radial_bins = np.linspace(0, np.max(r), 20)
    symmetry_scores = []

    for i in range(len(radial_bins) - 1):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
        if np.sum(mask) > 0:
            ring = fft_shifted[mask]
            # Lower variance = more symmetric
            variance = np.var(ring) if len(ring) > 1 else 0
            symmetry_scores.append(variance)

    if symmetry_scores:
        # Invert: lower variance = higher symmetry
        mean_var = np.mean(symmetry_scores)
        return 1.0 / (1.0 + mean_var)  # Normalize to [0, 1]
    return 0.5


def compute_order(images):
    """Compute visual order via bit-cost analysis."""
    # Simplified order metric: inverse of image entropy
    orders = []
    for img in images:
        # Normalize to [0, 1]
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        # Entropy
        hist, _ = np.histogram(img_norm, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        # Order is inverse entropy (max entropy = 8 for uniform)
        order = max(0, 1 - entropy / 8.0)
        orders.append(order)
    return np.mean(orders)


def run_experiment():
    """Execute the coordinate system comparison."""
    results = {
        "hypothesis": "Polar coordinate inputs create more symmetric weight space structure",
        "coordinate_systems": {},
        "symmetry_improvement": None,
        "eff_dim_ratio": None,
        "rotational_order": None,
        "summary": None
    }

    coord_systems = {
        "cartesian": CoordinateTransformer.cartesian,
        "polar": CoordinateTransformer.polar,
        "hybrid": CoordinateTransformer.hybrid
    }

    system_metrics = {}

    for system_name, coord_func in coord_systems.items():
        print(f"\nTesting {system_name.upper()} coordinate system...")

        images_list = []
        rotational_symmetries = []
        reflection_symmetries = []
        fourier_symmetries = []
        orders = []
        eff_dims = []

        # Generate CPPNs and images
        for cppn_idx in range(N_CPNS):
            cppn = SimpleCPPN(seed=RANDOM_SEED + cppn_idx)
            image = generate_image(cppn, coord_func, size=IMAGE_SIZE)
            images_list.append(image)

            # Compute symmetries
            rot_sym = compute_rotational_symmetry(image)
            ref_sym = compute_reflection_symmetry(image)
            fou_sym = compute_fourier_symmetry(image)

            rotational_symmetries.append(rot_sym)
            reflection_symmetries.append(ref_sym)
            fourier_symmetries.append(fou_sym)

        # Convert to array for batch operations
        images_array = np.array(images_list)

        # Compute order
        order = compute_order(images_array)

        # Compute intrinsic dimension
        eff_dim, eff_dim_se = estimate_intrinsic_dimension(images_array, k_values=K_NEIGHBORS)

        # Store metrics
        system_metrics[system_name] = {
            "rotational_symmetry": {
                "mean": float(np.mean(rotational_symmetries)),
                "std": float(np.std(rotational_symmetries))
            },
            "reflection_symmetry": {
                "mean": float(np.mean(reflection_symmetries)),
                "std": float(np.std(reflection_symmetries))
            },
            "fourier_symmetry": {
                "mean": float(np.mean(fourier_symmetries)),
                "std": float(np.std(fourier_symmetries))
            },
            "order": float(order),
            "eff_dim": float(eff_dim),
            "eff_dim_se": float(eff_dim_se)
        }

        print(f"  Rotational symmetry: {np.mean(rotational_symmetries):.4f} ± {np.std(rotational_symmetries):.4f}")
        print(f"  Reflection symmetry: {np.mean(reflection_symmetries):.4f} ± {np.std(reflection_symmetries):.4f}")
        print(f"  Fourier symmetry: {np.mean(fourier_symmetries):.4f} ± {np.std(fourier_symmetries):.4f}")
        print(f"  Visual order: {order:.4f}")
        print(f"  Eff. dimension: {eff_dim:.2f} ± {eff_dim_se:.2f}")

    # Compute improvements
    cart_rot = system_metrics["cartesian"]["rotational_symmetry"]["mean"]
    polar_rot = system_metrics["polar"]["rotational_symmetry"]["mean"]
    hybrid_rot = system_metrics["hybrid"]["rotational_symmetry"]["mean"]

    cart_eff = system_metrics["cartesian"]["eff_dim"]
    polar_eff = system_metrics["polar"]["eff_dim"]

    symmetry_improvement = polar_rot / (cart_rot + 1e-8) if cart_rot > 0 else 0
    eff_dim_ratio = polar_eff / (cart_eff + 1e-8)

    # Determine rotational order change
    if polar_rot > cart_rot + 0.05:
        rotational_order = "improvement"
    elif polar_rot < cart_rot - 0.05:
        rotational_order = "degradation"
    else:
        rotational_order = "no_change"

    # Update results
    results["coordinate_systems"] = system_metrics
    results["symmetry_improvement"] = round(symmetry_improvement, 3)
    results["eff_dim_ratio"] = round(eff_dim_ratio, 3)
    results["rotational_order"] = rotational_order

    # Determine validation status
    hypothesis_validated = (symmetry_improvement >= 1.3) and (polar_eff <= 3.0)

    results["status"] = "validated" if hypothesis_validated else "refuted"
    results["summary"] = f"Polar coordinates {'achieved' if hypothesis_validated else 'did not achieve'} 1.3× symmetry improvement. " \
                        f"Rotational symmetry: {polar_rot:.4f} (polar) vs {cart_rot:.4f} (cartesian). " \
                        f"Eff. dim: {polar_eff:.2f} (polar) vs {cart_eff:.2f} (cartesian). " \
                        f"Hybrid showed {hybrid_rot:.4f} rotational symmetry."

    return results


if __name__ == "__main__":
    print("="*70)
    print("RES-253: Radial-Basis Coordinate Systems and Symmetry")
    print("="*70)

    # Run experiment
    results = run_experiment()

    # Create output directory
    output_dir = Path("/Users/matt/Development/monochrome_noise_converger/results/radial_basis_architecture")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = output_dir / "res_253_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Status: {results['status'].upper()}")
    print(f"Symmetry improvement (polar/cartesian): {results['symmetry_improvement']:.3f}×")
    print(f"Eff. dim ratio (polar/cartesian): {results['eff_dim_ratio']:.3f}")
    print(f"Rotational order: {results['rotational_order']}")
    print(f"\n{results['summary']}")
    print(f"\nResults saved to: {output_file}")

    # Update research log
    try:
        manager = ResearchLogManager()
        manager.add_entry(
            res_id="RES-253",
            domain="radial_basis_architecture",
            hypothesis="Polar coordinate inputs [distance, angle] create more symmetric weight space structure (lower eff_dim but higher rotational order) compared to Cartesian [x, y, r], achieving ≥1.3× symmetry improvement.",
            status=results['status'],
            metrics={
                "symmetry_improvement": results['symmetry_improvement'],
                "eff_dim_ratio": results['eff_dim_ratio'],
                "rotational_order": results['rotational_order'],
                "cartesian_rot_sym": round(results['coordinate_systems']['cartesian']['rotational_symmetry']['mean'], 4),
                "polar_rot_sym": round(results['coordinate_systems']['polar']['rotational_symmetry']['mean'], 4),
                "hybrid_rot_sym": round(results['coordinate_systems']['hybrid']['rotational_symmetry']['mean'], 4),
                "cartesian_eff_dim": round(results['coordinate_systems']['cartesian']['eff_dim'], 2),
                "polar_eff_dim": round(results['coordinate_systems']['polar']['eff_dim'], 2)
            },
            result_summary=results['summary'],
            prior_hypotheses=["RES-018", "RES-068"]
        )
        print("\nResearch log updated successfully.")
    except Exception as e:
        print(f"\nWarning: Could not update research log: {e}")
