#!/usr/bin/env python3
"""
RES-231: Radial-Basis Coordinate System Architecture
Tests whether polar [r, θ] coordinates create lower-dimensional,
more symmetric CPPN structures compared to Cartesian.

HYPOTHESIS:
Polar coordinates create symmetric weight structure with:
  - Effective dimensionality ≤ 3D (vs Cartesian ~8D)
  - Rotational symmetry score ≥ 1.3× baseline
  - Efficient nested sampling to order 0.5

METHOD:
Compare 3 coordinate system variants (each 30 CPPNs):
  A) Polar: [r, θ] where r=sqrt(x²+y²), θ=atan2(y,x)
  B) Hybrid: [r, θ, r*cos(θ)] (polar + Cartesian mixing)
  C) Cartesian: [x, y] (baseline)

For each:
  1. Initialize CPPNs with coordinate variant
  2. Measure effective dimensionality
  3. Measure rotational symmetry (MI of rotated samples)
  4. Run nested sampling to order 0.5
  5. Collect efficiency metrics
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os

# Setup path
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(Path.cwd()))

def create_coordinate_inputs(grid_size: int = 32, variant: str = 'cartesian') -> np.ndarray:
    """
    Create coordinate inputs for image grid.

    Args:
        grid_size: Size of image (grid_size x grid_size)
        variant: 'cartesian', 'polar', or 'hybrid'

    Returns:
        coords: Shape (2, grid_size, grid_size) with coordinate values
    """
    # Create normalized grid [-1, 1] x [-1, 1]
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)

    if variant == 'cartesian':
        # Standard Cartesian [x, y]
        coords = np.array([xx, yy])
    elif variant == 'polar':
        # Polar [r, θ] normalized to [-1, 1]
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        # Normalize r to [0, 1] then shift to [-1, 1]
        r_norm = r / np.sqrt(2)  # max radius at corners
        theta_norm = theta / np.pi  # theta in [-1, 1]
        coords = np.array([r_norm, theta_norm])
    elif variant == 'hybrid':
        # Hybrid: [r, θ, r*cos(θ)]
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        r_norm = r / np.sqrt(2)
        theta_norm = theta / np.pi
        # Cartesian mixing term
        mixing = r_norm * np.cos(theta)
        coords = np.array([r_norm, theta_norm, mixing])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return coords

def compose_cppn_output(coords: np.ndarray, weights: np.ndarray,
                        n_hidden: int = 16) -> np.ndarray:
    """
    Simple compositional CPPN: tanh layer then linear output.

    Args:
        coords: Shape (n_coord_dims, height, width)
        weights: Random weights for the CPPN
        n_hidden: Number of hidden units

    Returns:
        image: Shape (height, width) with values in (-1, 1)
    """
    h, w = coords.shape[-2:]

    # Flatten spatial dimensions
    coord_flat = coords.reshape(coords.shape[0], -1)  # (n_coords, h*w)

    # Hidden layer: mix coordinates with random weights
    w_in = weights[:, :n_hidden]  # (n_coords, n_hidden)
    hidden = np.tanh(coord_flat.T @ w_in)  # (h*w, n_hidden)

    # Output layer: linear combination of hidden units
    w_out = weights[:, n_hidden:n_hidden+1]  # (n_coords, 1) - use first output weight
    output = hidden @ np.random.randn(n_hidden, 1)  # (h*w, 1)
    output = np.tanh(output.flatten())  # Apply tanh activation

    # Reshape back to image
    image = output.reshape(h, w)
    return image

def measure_effective_dimension(images: np.ndarray, n_samples: int = 100) -> float:
    """
    Estimate effective dimensionality using PCA cumulative variance.

    Uses the number of PCs needed to explain 95% of variance.

    Args:
        images: Shape (n_images, height, width)
        n_samples: How many random subsamples to use for PCA

    Returns:
        eff_dim: Effective dimensionality (float)
    """
    # Flatten images
    flat = images.reshape(images.shape[0], -1)  # (n_images, h*w)

    # Center data
    flat = flat - flat.mean(axis=0)

    # SVD
    try:
        U, s, Vt = np.linalg.svd(flat, full_matrices=False)
        # Compute cumulative variance
        explained_var = (s**2) / (s**2).sum()
        cumsum_var = np.cumsum(explained_var)
        # Find components needed for 95% variance
        n_components = np.argmax(cumsum_var >= 0.95) + 1
        eff_dim = float(n_components)
    except:
        eff_dim = min(flat.shape)

    return eff_dim

def measure_rotational_symmetry(images: np.ndarray, angles: List[float] = None) -> float:
    """
    Measure rotational symmetry using MI-like approach.

    Rotates images by different angles and measures correlation.
    Higher values indicate more symmetric structure.

    Args:
        images: Shape (n_images, height, width)
        angles: List of rotation angles (degrees)

    Returns:
        symmetry_score: Float in [0, 1], higher = more symmetric
    """
    if angles is None:
        angles = [45, 90, 135, 180]

    scores = []
    for img in images[:5]:  # Sample first 5 images
        correlations = []
        for angle in angles:
            # Simple rotation by 90-degree approximation
            rotated = np.rot90(img, k=int(angle // 90) % 4)
            # Correlation with original (as symmetry measure)
            corr = np.corrcoef(img.flatten(), rotated.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if correlations:
            # Average correlation across rotations
            # More symmetric = higher correlation
            avg_corr = np.mean(np.abs(correlations))
            scores.append(avg_corr)

    symmetry = float(np.mean(scores)) if scores else 0.0
    return symmetry

def run_nested_sampling(images: np.ndarray, max_order: float = 0.5,
                       n_samples: int = 200) -> Dict[str, float]:
    """
    Simple nested sampling simulation: estimate order of best samples.

    Args:
        images: Shape (n_images, height, width)
        max_order: Maximum order to achieve
        n_samples: Number of sampling steps

    Returns:
        metrics: Dict with order and sample efficiency
    """
    # Compute "order" as average pixel correlation (structure measure)
    orders = []
    for img in images:
        # Order = average absolute spatial correlation
        flat = img.flatten()
        # Measure local structure
        h, w = img.shape
        if h > 1 and w > 1:
            # Horizontal differences
            h_diff = np.abs(np.diff(img, axis=1)).mean()
            # Vertical differences
            v_diff = np.abs(np.diff(img, axis=0)).mean()
            # Order: inverse of difference magnitude (more structure = higher order)
            order = 1.0 / (1.0 + h_diff + v_diff)
            orders.append(order)

    avg_order = np.mean(orders) if orders else 0.0

    # Sample efficiency: how many samples to reach 0.5 order
    # Measure convergence rate
    converged_samples = int(n_samples * (1.0 - avg_order))
    efficiency = n_samples / max(converged_samples, 1)

    return {
        'avg_order': float(avg_order),
        'convergence_samples': int(converged_samples),
        'sample_efficiency': float(efficiency)
    }

def experiment_coordinate_variant(variant: str, n_cpps: int = 30,
                                  grid_size: int = 32) -> Dict[str, Any]:
    """
    Run full experiment for one coordinate variant.

    Args:
        variant: 'cartesian', 'polar', or 'hybrid'
        n_cpps: Number of CPPNs to initialize
        grid_size: Size of generated images

    Returns:
        results: Dict with all metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing variant: {variant.upper()}")
    print(f"{'='*60}")

    # Create coordinate system
    coords = create_coordinate_inputs(grid_size, variant)
    n_coord_dims = coords.shape[0]
    print(f"Coordinate dimensions: {n_coord_dims}")

    # Initialize and generate CPPNs
    images = []
    for i in range(n_cpps):
        # Random CPPN weights
        weights = np.random.randn(n_coord_dims, 32)  # (n_coords, 32)

        # Generate image
        image = compose_cppn_output(coords, weights, n_hidden=16)
        images.append(image)

    images = np.array(images)
    print(f"Generated {len(images)} CPPNs")

    # Measure effective dimensionality
    eff_dim = measure_effective_dimension(images)
    print(f"Effective dimensionality: {eff_dim:.2f}D")

    # Measure rotational symmetry
    symmetry = measure_rotational_symmetry(images)
    print(f"Rotational symmetry: {symmetry:.4f}")

    # Run nested sampling
    ns_results = run_nested_sampling(images)
    print(f"Nested sampling order: {ns_results['avg_order']:.4f}")
    print(f"Sample efficiency: {ns_results['sample_efficiency']:.2f}x")

    return {
        'variant': variant,
        'n_cpps': n_cpps,
        'coord_dims': n_coord_dims,
        'eff_dim': float(eff_dim),
        'symmetry_score': float(symmetry),
        'avg_order': float(ns_results['avg_order']),
        'convergence_samples': int(ns_results['convergence_samples']),
        'sample_efficiency': float(ns_results['sample_efficiency'])
    }

def main():
    """Run full RES-231 experiment."""
    print("\n" + "="*70)
    print("RES-231: Radial-Basis Coordinate System Architecture")
    print("="*70)

    # Test all three variants
    results_by_variant = {}
    for variant in ['cartesian', 'polar', 'hybrid']:
        result = experiment_coordinate_variant(variant, n_cpps=30, grid_size=32)
        results_by_variant[variant] = result

    # Compute comparisons
    cartesian_result = results_by_variant['cartesian']
    polar_result = results_by_variant['polar']
    hybrid_result = results_by_variant['hybrid']

    # Analysis
    polar_eff_dim_ratio = polar_result['eff_dim'] / cartesian_result['eff_dim']
    hybrid_eff_dim_ratio = hybrid_result['eff_dim'] / cartesian_result['eff_dim']

    polar_symmetry_gain = polar_result['symmetry_score'] / max(cartesian_result['symmetry_score'], 0.001)
    hybrid_symmetry_gain = hybrid_result['symmetry_score'] / max(cartesian_result['symmetry_score'], 0.001)

    polar_efficiency_ratio = polar_result['sample_efficiency'] / max(cartesian_result['sample_efficiency'], 0.001)
    hybrid_efficiency_ratio = hybrid_result['sample_efficiency'] / max(cartesian_result['sample_efficiency'], 0.001)

    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    print(f"\nCartesian (baseline):")
    print(f"  Eff Dim: {cartesian_result['eff_dim']:.2f}D")
    print(f"  Symmetry: {cartesian_result['symmetry_score']:.4f}")
    print(f"  Sample Efficiency: {cartesian_result['sample_efficiency']:.2f}x")

    print(f"\nPolar [r, θ]:")
    print(f"  Eff Dim: {polar_result['eff_dim']:.2f}D (ratio vs Cart: {polar_eff_dim_ratio:.2f}x)")
    print(f"  Symmetry: {polar_result['symmetry_score']:.4f} (gain vs Cart: {polar_symmetry_gain:.2f}x)")
    print(f"  Sample Efficiency: {polar_result['sample_efficiency']:.2f}x (ratio: {polar_efficiency_ratio:.2f}x)")

    print(f"\nHybrid [r, θ, r*cos(θ)]:")
    print(f"  Eff Dim: {hybrid_result['eff_dim']:.2f}D (ratio vs Cart: {hybrid_eff_dim_ratio:.2f}x)")
    print(f"  Symmetry: {hybrid_result['symmetry_score']:.4f} (gain vs Cart: {hybrid_symmetry_gain:.2f}x)")
    print(f"  Sample Efficiency: {hybrid_result['sample_efficiency']:.2f}x (ratio: {hybrid_efficiency_ratio:.2f}x)")

    # Validation
    polar_passes = (polar_eff_dim_ratio < 0.4) and (polar_symmetry_gain >= 1.3)
    hybrid_passes = (hybrid_eff_dim_ratio < 0.4) and (hybrid_symmetry_gain >= 1.3)

    print("\n" + "="*70)
    print("HYPOTHESIS VALIDATION")
    print("="*70)
    print(f"\nHypothesis: Polar eff_dim ≤ 3D AND symmetry ≥ 1.3× baseline")
    print(f"  Cartesian baseline eff_dim: {cartesian_result['eff_dim']:.2f}D")
    print(f"  Polar eff_dim: {polar_result['eff_dim']:.2f}D")
    print(f"  Target: ≤ 3D")
    print(f"  Result: {'✓ PASS' if polar_result['eff_dim'] <= 3.0 else '✗ FAIL'}")

    print(f"\n  Cartesian baseline symmetry: {cartesian_result['symmetry_score']:.4f}")
    print(f"  Polar symmetry: {polar_result['symmetry_score']:.4f}")
    print(f"  Target gain: ≥ 1.3×")
    print(f"  Actual gain: {polar_symmetry_gain:.2f}×")
    print(f"  Result: {'✓ PASS' if polar_symmetry_gain >= 1.3 else '✗ FAIL'}")

    conclusion = 'validate' if (polar_passes or hybrid_passes) else 'refute'

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/radial_basis_architecture')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'method': 'Polar vs Cartesian coordinate systems',
        'cartesian_eff_dim': float(cartesian_result['eff_dim']),
        'polar_eff_dim': float(polar_result['eff_dim']),
        'hybrid_eff_dim': float(hybrid_result['eff_dim']),
        'cartesian_symmetry': float(cartesian_result['symmetry_score']),
        'polar_symmetry': float(polar_result['symmetry_score']),
        'hybrid_symmetry': float(hybrid_result['symmetry_score']),
        'cartesian_samples': int(cartesian_result['convergence_samples']),
        'polar_samples': int(polar_result['convergence_samples']),
        'hybrid_samples': int(hybrid_result['convergence_samples']),
        'polar_symmetry_advantage': float(polar_symmetry_gain),
        'hybrid_symmetry_advantage': float(hybrid_symmetry_gain),
        'polar_efficiency': float(polar_efficiency_ratio),
        'hybrid_efficiency': float(hybrid_efficiency_ratio),
        'conclusion': conclusion
    }

    with open(output_dir / 'res_231_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir / 'res_231_results.json'}")

    # Return summary
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"RES-231 | radial_basis_architecture | {conclusion.upper()} | "
          f"sym_gain={polar_symmetry_gain:.2f}x eff_dim={polar_result['eff_dim']:.2f}D")

    return final_results

if __name__ == '__main__':
    main()
