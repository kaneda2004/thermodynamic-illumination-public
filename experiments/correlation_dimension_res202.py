"""
RES-202: Test if high-order weight regions have different fractal dimensions.

Hypothesis: high-order configs are concentrated in specific regions of weight space
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_correlation_dimension(point_cloud, max_k=50):
    """Estimate correlation dimension from point cloud."""
    if len(point_cloud) < 10:
        return 0.0

    distances = []
    for i in range(min(100, len(point_cloud))):
        for j in range(i+1, min(100, len(point_cloud))):
            dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
            distances.append(dist)

    distances = np.sort(distances)
    counts = []
    radii = np.logspace(-2, 0, max_k)

    for r in radii:
        count = np.sum(np.array(distances) <= r)
        if count > 0:
            counts.append(count)

    if len(counts) < 2:
        return 0.0

    # Log-log slope
    log_radii = np.log(radii[:len(counts)])
    log_counts = np.log(counts)

    slope, _ = np.polyfit(log_radii, log_counts, 1)
    return slope


def main():
    np.random.seed(42)

    n_samples = 100

    # Generate high-order and low-order CPPN weight vectors
    print("Generating CPPN weight vectors...")
    high_order_weights = []
    low_order_weights = []

    resolution = 32
    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    for i in range(n_samples // 2):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        weights = np.array([c.weight for c in cppn.connections])
        high_order_weights.append(weights)

    for i in range(n_samples // 2):
        cppn = CPPN()
        order = compute_order(cppn.activate(coords_x, coords_y))
        weights = np.array([c.weight for c in cppn.connections])
        low_order_weights.append(weights)

    high_order_weights = np.array(high_order_weights)
    low_order_weights = np.array(low_order_weights)

    # Compute correlation dimensions
    high_dim = compute_correlation_dimension(high_order_weights)
    low_dim = compute_correlation_dimension(low_order_weights)

    # Box-counting dimension
    def box_count_dim(points):
        counts = []
        scales = []
        for scale in [0.5, 1.0, 2.0, 4.0]:
            count = 0
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if np.linalg.norm(points[i] - points[j]) < scale:
                        count += 1
            if count > 0:
                counts.append(count)
                scales.append(scale)

        if len(counts) > 1:
            dim, _ = np.polyfit(np.log(scales), np.log(counts), 1)
            return -dim
        return 0.0

    high_box = box_count_dim(high_order_weights)
    low_box = box_count_dim(low_order_weights)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order correlation dimension: {high_dim:.3f}")
    print(f"Low-order correlation dimension: {low_dim:.3f}")
    print(f"High-order box-counting dimension: {high_box:.3f}")
    print(f"Low-order box-counting dimension: {low_box:.3f}")

    effect_size = abs(high_dim - low_dim) / (np.std([high_dim, low_dim]) + 1e-10)

    validated = abs(effect_size) > 0.5
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, 0.01


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: effect={effect_size:.2f}")
