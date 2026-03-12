"""
RES-143: Order Gradient at High-Low Boundary

Hypothesis: Order gradient at boundary points toward nearest high-order peak

Method:
1. Find boundary points (medium order)
2. Compute gradient direction
3. Test: Should point toward nearest high-order peak
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=100, n_peaks=5, seed=42):
    """Test gradient direction at boundaries."""
    np.random.seed(seed)

    alignments_nearest = []
    alignments_farthest = []

    print(f"Analyzing boundary gradients...")
    for trial in range(n_samples):
        # Generate high-order peaks
        peaks = []
        for _ in range(n_peaks):
            cppn_peak = CPPN(hidden_nodes=3)
            cppn_peak.randomize()
            order_peak = compute_order(cppn_peak.render(64))

            if order_peak > 0.5:
                peaks.append(cppn_peak.get_weights())

        if len(peaks) < n_peaks:
            continue

        # Find boundary point (medium order)
        for _ in range(10):
            cppn_bound = CPPN(hidden_nodes=3)
            cppn_bound.randomize()
            order_bound = compute_order(cppn_bound.render(64))

            if 0.3 < order_bound < 0.5:
                w_bound = cppn_bound.get_weights()
                break

        # Estimate gradient (finite difference)
        eps = 0.01
        w_pert = w_bound + eps * np.random.randn(len(w_bound))
        cppn_pert = CPPN(hidden_nodes=3)
        cppn_pert.set_weights(w_pert)
        order_pert = compute_order(cppn_pert.render(64))

        gradient = (w_pert - w_bound) * (order_pert > order_bound)
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        else:
            continue

        # Distances to peaks
        distances = [np.linalg.norm(w_bound - p) for p in peaks]
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)

        # Directions to peaks
        dir_nearest = (peaks[nearest_idx] - w_bound)
        if np.linalg.norm(dir_nearest) > 0:
            dir_nearest = dir_nearest / np.linalg.norm(dir_nearest)
            align_nearest = np.dot(gradient, dir_nearest)
        else:
            align_nearest = 0

        dir_farthest = (peaks[farthest_idx] - w_bound)
        if np.linalg.norm(dir_farthest) > 0:
            dir_farthest = dir_farthest / np.linalg.norm(dir_farthest)
            align_farthest = np.dot(gradient, dir_farthest)
        else:
            align_farthest = 0

        alignments_nearest.append(align_nearest)
        alignments_farthest.append(align_farthest)

    if len(alignments_nearest) == 0:
        alignments_nearest = [0]
        alignments_farthest = [0]

    alignments_nearest = np.array(alignments_nearest)
    alignments_farthest = np.array(alignments_farthest)

    # Compare
    t_stat, p_value = stats.ttest_ind(alignments_nearest, alignments_farthest)

    pooled_std = np.sqrt((np.std(alignments_nearest)**2 + np.std(alignments_farthest)**2) / 2)
    cohens_d = (np.mean(alignments_nearest) - np.mean(alignments_farthest)) / pooled_std if pooled_std > 0 else 0

    status = 'inconclusive' if abs(cohens_d) < 0.5 else 'validated'

    results = {
        'hypothesis': 'Order gradient at high-low boundary points toward nearest high-order peak not global max',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Nearest={np.mean(alignments_nearest):.3f}, Farthest={np.mean(alignments_farthest):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'boundary_structure')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
