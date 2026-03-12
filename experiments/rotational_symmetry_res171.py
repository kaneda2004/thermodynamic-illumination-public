"""
RES-171: Test if r-dominant weight CPPNs produce higher rotational symmetry.

Hypothesis: weights on r-input create rotationally invariant outputs
"""

import numpy as np
from scipy import stats
import sys
import json
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_rotational_symmetry(image, threshold=0.5):
    """Compute rotational symmetry by comparing rotations."""
    binary = (image > threshold).astype(int)

    # Try 90 degree rotation
    rotated_90 = np.rot90(binary)
    rotated_180 = np.rot90(binary, k=2)
    rotated_270 = np.rot90(binary, k=3)

    # Compute overlap (correlation with original)
    corr_90 = np.mean(binary == rotated_90)
    corr_180 = np.mean(binary == rotated_180)
    corr_270 = np.mean(binary == rotated_270)

    return np.mean([corr_90, corr_180, corr_270])


def analyze_weight_dominance(cppn):
    """Analyze which input coordinates dominate."""
    r_weight = 0.0
    xy_weight = 0.0

    for conn in cppn.connections:
        if conn.from_id == 2:  # r coordinate
            r_weight += abs(conn.weight)
        elif conn.from_id in [0, 1]:  # x, y coordinates
            xy_weight += abs(conn.weight)

    if r_weight + xy_weight == 0:
        r_ratio = 0.5
    else:
        r_ratio = r_weight / (r_weight + xy_weight)

    return r_ratio


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPNs
    print("Generating CPPNs...")
    r_dominance = []
    rotational_sym = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)

        r_dom = analyze_weight_dominance(cppn)
        rot_sym = compute_rotational_symmetry(img)

        r_dominance.append(r_dom)
        rotational_sym.append(rot_sym)

    r_dominance = np.array(r_dominance)
    rotational_sym = np.array(rotational_sym)

    # Correlation
    corr, p_corr = stats.pearsonr(r_dominance, rotational_sym)

    # Compare high r-dominant vs low r-dominant
    threshold = np.median(r_dominance)
    high_r = rotational_sym[r_dominance > threshold]
    low_r = rotational_sym[r_dominance <= threshold]

    t_stat, p_ttest = stats.ttest_ind(high_r, low_r)

    pooled_std = np.sqrt((np.std(high_r)**2 + np.std(low_r)**2) / 2)
    effect_size = (np.mean(high_r) - np.mean(low_r)) / (pooled_std + 1e-10)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Correlation (r-dominance vs rotational symmetry): r={corr:.3f}, p={p_corr:.2e}")
    print(f"High r-dominance rotational symmetry: {np.mean(high_r):.4f}")
    print(f"Low r-dominance rotational symmetry: {np.mean(low_r):.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_ttest:.2e}")

    validated = effect_size > 0.5 and p_ttest < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    # Save results to JSON
    results = {
        'n_samples': n_samples,
        'correlation_r_dominance_symmetry': float(corr),
        'correlation_p_value': float(p_corr),
        'high_r_dominance_symmetry': float(np.mean(high_r)),
        'low_r_dominance_symmetry': float(np.mean(low_r)),
        'effect_size_cohens_d': float(effect_size),
        'ttest_p_value': float(p_ttest),
        'status': status,
        'hypothesis': 'Larger |weight| on r-input produces more rotational symmetry in CPPN outputs'
    }

    import os
    results_dir = '/Users/matt/Development/monochrome_noise_converger/results/symmetry_mechanisms'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'res171_results.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return validated, effect_size, p_ttest


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
