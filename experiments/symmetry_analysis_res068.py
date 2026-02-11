"""
RES-068: Test if high-order CPPN images have bilateral symmetry over rotational.

Hypothesis: smooth coordinate functions naturally produce vertical/horizontal symmetry
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_symmetries(image, threshold=0.5):
    """Compute bilateral and rotational symmetries."""
    binary = (image > threshold).astype(int)

    # Horizontal reflection
    h_flip = np.fliplr(binary)
    h_sym = np.mean(binary == h_flip)

    # Vertical reflection
    v_flip = np.flipud(binary)
    v_sym = np.mean(binary == v_flip)

    # 90 degree rotation
    r90 = np.rot90(binary)
    r_sym = np.mean(binary == r90)

    return {
        'horizontal': h_sym,
        'vertical': v_sym,
        'rotational': r_sym,
        'bilateral': max(h_sym, v_sym)
    }


def main():
    np.random.seed(42)

    n_samples = 150
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    print("Generating CPPN images...")
    horizontal_syms = []
    vertical_syms = []
    rotational_syms = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)

        syms = compute_symmetries(img)
        horizontal_syms.append(syms['horizontal'])
        vertical_syms.append(syms['vertical'])
        rotational_syms.append(syms['rotational'])

    horizontal_syms = np.array(horizontal_syms)
    vertical_syms = np.array(vertical_syms)
    rotational_syms = np.array(rotational_syms)

    # Statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Horizontal symmetry: {np.mean(horizontal_syms):.1%}")
    print(f"Vertical symmetry: {np.mean(vertical_syms):.1%}")
    print(f"Rotational symmetry: {np.mean(rotational_syms):.1%}")

    # Chi-square test
    bilateral_count = np.sum((horizontal_syms > 0.5) | (vertical_syms > 0.5))
    rotational_count = np.sum(rotational_syms > 0.5)

    contingency = np.array([
        [bilateral_count, n_samples - bilateral_count],
        [rotational_count, n_samples - rotational_count]
    ])

    chi2 = ((contingency[0, 0] - contingency[1, 0])**2) / (contingency[0, 0] + 1)

    # Correlation test
    t_stat, p_ttest = stats.ttest_ind(
        np.maximum(horizontal_syms, vertical_syms),
        rotational_syms
    )

    # Effect size
    bilateral_mean = np.maximum(horizontal_syms, vertical_syms)
    pooled_std = np.sqrt((np.std(bilateral_mean)**2 + np.std(rotational_syms)**2) / 2)
    effect_size = (np.mean(bilateral_mean) - np.mean(rotational_syms)) / (pooled_std + 1e-10)

    print(f"\nBilateral vs rotational:")
    print(f"  Effect size (Cohen's d): {effect_size:.2f}")
    print(f"  Chi-square: {chi2:.2f}")
    print(f"  p-value: {p_ttest:.2e}")

    validated = effect_size > 0.5 and p_ttest < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_ttest


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
