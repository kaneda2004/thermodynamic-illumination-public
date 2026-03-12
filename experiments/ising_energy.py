"""
RES-105: Test if CPPN images have lower Ising model energy than random images.

Ising energy = sum of adjacent pixel mismatches (XOR between neighbors)
Lower energy = more spatial coherence (like physical Ising model at low temperature)
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN


def compute_ising_energy(image, threshold=0.5):
    """
    Compute Ising model energy for a binarized image.
    Energy = number of adjacent pixel mismatches (horizontal + vertical)
    Normalized by total number of edges.
    """
    binary = (image > threshold).astype(int)

    # Horizontal mismatches
    h_mismatch = np.sum(binary[:, :-1] != binary[:, 1:])

    # Vertical mismatches
    v_mismatch = np.sum(binary[:-1, :] != binary[1:, :])

    total_edges = (binary.shape[0] * (binary.shape[1] - 1) +
                   (binary.shape[0] - 1) * binary.shape[1])

    # Normalized energy (0 = all aligned, 1 = all mismatched)
    return (h_mismatch + v_mismatch) / total_edges


def generate_cppn_image(resolution):
    """Generate a CPPN image at given resolution."""
    cppn = CPPN()
    coords = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    # Generate CPPN images
    print("Generating CPPN images...")
    cppn_images = []
    for i in range(n_samples):
        img = generate_cppn_image(resolution)
        cppn_images.append(img)

    # Generate random images
    print("Generating random images...")
    random_images = [np.random.random((resolution, resolution)) for _ in range(n_samples)]

    # Compute Ising energies at multiple thresholds
    thresholds = [0.3, 0.5, 0.7]

    results = {}
    for thresh in thresholds:
        cppn_energies = [compute_ising_energy(img, thresh) for img in cppn_images]
        random_energies = [compute_ising_energy(img, thresh) for img in random_images]

        # Statistical test
        t_stat, p_value = stats.ttest_ind(cppn_energies, random_energies)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(cppn_energies)**2 + np.std(random_energies)**2) / 2)
        effect_size = (np.mean(random_energies) - np.mean(cppn_energies)) / pooled_std

        results[thresh] = {
            'cppn_mean': np.mean(cppn_energies),
            'cppn_std': np.std(cppn_energies),
            'random_mean': np.mean(random_energies),
            'random_std': np.std(random_energies),
            't_stat': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }

        print(f"\nThreshold {thresh}:")
        print(f"  CPPN energy: {np.mean(cppn_energies):.4f} +/- {np.std(cppn_energies):.4f}")
        print(f"  Random energy: {np.mean(random_energies):.4f} +/- {np.std(random_energies):.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.2f}")
        print(f"  p-value: {p_value:.2e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Use threshold 0.5 as primary result
    primary = results[0.5]
    validated = primary['p_value'] < 0.01 and primary['effect_size'] > 0.5

    print(f"Primary (threshold=0.5):")
    print(f"  Effect size: {primary['effect_size']:.2f}")
    print(f"  p-value: {primary['p_value']:.2e}")
    print(f"  VALIDATED: {validated}")

    # Check consistency across thresholds
    all_validated = all(r['p_value'] < 0.01 and r['effect_size'] > 0.5 for r in results.values())
    print(f"  Consistent across all thresholds: {all_validated}")

    return validated, primary['effect_size'], primary['p_value']


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: validated={validated}, d={effect_size:.2f}, p={p_value:.2e}")
