"""
RES-069: Test if same CPPN produces higher order at higher resolutions.

Hypothesis: resolution affects multiplicative vs spectral order metrics differently
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_spectral_order(image):
    """Compute order via spectral decay (fraction of energy in low frequencies)."""
    # FFT
    fft = np.fft.fft2(image)
    power = np.abs(fft)**2

    # Radial average
    y, x = np.ogrid[-1:1:image.shape[0]*1j, -1:1:image.shape[1]*1j]
    r = np.sqrt(x**2 + y**2)

    # Energy in low frequencies (r < 0.5)
    low_freq = np.sum(power[r < 0.5])
    total = np.sum(power)

    return low_freq / (total + 1e-10)


def main():
    np.random.seed(42)

    n_samples = 50
    resolutions = [32, 64, 128, 256]

    results = {res: {'multiplicative': [], 'spectral': []} for res in resolutions}

    print("Testing resolution effects...")
    for sample in range(n_samples):
        cppn, _ = nested_sampling(max_iterations=100, n_live=20)

        for resolution in resolutions:
            coords = np.linspace(-1, 1, resolution)
            coords_x, coords_y = np.meshgrid(coords, coords)

            img = cppn.activate(coords_x, coords_y)

            # Multiplicative order
            mult_order = compute_order(img)
            results[resolution]['multiplicative'].append(mult_order)

            # Spectral order
            spec_order = compute_spectral_order(img)
            results[resolution]['spectral'].append(spec_order)

    # Convert to arrays
    for res in resolutions:
        results[res]['multiplicative'] = np.array(results[res]['multiplicative'])
        results[res]['spectral'] = np.array(results[res]['spectral'])

    # Correlation with resolution
    res_array = np.array(resolutions)
    mult_array = np.array([np.mean(results[r]['multiplicative']) for r in resolutions])
    spec_array = np.array([np.mean(results[r]['spectral']) for r in resolutions])

    corr_mult, p_mult = stats.pearsonr(np.log(res_array), mult_array)
    corr_spec, p_spec = stats.pearsonr(np.log(res_array), spec_array)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for res in resolutions:
        print(f"Resolution {res:3d}: mult={np.mean(results[res]['multiplicative']):.4f}, spec={np.mean(results[res]['spectral']):.4f}")

    print(f"\nCorrelation with log(resolution):")
    print(f"  Multiplicative order: r={corr_mult:.3f}, p={p_mult:.2e}")
    print(f"  Spectral order: r={corr_spec:.3f}, p={p_spec:.2e}")

    # Effect size
    effect_size = max(abs(corr_mult), abs(corr_spec))

    # Check if opposite directions (which is the validated finding)
    opposite_trend = np.sign(corr_mult) != np.sign(corr_spec)

    validated = opposite_trend and max(abs(p_mult), abs(p_spec)) < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_mult


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: r={effect_size:.3f}, p={p_value:.2e}")
