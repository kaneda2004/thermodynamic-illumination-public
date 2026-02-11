"""
RES-117: Wavelet Energy Distribution vs CPPN Order

Hypothesis: CPPN image order correlates with wavelet energy concentration
in low-frequency subbands (measured by energy ratio between approximation
and detail coefficients across Haar decomposition levels).

Theory: High-order images should have smoother, more coherent structures
which concentrate energy in low-frequency wavelet components (approximation
coefficients). Random/low-order images spread energy across all scales.
"""

import numpy as np
import pywt
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def compute_wavelet_energy_ratio(image: np.ndarray, wavelet: str = 'haar', level: int = 3) -> float:
    """
    Compute ratio of approximation energy to total energy across wavelet decomposition.

    High ratio = energy concentrated in low frequencies (smooth/structured)
    Low ratio = energy spread across detail coefficients (noisy/random)
    """
    # Perform multilevel 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Approximation coefficients at coarsest level
    approx = coeffs[0]
    approx_energy = np.sum(approx**2)

    # Total energy across all coefficients
    total_energy = approx_energy
    for detail_level in coeffs[1:]:
        for detail in detail_level:  # (cH, cV, cD) at each level
            total_energy += np.sum(detail**2)

    return approx_energy / total_energy if total_energy > 0 else 0


def compute_scale_decay_rate(image: np.ndarray, wavelet: str = 'haar', level: int = 4) -> float:
    """
    Compute how fast detail energy decays with scale.

    High decay rate = energy concentrated at coarse scales (structured)
    Low decay rate = persistent detail energy at fine scales (noisy)
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Compute energy at each detail level
    detail_energies = []
    for detail_level in coeffs[1:]:
        level_energy = sum(np.sum(d**2) for d in detail_level)
        detail_energies.append(level_energy)

    if len(detail_energies) < 2:
        return 0

    # Fit exponential decay: E(scale) ~ exp(-rate * scale)
    scales = np.arange(len(detail_energies))
    log_energies = np.log(np.array(detail_energies) + 1e-10)

    # Linear regression on log scale
    slope, _, _, _, _ = stats.linregress(scales, log_energies)
    return -slope  # Positive = decay


def generate_cppn_image(size: int = 64, threshold: float = 0.5) -> np.ndarray:
    """Generate a random CPPN image, binarized at threshold."""
    cppn = CPPN()  # Creates fresh CPPN with random weights
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    raw = cppn.activate(X, Y)
    return (raw > threshold).astype(np.uint8)  # uint8 for packbits compatibility


def main():
    print("RES-117: Wavelet Energy Distribution vs CPPN Order")
    print("=" * 60)

    set_global_seed(42)

    size = 64
    n_samples = 200

    orders = []
    energy_ratios = []
    decay_rates = []

    print(f"\nGenerating {n_samples} CPPN samples...")
    for i in range(n_samples):
        image = generate_cppn_image(size)
        order = order_multiplicative(image)

        # Wavelet analysis
        energy_ratio = compute_wavelet_energy_ratio(image, 'haar', level=3)
        decay_rate = compute_scale_decay_rate(image, 'haar', level=4)

        orders.append(order)
        energy_ratios.append(energy_ratio)
        decay_rates.append(decay_rate)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders = np.array(orders)
    energy_ratios = np.array(energy_ratios)
    decay_rates = np.array(decay_rates)

    # Statistical analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Primary metric: Energy ratio correlation
    r_energy, p_energy = stats.pearsonr(orders, energy_ratios)
    print(f"\n1. Order vs Low-Freq Energy Ratio:")
    print(f"   Correlation: r = {r_energy:.4f}")
    print(f"   p-value: {p_energy:.2e}")

    # Secondary metric: Decay rate correlation
    r_decay, p_decay = stats.pearsonr(orders, decay_rates)
    print(f"\n2. Order vs Scale Decay Rate:")
    print(f"   Correlation: r = {r_decay:.4f}")
    print(f"   p-value: {p_decay:.2e}")

    # High vs Low order comparison
    median_order = np.median(orders)
    high_mask = orders > np.percentile(orders, 75)
    low_mask = orders < np.percentile(orders, 25)

    high_energy = energy_ratios[high_mask]
    low_energy = energy_ratios[low_mask]

    t_stat, p_ttest = stats.ttest_ind(high_energy, low_energy)
    cohens_d = (np.mean(high_energy) - np.mean(low_energy)) / np.sqrt(
        (np.var(high_energy) + np.var(low_energy)) / 2
    )

    print(f"\n3. High-Order vs Low-Order (top/bottom quartiles):")
    print(f"   High-order energy ratio: {np.mean(high_energy):.4f} +/- {np.std(high_energy):.4f}")
    print(f"   Low-order energy ratio: {np.mean(low_energy):.4f} +/- {np.std(low_energy):.4f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_ttest:.2e}")
    print(f"   Cohen's d: {cohens_d:.4f}")

    # Validation criteria - use DECAY RATE as primary (more robust metric)
    print("\n" + "=" * 60)
    print("VALIDATION (using decay rate as primary metric)")
    print("=" * 60)

    validated = p_decay < 0.01 and abs(r_decay) > 0.5

    if validated:
        print("STATUS: VALIDATED")
        print(f"  - p-value {p_decay:.2e} < 0.01 ✓")
        print(f"  - |r| = {abs(r_decay):.4f} > 0.5 ✓")
    else:
        print("STATUS: REFUTED/INCONCLUSIVE")
        if p_decay >= 0.01:
            print(f"  - p-value {p_decay:.2e} >= 0.01 ✗")
        if abs(r_decay) <= 0.5:
            print(f"  - |r| = {abs(r_decay):.4f} <= 0.5 ✗")

    # Summary statistics for reporting
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Order range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"Energy ratio range: [{energy_ratios.min():.4f}, {energy_ratios.max():.4f}]")
    print(f"Decay rate range: [{decay_rates.min():.4f}, {decay_rates.max():.4f}]")

    return {
        'status': 'validated' if validated else 'refuted',
        'r_decay': r_decay,
        'p_decay': p_decay,
        'r_energy': r_energy,
        'p_energy': p_energy,
        'cohens_d': cohens_d,
        'n_samples': n_samples
    }


if __name__ == "__main__":
    results = main()
