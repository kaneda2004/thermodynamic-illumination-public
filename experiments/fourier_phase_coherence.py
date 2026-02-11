#!/usr/bin/env python
"""
RES-208: Phase coherence of magnitude spectrum predicts CPPN order

Hypothesis: Phase coherence (low phase std across frequency components)
predicts high CPPN output order because structured patterns emerge from
aligned frequency phase relationships.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft2
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_phase_coherence(image):
    """Compute phase coherence metrics from 2D FFT."""
    # Compute 2D FFT
    fft_out = fft2(image.astype(float))
    magnitude = np.abs(fft_out)
    phase = np.angle(fft_out)

    # Skip DC component
    magnitude_flat = magnitude[1:, 1:].flatten()
    phase_flat = phase[1:, 1:].flatten()

    # Phase std (lower = more coherent)
    phase_std = np.std(phase_flat)

    # Phase entropy (lower = more coherent)
    phase_bins = np.histogram(phase_flat, bins=8)[0]
    phase_prob = phase_bins / phase_bins.sum()
    phase_entropy = -np.sum(phase_prob[phase_prob > 0] * np.log2(phase_prob[phase_prob > 0] + 1e-10))

    # Mean absolute phase (avg phase alignment)
    mean_abs_phase = np.mean(np.abs(phase_flat))

    return {
        'phase_std': phase_std,
        'phase_entropy': phase_entropy,
        'mean_abs_phase': mean_abs_phase
    }


def main():
    """Test phase coherence vs CPPN order."""
    n_samples = 200
    set_global_seed(42)

    # Generate CPPNs and compute metrics
    samples_data = []

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization
        image = cppn.render(size=64)
        order = order_multiplicative(image)

        coherence = compute_phase_coherence(image)

        samples_data.append({
            'order': order,
            'phase_std': coherence['phase_std'],
            'phase_entropy': coherence['phase_entropy'],
            'mean_abs_phase': coherence['mean_abs_phase']
        })

    # Compute correlations
    orders = np.array([s['order'] for s in samples_data])
    phase_stds = np.array([s['phase_std'] for s in samples_data])
    phase_entropies = np.array([s['phase_entropy'] for s in samples_data])
    mean_abs_phases = np.array([s['mean_abs_phase'] for s in samples_data])

    # Pearson correlations
    r_std, p_std = stats.pearsonr(orders, phase_stds)
    r_entropy, p_entropy = stats.pearsonr(orders, phase_entropies)
    r_abs, p_abs = stats.pearsonr(orders, mean_abs_phases)

    # Spearman correlations
    rho_std, p_sp_std = stats.spearmanr(orders, phase_stds)
    rho_entropy, p_sp_entropy = stats.spearmanr(orders, phase_entropies)
    rho_abs, p_sp_abs = stats.spearmanr(orders, mean_abs_phases)

    # Effect sizes (Cohen's d)
    high_order_mask = orders > np.median(orders)
    d_std = (phase_stds[~high_order_mask].mean() - phase_stds[high_order_mask].mean()) / np.std(phase_stds)
    d_entropy = (phase_entropies[~high_order_mask].mean() - phase_entropies[high_order_mask].mean()) / np.std(phase_entropies)
    d_abs = (mean_abs_phases[~high_order_mask].mean() - mean_abs_phases[high_order_mask].mean()) / np.std(mean_abs_phases)

    print("=== Phase Coherence Analysis ===")
    print(f"\nPhase Std (lower = more coherent):")
    print(f"  Pearson r={r_std:.3f} (p={p_std:.4f}), Spearman rho={rho_std:.3f} (p={p_sp_std:.4f})")
    print(f"  Effect d={d_std:.2f}, low-order mean={phase_stds[~high_order_mask].mean():.3f}, high={phase_stds[high_order_mask].mean():.3f}")

    print(f"\nPhase Entropy (lower = more coherent):")
    print(f"  Pearson r={r_entropy:.3f} (p={p_entropy:.4f}), Spearman rho={rho_entropy:.3f} (p={p_sp_entropy:.4f})")
    print(f"  Effect d={d_entropy:.2f}, low-order mean={phase_entropies[~high_order_mask].mean():.3f}, high={phase_entropies[high_order_mask].mean():.3f}")

    print(f"\nMean Absolute Phase:")
    print(f"  Pearson r={r_abs:.3f} (p={p_abs:.4f}), Spearman rho={rho_abs:.3f} (p={p_sp_abs:.4f})")
    print(f"  Effect d={d_abs:.2f}, low-order mean={mean_abs_phases[~high_order_mask].mean():.3f}, high={mean_abs_phases[high_order_mask].mean():.3f}")

    # Determine best predictor
    correlations = [
        ('phase_std', abs(r_std), d_std, p_std),
        ('phase_entropy', abs(r_entropy), d_entropy, p_entropy),
        ('mean_abs_phase', abs(r_abs), d_abs, p_abs)
    ]

    best_name, best_r, best_d, best_p = max(correlations, key=lambda x: x[1])

    print(f"\n=== Summary ===")
    print(f"Best predictor: {best_name} (r={best_r:.3f}, d={best_d:.2f}, p={best_p:.4f})")

    # Statistical assessment
    if abs(best_d) >= 0.5 and best_p < 0.01:
        status = "VALIDATED"
    elif abs(best_d) >= 0.2 and best_p < 0.05:
        status = "INCONCLUSIVE"
    else:
        status = "REFUTED"

    print(f"Status: {status}")
    print(f"Metric: effect_size={abs(best_d):.2f}")


if __name__ == '__main__':
    main()
