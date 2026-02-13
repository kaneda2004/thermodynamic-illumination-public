"""
RES-149: Test if multiplicative gates show high redundancy.

Hypothesis: improving one gate rarely helps if another is bottleneck
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_multiplicative_gates(image, threshold=0.5):
    """Compute the four multiplicative gates from order metric."""
    binary = (image > threshold).astype(int)

    # Density gate (fraction of 1s)
    density = np.mean(binary)
    density_gate = (1 - np.abs(density - 0.5) * 2) if density > 0 else 0

    # Compression gate (via RLE)
    runs = []
    for row in binary:
        current = 1
        for i in range(1, len(row)):
            if row[i] == row[i-1]:
                current += 1
            else:
                runs.append(current)
                current = 1
    compress_gate = min(1.0, np.mean(runs) / 32) if runs else 0

    # Coherence gate (via structure tensor)
    gy, gx = np.gradient(binary.astype(float))
    gx2 = gx**2
    gy2 = gy**2
    gxy = gx * gy

    coherence = (((gx2 - gy2)**2 + 4*gxy**2)**0.5) / (gx2 + gy2 + 1e-10)
    coherence_gate = np.mean(coherence[gx2 + gy2 > 0.01]) if np.any(gx2 + gy2 > 0.01) else 0

    # Edge gate (edge density)
    edge_density = np.mean(np.abs(np.diff(binary, axis=0))) + np.mean(np.abs(np.diff(binary, axis=1)))
    edge_gate = min(1.0, edge_density)

    gates = {
        'density': max(0, min(1, density_gate)),
        'compress': max(0, min(1, compress_gate)),
        'coherence': max(0, min(1, coherence_gate)),
        'edge': max(0, min(1, edge_gate))
    }

    return gates


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPNs with varying order
    print("Generating CPPNs...")
    orders = []
    gate_values = []
    min_gates = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)
        gates = compute_multiplicative_gates(img)

        orders.append(order)
        gate_values.append([gates['density'], gates['compress'], gates['coherence'], gates['edge']])
        min_gate = min(gates.values())
        min_gates.append(min_gate)

    orders = np.array(orders)
    gate_values = np.array(gate_values)
    min_gates = np.array(min_gates)

    # Correlation between min gate and order
    corr_min, p_min = stats.pearsonr(orders, min_gates)

    # Gate correlations (should be high if redundant)
    gate_corrs = []
    gate_names = ['density', 'compress', 'coherence', 'edge']
    for i in range(4):
        for j in range(i+1, 4):
            corr, _ = stats.pearsonr(gate_values[:, i], gate_values[:, j])
            gate_corrs.append(abs(corr))

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Correlation (min gate vs order): r={corr_min:.3f}, p={p_min:.2e}")
    print(f"Mean inter-gate correlation: {np.mean(gate_corrs):.3f}")
    print(f"Gate redundancy (eta-squared): {np.var(orders - min_gates) / np.var(orders):.3f}")

    effect_size = abs(corr_min)
    validated = effect_size > 0.5 and p_min < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_min


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: r={effect_size:.3f}, p={p_value:.2e}")
