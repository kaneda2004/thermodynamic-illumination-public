"""
RES-095: Test if high-order CPPNs exhibit better gradient flow than low-order CPPNs.

Gradient flow = how well gradients propagate through network layers
Focus on: output range, saturation effects
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def analyze_gradient_flow(cppn, coords_x, coords_y):
    """
    Compute gradient flow metrics for a CPPN.
    - Output range: span of activations at each layer
    - Gradient magnitude: sensitivity to weight changes
    - Saturation: fraction of neurons in flat regions
    """
    # Forward pass to get activations
    r = np.sqrt(coords_x**2 + coords_y**2)
    bias = np.ones_like(coords_x)
    values = {0: coords_x, 1: coords_y, 2: r, 3: bias}

    layer_activations = []
    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(coords_x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        from core.thermo_sampler_v3 import ACTIVATIONS
        values[nid] = ACTIVATIONS[node.activation](total)
        layer_activations.append(values[nid])

    # Gradient flow metrics
    output_activation = values[cppn.output_id]

    # Output range (should be [0,1] for good flow)
    output_range = np.max(output_activation) - np.min(output_activation)

    # Saturation: fraction of output near extremes
    saturation = np.mean((output_activation < 0.1) | (output_activation > 0.9))

    # Mean gradient magnitude (via finite differences on weights)
    weight_grads = []
    eps = 1e-4
    for conn in cppn.connections:
        old_w = conn.weight

        # Forward difference
        conn.weight = old_w + eps
        output_plus = values.copy()
        for nid in eval_order:
            node = next(n for n in cppn.nodes if n.id == nid)
            total = np.zeros_like(coords_x) + node.bias
            for c in cppn.connections:
                if c.to_id == nid and c.enabled and c.from_id in output_plus:
                    total += output_plus[c.from_id] * c.weight
            from core.thermo_sampler_v3 import ACTIVATIONS
            output_plus[nid] = ACTIVATIONS[node.activation](total)

        grad = (output_plus[cppn.output_id] - output_activation) / eps
        weight_grads.append(np.mean(np.abs(grad)))
        conn.weight = old_w

    mean_grad_magnitude = np.mean(weight_grads) if weight_grads else 0.0

    return {
        'output_range': output_range,
        'saturation': saturation,
        'mean_grad_magnitude': mean_grad_magnitude
    }


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    # Generate coordinates
    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate high-order and low-order CPPNs via nested sampling
    print("Generating high-order CPPNs...")
    high_order_cppns = []
    high_order_scores = []

    for i in range(n_samples):
        cppn, score = nested_sampling(max_iterations=100, n_live=20)
        high_order_cppns.append(cppn)
        high_order_scores.append(score)

    print("Generating low-order CPPNs...")
    low_order_cppns = []
    low_order_scores = []

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization
        order = compute_order(cppn.activate(coords_x, coords_y))
        low_order_cppns.append(cppn)
        low_order_scores.append(order)

    # Analyze gradient flow
    print("Analyzing gradient flow...")
    high_grad_flows = [analyze_gradient_flow(c, coords_x, coords_y) for c in high_order_cppns]
    low_grad_flows = [analyze_gradient_flow(c, coords_x, coords_y) for c in low_order_cppns]

    # Extract metrics
    high_mag = np.array([g['mean_grad_magnitude'] for g in high_grad_flows])
    low_mag = np.array([g['mean_grad_magnitude'] for g in low_grad_flows])

    high_saturation = np.array([g['saturation'] for g in high_grad_flows])
    low_saturation = np.array([g['saturation'] for g in low_grad_flows])

    # Statistical tests
    t_mag, p_mag = stats.ttest_ind(high_mag, low_mag)
    t_sat, p_sat = stats.ttest_ind(high_saturation, low_saturation)

    # Effect sizes
    pooled_std_mag = np.sqrt((np.std(high_mag)**2 + np.std(low_mag)**2) / 2)
    d_mag = (np.mean(high_mag) - np.mean(low_mag)) / pooled_std_mag

    pooled_std_sat = np.sqrt((np.std(high_saturation)**2 + np.std(low_saturation)**2) / 2)
    d_sat = (np.mean(high_saturation) - np.mean(low_saturation)) / pooled_std_sat

    # Correlation with order
    all_scores = np.concatenate([high_order_scores, low_order_scores])
    all_mag = np.concatenate([high_mag, low_mag])
    all_sat = np.concatenate([high_saturation, low_saturation])

    corr_mag, p_corr_mag = stats.pearsonr(all_scores, all_mag)
    corr_sat, p_corr_sat = stats.pearsonr(all_scores, all_sat)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order gradient magnitude: {np.mean(high_mag):.4f} +/- {np.std(high_mag):.4f}")
    print(f"Low-order gradient magnitude: {np.mean(low_mag):.4f} +/- {np.std(low_mag):.4f}")
    print(f"Effect size (Cohen's d): {d_mag:.2f}")
    print(f"p-value: {p_mag:.2e}")
    print(f"Correlation with order (r): {corr_mag:.3f}, p={p_corr_mag:.2e}")

    print(f"\nHigh-order saturation: {np.mean(high_saturation):.4f}")
    print(f"Low-order saturation: {np.mean(low_saturation):.4f}")
    print(f"Saturation effect size: {d_sat:.2f}")
    print(f"Saturation correlation with order (r): {corr_sat:.3f}, p={p_corr_sat:.2e}")

    # Determine validation
    validated = d_mag > 0.5 and p_corr_mag < 0.01
    status = 'validated' if validated else 'refuted'

    print(f"\nSTATUS: {status}")

    return validated, d_mag, p_mag


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
