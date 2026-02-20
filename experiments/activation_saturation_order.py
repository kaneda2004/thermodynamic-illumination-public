"""
RES-130: Activation saturation vs CPPN order correlation

Hypothesis: Total network activation saturation (fraction of neurons at extreme values
>0.95 or <-0.95) positively correlates with CPPN output order.

Tests whether high-order structured images require more binary-like activations.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, ACTIVATIONS

def measure_saturation(cppn: CPPN, size: int = 32, threshold: float = 0.95) -> dict:
    """
    Measure activation saturation across all layers.

    Returns:
        dict with total_saturation, layer_saturations, extreme_positive, extreme_negative
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}

    # Track saturations per layer
    all_activations = []
    layer_saturations = []

    # Get evaluation order (hidden + output)
    hidden_ids = sorted([n.id for n in cppn.nodes
                        if n.id not in cppn.input_ids and n.id != cppn.output_id])
    eval_order = hidden_ids + [cppn.output_id]

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        # Apply activation and store
        activated = ACTIVATIONS[node.activation](total)
        values[nid] = activated

        # Measure saturation for this layer
        flat = activated.flatten()
        all_activations.append(flat)

        # Count extreme values (> threshold or < -threshold, or for [0,1] activations: very close to 0 or 1)
        if node.activation in ['sigmoid']:
            # Output range [0,1] - check near boundaries
            extreme = np.sum((flat < (1 - threshold)) | (flat > threshold)) / len(flat)
        else:
            # Check for extreme absolute values
            extreme = np.sum(np.abs(flat) > threshold) / len(flat)
        layer_saturations.append(extreme)

    # Combine all activations
    all_flat = np.concatenate(all_activations)

    # Total saturation across all layers
    extreme_positive = np.sum(all_flat > threshold) / len(all_flat)
    extreme_negative = np.sum(all_flat < -threshold) / len(all_flat)
    total_saturation = (np.sum(np.abs(all_flat) > threshold) / len(all_flat))

    return {
        'total_saturation': total_saturation,
        'extreme_positive': extreme_positive,
        'extreme_negative': extreme_negative,
        'layer_saturations': layer_saturations,
        'mean_abs_activation': np.mean(np.abs(all_flat)),
    }


def run_experiment(n_samples: int = 500, seed: int = 42):
    """Run correlation experiment between saturation and order."""
    np.random.seed(seed)

    saturations = []
    orders = []
    mean_abs = []

    for i in range(n_samples):
        # Create random CPPN
        cppn = CPPN()

        # Render and compute order
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        # Measure saturation
        sat_metrics = measure_saturation(cppn, size=32)

        saturations.append(sat_metrics['total_saturation'])
        orders.append(order)
        mean_abs.append(sat_metrics['mean_abs_activation'])

    # Statistical analysis
    saturations = np.array(saturations)
    orders = np.array(orders)
    mean_abs = np.array(mean_abs)

    # Correlation: saturation vs order
    r_sat, p_sat = stats.pearsonr(saturations, orders)

    # Correlation: mean absolute activation vs order
    r_abs, p_abs = stats.pearsonr(mean_abs, orders)

    # Effect size (Cohen's d): compare high vs low order groups
    median_order = np.median(orders)
    high_order = saturations[orders > median_order]
    low_order = saturations[orders <= median_order]

    pooled_std = np.sqrt((np.std(high_order)**2 + np.std(low_order)**2) / 2)
    cohens_d = (np.mean(high_order) - np.mean(low_order)) / pooled_std if pooled_std > 0 else 0

    # Print results
    print("=" * 60)
    print("RES-130: Activation Saturation vs CPPN Order")
    print("=" * 60)
    print(f"\nSamples: {n_samples}")
    print(f"\nSaturation (fraction |activation| > 0.95):")
    print(f"  Mean: {np.mean(saturations):.4f}")
    print(f"  Std:  {np.std(saturations):.4f}")
    print(f"  Range: [{np.min(saturations):.4f}, {np.max(saturations):.4f}]")
    print(f"\nOrder metric:")
    print(f"  Mean: {np.mean(orders):.4f}")
    print(f"  Std:  {np.std(orders):.4f}")
    print(f"  Range: [{np.min(orders):.4f}, {np.max(orders):.4f}]")
    print(f"\n--- Saturation vs Order ---")
    print(f"  Pearson r: {r_sat:.4f}")
    print(f"  p-value:   {p_sat:.2e}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"\n--- Mean |Activation| vs Order ---")
    print(f"  Pearson r: {r_abs:.4f}")
    print(f"  p-value:   {p_abs:.2e}")

    # High vs low order saturation comparison
    print(f"\nHigh-order group saturation: {np.mean(high_order):.4f} +/- {np.std(high_order):.4f}")
    print(f"Low-order group saturation:  {np.mean(low_order):.4f} +/- {np.std(low_order):.4f}")

    # Conclusion
    print("\n" + "=" * 60)
    significant = p_sat < 0.01
    positive_effect = r_sat > 0
    large_effect = abs(cohens_d) > 0.5

    if significant and positive_effect and large_effect:
        status = "VALIDATED"
        conclusion = "Saturation positively correlates with order (r={:.3f}, d={:.3f})".format(r_sat, cohens_d)
    elif significant and not positive_effect and large_effect:
        status = "REFUTED"
        conclusion = "Saturation NEGATIVELY correlates with order (r={:.3f}, d={:.3f})".format(r_sat, cohens_d)
    elif not significant:
        status = "REFUTED" if p_sat > 0.05 else "INCONCLUSIVE"
        conclusion = "No significant correlation (r={:.3f}, p={:.2e})".format(r_sat, p_sat)
    else:
        status = "INCONCLUSIVE"
        conclusion = "Significant but small effect (r={:.3f}, d={:.3f})".format(r_sat, cohens_d)

    print(f"STATUS: {status}")
    print(f"CONCLUSION: {conclusion}")
    print("=" * 60)

    return {
        'status': status.lower(),
        'r': r_sat,
        'p': p_sat,
        'd': cohens_d,
        'conclusion': conclusion
    }


if __name__ == '__main__':
    results = run_experiment(n_samples=500, seed=42)
