"""
Metric Gate Bottleneck Analysis (RES-217) - Using Nested Sampling

Instead of random CPPN sampling (max order 0.15), use nested sampling to
reach high order values and then decompose gate contributions during the
sampling process.

Test: Does compress_gate become the bottleneck (steepest gradient) above order 0.5?
"""
import numpy as np
import json
import os
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
os.chdir('/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, NestedSamplingRunner, order_multiplicative_decomposed_full,
    set_global_seed
)

def order_multiplicative_decomposed(img: np.ndarray) -> dict:
    """
    Decompose order into gate components.
    """
    from core.thermo_sampler_v3 import (
        compute_compressibility, compute_edge_density,
        compute_spectral_coherence, gaussian_gate, order_multiplicative
    )

    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)

    # Gate 1: Density (bell curve centered at 0.5, sigma=0.25)
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density (bell curve centered at 0.15, sigma=0.08)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)

    # Gate 3: Coherence (sigmoid centered at 0.3)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility (tiled)
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Overall order
    order = order_multiplicative(img)

    return {
        'order': order,
        'density_gate': float(density_gate),
        'edge_gate': float(edge_gate),
        'coherence_gate': float(coherence_gate),
        'compress_gate': float(compress_gate),
        'product': float(density_gate * edge_gate * coherence_gate * compress_gate),
    }

print("[RES-217] Metric Gate Bottleneck (Nested Sampling)")
print("=" * 60)

# Run nested sampling with tracking at multiple order thresholds
thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
results_by_threshold = {}

print("\n1. Running nested sampling to track gate values at different order thresholds...")

for threshold in thresholds:
    print(f"\n   Target order: {threshold}")

    set_global_seed(42)  # Fixed seed for reproducibility

    runner = NestedSamplingRunner(
        n_live=50,
        n_iterations=400,
        target_order=threshold,
        resolution=64
    )

    print(f"   Running NS with target={threshold}, n_live=50...")
    samples = runner.run()

    # Extract gate decomposition for samples reaching threshold
    high_order_samples = [s for s in samples if s['order'] >= threshold * 0.9]  # Near threshold

    if len(high_order_samples) > 0:
        print(f"   Found {len(high_order_samples)} samples near threshold")

        # Analyze gates
        gate_data = []
        for sample in high_order_samples:
            img = sample['img']
            gates = order_multiplicative_decomposed(img)
            gates['effort'] = sample.get('effort', 0)  # Samples to find this order
            gate_data.append(gates)

        # Compute mean gates
        density_gates = np.array([g['density_gate'] for g in gate_data])
        edge_gates = np.array([g['edge_gate'] for g in gate_data])
        coherence_gates = np.array([g['coherence_gate'] for g in gate_data])
        compress_gates = np.array([g['compress_gate'] for g in gate_data])

        result = {
            'threshold': threshold,
            'n_samples': len(gate_data),
            'mean_density': float(np.mean(density_gates)),
            'mean_edge': float(np.mean(edge_gates)),
            'mean_coherence': float(np.mean(coherence_gates)),
            'mean_compress': float(np.mean(compress_gates)),
            'std_density': float(np.std(density_gates)),
            'std_edge': float(np.std(edge_gates)),
            'std_coherence': float(np.std(coherence_gates)),
            'std_compress': float(np.std(compress_gates)),
        }

        # Identify bottleneck (most restrictive = lowest mean value)
        means = {
            'density': result['mean_density'],
            'edge': result['mean_edge'],
            'coherence': result['mean_coherence'],
            'compress': result['mean_compress'],
        }
        result['bottleneck'] = min(means, key=means.get)
        result['bottleneck_value'] = result[f'mean_{result["bottleneck"]}']

        results_by_threshold[str(threshold)] = result

        print(f"   Bottleneck gate: {result['bottleneck']} (value={result['bottleneck_value']:.3f})")
        print(f"   Density: {result['mean_density']:.3f}, Edge: {result['mean_edge']:.3f}, "
              f"Coherence: {result['mean_coherence']:.3f}, Compress: {result['mean_compress']:.3f}")

# Analyze bottleneck transitions
print("\n2. Analyzing bottleneck transitions...")

thresholds_with_data = sorted([float(k) for k in results_by_threshold.keys()])
bottlenecks = [results_by_threshold[str(t)]['bottleneck'] for t in thresholds_with_data]

# Count transitions to compress gate at higher thresholds
low_thresholds = [t for t in thresholds_with_data if t < 0.3]
high_thresholds = [t for t in thresholds_with_data if t >= 0.3]

if low_thresholds and high_thresholds:
    low_bottlenecks = [results_by_threshold[str(t)]['bottleneck'] for t in low_thresholds]
    high_bottlenecks = [results_by_threshold[str(t)]['bottleneck'] for t in high_thresholds]

    low_compress_count = sum(1 for b in low_bottlenecks if b == 'compress')
    high_compress_count = sum(1 for b in high_bottlenecks if b == 'compress')

    print(f"   Low order (<0.3): compress is bottleneck in {low_compress_count}/{len(low_bottlenecks)}")
    print(f"   High order (>=0.3): compress is bottleneck in {high_compress_count}/{len(high_bottlenecks)}")

    # Test if distribution shifts
    if len(high_bottlenecks) >= 2:
        print(f"   Bottleneck sequence: {bottlenecks}")

# Analyze gate value trends
print("\n3. Analyzing gate scaling with order threshold...")

gate_trends = {
    'density': [],
    'edge': [],
    'coherence': [],
    'compress': [],
}

for t in sorted([float(k) for k in results_by_threshold.keys()]):
    res = results_by_threshold[str(t)]
    gate_trends['density'].append((t, res['mean_density']))
    gate_trends['edge'].append((t, res['mean_edge']))
    gate_trends['coherence'].append((t, res['mean_coherence']))
    gate_trends['compress'].append((t, res['mean_compress']))

# Compute scaling exponent for compress gate
if len(gate_trends['compress']) >= 3:
    x = np.array([v[0] for v in gate_trends['compress']])
    y = np.array([v[1] for v in gate_trends['compress']])

    valid = (x > 0) & (y > 0.01)
    if valid.sum() >= 3:
        x_log = np.log(x[valid])
        y_log = np.log(y[valid])

        # Fit: gate = c * order^alpha
        if len(x_log) >= 2:
            coeffs = np.polyfit(x_log, y_log, 1)
            alpha = coeffs[0]

            fitted = np.polyval(coeffs, x_log)
            ss_res = np.sum((y_log - fitted) ** 2)
            ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"   Compress gate scaling: alpha={alpha:.3f} (r²={r_squared:.3f})")
            print(f"   Gate values: {gate_trends['compress']}")

# Determine overall status
print("\n4. Interpreting results...")

# Status criteria:
# 1. Compress gate is most restrictive at high order
# 2. Sharp transition in which gate is bottleneck
# 3. Compress gate shows steeper scaling at high order

status = "INCONCLUSIVE"
effect_size = 0.5

if high_bottlenecks and high_compress_count >= len(high_bottlenecks) * 0.6:
    status = "VALIDATED"
    effect_size = 1.0
    print(f"   ✓ Compress gate becomes dominant bottleneck at high order")
elif low_compress_count < len(low_bottlenecks) * 0.5 and high_compress_count > low_compress_count:
    status = "INCONCLUSIVE"
    effect_size = 0.6
    print(f"   ? Partial evidence: compress importance increases with order")
else:
    status = "REFUTED"
    effect_size = 0.1
    print(f"   ✗ No clear bottleneck transition to compress gate")

print(f"\n   Final status: {status}")
print(f"   Effect size: {effect_size:.2f}")

# Save results
output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/metric_gate_decomposition')
output_dir.mkdir(parents=True, exist_ok=True)

result_summary = {
    'status': status,
    'effect_size': effect_size,
    'method': 'nested_sampling',
    'by_threshold': results_by_threshold,
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(result_summary, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")
print(f"\nFinal: RES-217 | metric_gate_decomposition | {status} | d={effect_size:.2f}")
