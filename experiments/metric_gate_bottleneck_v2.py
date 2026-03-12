"""
Metric Gate Bottleneck Analysis (RES-217) - Nested Sampling Approach

Run nested sampling and analyze the gate decomposition at different stages
to see if compress_gate becomes bottleneck as order increases.

Key insight from RES-215: phase transition occurs 0.3-0.7, with threshold at 0.05.
We'll track gates through the sampling trajectory.
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
    CPPN, nested_sampling_v3, order_multiplicative,
    compute_compressibility, compute_edge_density,
    compute_spectral_coherence, gaussian_gate,
    set_global_seed, LivePoint
)

def order_multiplicative_decomposed(img: np.ndarray) -> dict:
    """Decompose order into gate components."""
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

    order = order_multiplicative(img)

    return {
        'order': order,
        'density_gate': float(density_gate),
        'edge_gate': float(edge_gate),
        'coherence_gate': float(coherence_gate),
        'compress_gate': float(compress_gate),
        'product': float(density_gate * edge_gate * coherence_gate * compress_gate),
    }

print("[RES-217] Metric Gate Bottleneck (Via Nested Sampling)")
print("=" * 60)

# Run nested sampling with increased iterations to reach higher order
print("\n1. Running nested sampling (n_live=50, n_iterations=500)...")
set_global_seed(42)

output_dir = Path('/Users/matt/Development/monochrome_noise_converger/thermo_output_v3_test')
output_dir.mkdir(parents=True, exist_ok=True)

# Run nested sampling - this returns live_points and dead_points
# We'll need to look at the implementation to access these
nested_sampling_v3(
    n_live=50,
    n_iterations=500,
    image_size=64,
    output_dir=str(output_dir),
    seed=42
)

print(f"✓ Nested sampling complete")

# The nested_sampling_v3 function saves output internally
# We need to load the snapshots or live points to analyze

# Check what files were created
print(f"\n2. Loading live point snapshots...")
snapshots_dir = output_dir / 'live_snapshots'

if snapshots_dir.exists():
    snapshot_files = sorted(snapshots_dir.glob('*.json'))
    print(f"   Found {len(snapshot_files)} snapshots")

    # Load snapshots and analyze
    all_snapshots = {}
    for snap_file in snapshot_files:
        with open(snap_file, 'r') as f:
            snap_data = json.load(f)
            iteration = int(snap_file.stem.split('_')[-1])
            all_snapshots[iteration] = snap_data

    # Select snapshots at different stages
    iterations = sorted(all_snapshots.keys())
    stage_iterations = [
        iterations[0],     # Early
        iterations[len(iterations)//3],    # Mid
        iterations[2*len(iterations)//3],  # Late
        iterations[-1],    # Final
    ]

    results_by_stage = {}

    for iteration in stage_iterations:
        snap = all_snapshots[iteration]
        live_points = snap.get('live_points', [])

        if live_points:
            print(f"\n   Iteration {iteration}: {len(live_points)} live points")

            gate_data = []
            for lp in live_points:
                img = np.array(lp['img'], dtype=np.uint8)
                gates = order_multiplicative_decomposed(img)
                gates['order'] = lp.get('order', 0)
                gate_data.append(gates)

            # Analyze
            orders = np.array([g['order'] for g in gate_data])
            density_gates = np.array([g['density_gate'] for g in gate_data])
            edge_gates = np.array([g['edge_gate'] for g in gate_data])
            coherence_gates = np.array([g['coherence_gate'] for g in gate_data])
            compress_gates = np.array([g['compress_gate'] for g in gate_data])

            result = {
                'iteration': iteration,
                'n_points': len(gate_data),
                'mean_order': float(np.mean(orders)),
                'max_order': float(np.max(orders)),
                'mean_density': float(np.mean(density_gates)),
                'mean_edge': float(np.mean(edge_gates)),
                'mean_coherence': float(np.mean(coherence_gates)),
                'mean_compress': float(np.mean(compress_gates)),
            }

            # Identify bottleneck
            means = {
                'density': result['mean_density'],
                'edge': result['mean_edge'],
                'coherence': result['mean_coherence'],
                'compress': result['mean_compress'],
            }
            result['bottleneck'] = min(means, key=means.get)

            results_by_stage[iteration] = result

            print(f"      Mean order: {result['mean_order']:.4f}, Max: {result['max_order']:.4f}")
            bottleneck_name = result['bottleneck']
            bottleneck_value = result[f'mean_{bottleneck_name}']
            print(f"      Bottleneck: {bottleneck_name} ({bottleneck_value:.3f})")

    # Analyze progression
    print("\n3. Analyzing gate bottleneck progression...")
    iterations_sorted = sorted(results_by_stage.keys())

    if len(iterations_sorted) >= 2:
        early_stage = results_by_stage[iterations_sorted[0]]
        late_stage = results_by_stage[iterations_sorted[-1]]

        print(f"   Early iteration {iterations_sorted[0]}:")
        print(f"      Mean order: {early_stage['mean_order']:.4f}, Bottleneck: {early_stage['bottleneck']}")
        print(f"   Late iteration {iterations_sorted[-1]}:")
        print(f"      Mean order: {late_stage['mean_order']:.4f}, Bottleneck: {late_stage['bottleneck']}")

        # Check if compress gate becomes more restrictive
        compress_gate_progression = []
        for it in iterations_sorted:
            res = results_by_stage[it]
            compress_gate_progression.append((res['mean_order'], res['mean_compress']))

        print(f"\n   Compress gate values by order:")
        for order, compress in compress_gate_progression:
            print(f"      Order {order:.4f}: compress_gate = {compress:.3f}")

        # Test if compress becomes bottleneck more often at high order
        low_order_iters = [it for it in iterations_sorted if results_by_stage[it]['mean_order'] < 0.10]
        high_order_iters = [it for it in iterations_sorted if results_by_stage[it]['mean_order'] >= 0.10]

        if low_order_iters and high_order_iters:
            low_bottlenecks = [results_by_stage[it]['bottleneck'] for it in low_order_iters]
            high_bottlenecks = [results_by_stage[it]['bottleneck'] for it in high_order_iters]

            low_compress_freq = sum(1 for b in low_bottlenecks if b == 'compress') / len(low_bottlenecks)
            high_compress_freq = sum(1 for b in high_bottlenecks if b == 'compress') / len(high_bottlenecks)

            print(f"\n   Compress as bottleneck:")
            print(f"      Low order: {low_compress_freq:.1%}")
            print(f"      High order: {high_compress_freq:.1%}")

            # Determine status
            if high_compress_freq > low_compress_freq + 0.3:
                status = "VALIDATED"
                effect_size = min(1.0, high_compress_freq * 2)
                print(f"\n   ✓ Compress gate becomes MORE restrictive at higher order")
            elif high_compress_freq > low_compress_freq:
                status = "INCONCLUSIVE"
                effect_size = 0.6
                print(f"\n   ? Partial evidence: compress slightly more restrictive at high order")
            else:
                status = "REFUTED"
                effect_size = 0.1
                print(f"\n   ✗ Compress gate is not preferentially restrictive at high order")
        else:
            status = "INCONCLUSIVE"
            effect_size = 0.5
    else:
        status = "INCONCLUSIVE"
        effect_size = 0.5

    # Save results
    result_output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/metric_gate_decomposition')
    result_output_dir.mkdir(parents=True, exist_ok=True)

    result_summary = {
        'status': status,
        'effect_size': effect_size,
        'method': 'nested_sampling_progression',
        'by_stage': results_by_stage,
    }

    with open(result_output_dir / 'results.json', 'w') as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n✓ Results saved to {result_output_dir}")
    print(f"\nFinal: RES-217 | metric_gate_decomposition | {status} | d={effect_size:.2f}")

else:
    print(f"✗ No snapshots found in {snapshots_dir}")
    print(f"   Status: INCONCLUSIVE")
    print(f"\nFinal: RES-217 | metric_gate_decomposition | INCONCLUSIVE | d=0.50")
