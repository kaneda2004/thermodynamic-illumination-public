"""
RES-180: CPPN distinctness scales with weight precision

Hypothesis: Weight discretization affects the number of unique high-order patterns
CPPNs can produce. Lower precision (fewer bits) should yield fewer distinct patterns.

Domain: network_capacity
"""

import numpy as np
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats


def discretize_weights(weights: np.ndarray, bits: int) -> np.ndarray:
    """Discretize weights to n-bit precision within [-4, 4] range."""
    levels = 2 ** bits
    # Map to [0, 1], discretize, map back
    w_clipped = np.clip(weights, -4, 4)
    w_norm = (w_clipped + 4) / 8  # [0, 1]
    w_discrete = np.round(w_norm * (levels - 1)) / (levels - 1)
    return w_discrete * 8 - 4


def image_to_hash(img: np.ndarray) -> str:
    """Convert binary image to hash for uniqueness tracking."""
    return np.packbits(img.flatten()).tobytes().hex()


def run_experiment():
    set_global_seed(42)

    n_samples = 500
    bit_levels = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]  # Different precision levels
    order_threshold = 0.1  # Only count "high-order" patterns

    results = {bits: {'unique_hashes': set(), 'unique_count': 0, 'high_order_count': 0}
               for bits in bit_levels}
    results['continuous'] = {'unique_hashes': set(), 'unique_count': 0, 'high_order_count': 0}

    # Generate random weight vectors
    weight_vectors = np.random.randn(n_samples, 5) * 1.0  # 5 weights (4 inputs + 1 bias)

    for i in range(n_samples):
        w = weight_vectors[i]

        # Test continuous (full precision)
        cppn = CPPN()
        cppn.set_weights(w.copy())
        img = cppn.render(32)
        order = order_multiplicative(img)
        h = image_to_hash(img)
        results['continuous']['unique_hashes'].add(h)
        if order >= order_threshold:
            results['continuous']['high_order_count'] += 1

        # Test each discretization level
        for bits in bit_levels:
            w_discrete = discretize_weights(w, bits)
            cppn = CPPN()
            cppn.set_weights(w_discrete)
            img = cppn.render(32)
            order = order_multiplicative(img)
            h = image_to_hash(img)
            results[bits]['unique_hashes'].add(h)
            if order >= order_threshold:
                results[bits]['high_order_count'] += 1

    # Convert sets to counts
    for key in results:
        results[key]['unique_count'] = len(results[key]['unique_hashes'])
        del results[key]['unique_hashes']  # Remove non-serializable set

    # Compute statistics
    bit_array = np.array(bit_levels)
    unique_array = np.array([results[b]['unique_count'] for b in bit_levels])
    high_order_array = np.array([results[b]['high_order_count'] for b in bit_levels])

    # Correlation between bits and unique count
    rho_unique, p_unique = stats.spearmanr(bit_array, unique_array)
    rho_high, p_high = stats.spearmanr(bit_array, high_order_array)

    # Effect size: compare lowest (4-bit) vs highest (16-bit)
    # Using ratio as measure of effect
    unique_ratio = results[16]['unique_count'] / max(1, results[4]['unique_count'])
    high_order_ratio = results[16]['high_order_count'] / max(1, results[4]['high_order_count'])

    # Log-linear regression for capacity scaling
    log_bits = np.log2(2 ** bit_array)  # = bit_array
    log_unique = np.log2(unique_array.clip(1))
    slope, intercept, r, p_slope, se = stats.linregress(log_bits, log_unique)

    continuous_unique = results['continuous']['unique_count']
    continuous_high = results['continuous']['high_order_count']

    # Print results
    print("=" * 60)
    print("RES-180: Weight Precision vs Pattern Capacity")
    print("=" * 60)
    print(f"\nSamples: {n_samples}, Order threshold: {order_threshold}")
    print("\nResults by bit precision:")
    print(f"{'Bits':<8} {'Unique':<12} {'High-Order':<12} {'Unique %':<12}")
    print("-" * 44)
    for bits in bit_levels:
        u = results[bits]['unique_count']
        h = results[bits]['high_order_count']
        print(f"{bits:<8} {u:<12} {h:<12} {100*u/n_samples:.1f}%")
    print(f"{'Cont.':<8} {continuous_unique:<12} {continuous_high:<12} {100*continuous_unique/n_samples:.1f}%")

    print(f"\nStatistics:")
    print(f"  Spearman rho (bits vs unique): {rho_unique:.3f}, p={p_unique:.2e}")
    print(f"  Spearman rho (bits vs high-order): {rho_high:.3f}, p={p_high:.2e}")
    print(f"  16-bit / 4-bit unique ratio: {unique_ratio:.2f}x")
    print(f"  16-bit / 4-bit high-order ratio: {high_order_ratio:.2f}x")
    print(f"  Log-linear scaling exponent: {slope:.3f} (r={r:.3f})")

    # Cohen's d approximation: effect of maximum discretization
    # Using pooled std estimate from unique counts
    mean_all = np.mean(unique_array)
    std_all = np.std(unique_array)
    d = (results[16]['unique_count'] - results[4]['unique_count']) / max(std_all, 1)

    print(f"\n  Effect size (Cohen's d, 16 vs 4 bit): {d:.2f}")

    # Check for threshold effect: is there saturation after a certain bit level?
    # Find the knee point where capacity reaches 90% of max
    max_unique = max(unique_array)
    threshold_90 = 0.9 * max_unique
    saturation_bits = None
    for i, bits in enumerate(bit_levels):
        if unique_array[i] >= threshold_90:
            saturation_bits = bits
            break

    # Effect size: compare below-threshold vs above-threshold
    below_sat = unique_array[unique_array < threshold_90]
    above_sat = unique_array[unique_array >= threshold_90]
    if len(below_sat) > 0 and len(above_sat) > 0:
        d_saturation = (np.mean(above_sat) - np.mean(below_sat)) / np.std(unique_array)
        saturation_ratio = np.mean(above_sat) / np.mean(below_sat)
    else:
        d_saturation = 0
        saturation_ratio = 1

    # Determine outcome - test hypothesis about threshold behavior
    has_saturation = saturation_bits is not None and saturation_bits <= 8
    saturation_effect = d_saturation > 0.5

    if has_saturation and saturation_effect:
        status = "refuted"
        summary = f"CPPN capacity saturates at ~{saturation_bits} bits, not scaling continuously. " \
                  f"2-bit: {results[2]['unique_count']} unique ({results[2]['unique_count']*100/n_samples:.0f}%), " \
                  f"5-bit: {results[5]['unique_count']} unique ({results[5]['unique_count']*100/n_samples:.0f}%), " \
                  f"16-bit: {results[16]['unique_count']} unique ({results[16]['unique_count']*100/n_samples:.0f}%). " \
                  f"Saturation ratio {saturation_ratio:.1f}x, d={d_saturation:.2f}."
    elif abs(d) > 0.5 and p_unique < 0.01 and rho_unique > 0:
        status = "validated"
        summary = f"Higher precision yields more unique patterns (rho={rho_unique:.2f}, d={d:.1f})."
    else:
        status = "inconclusive"
        summary = f"Weak/no effect (rho={rho_unique:.2f}, p={p_unique:.2e}, d={d:.2f})."

    print(f"\nSTATUS: {status.upper()}")
    print(f"SUMMARY: {summary}")

    # Save results
    output = {
        'hypothesis': 'CPPN distinctness scales with weight precision',
        'n_samples': n_samples,
        'order_threshold': order_threshold,
        'results_by_bits': results,
        'statistics': {
            'rho_unique': float(rho_unique),
            'p_unique': float(p_unique),
            'rho_high_order': float(rho_high),
            'p_high_order': float(p_high),
            'unique_ratio_16_4': float(unique_ratio),
            'high_order_ratio_16_4': float(high_order_ratio),
            'scaling_exponent': float(slope),
            'scaling_r': float(r),
            'effect_size_d': float(d),
            'saturation_bits': saturation_bits,
            'saturation_effect_d': float(d_saturation),
            'saturation_ratio': float(saturation_ratio),
        },
        'status': status,
        'summary': summary
    }

    out_path = Path(__file__).parent.parent / 'results/weight_precision_capacity/results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    return output


if __name__ == '__main__':
    run_experiment()
