#!/usr/bin/env python3
"""
Hypothesis Testing: Exponential Rarity in 64×64 (4096-bit) Image Space

This module tests the hypothesis that structured images are exponentially rare
under uniform random sampling, with the fraction of images having Kolmogorov
complexity < K bits being at most 2^(K-N) where N=4096 for 64×64 images.

Key experiments:
1. Uniform baseline - measure bits for random 64×64 images (expect B >> 100)
2. CPPN comparison - quantify bit savings at 64×64 (expect B ~ 2-5)
3. Scaling analysis - verify bits scale with image dimension
4. Kolmogorov validation - compare empirical to theoretical bounds
5. Phase transition detection - find order/disorder boundary

Usage:
    python experiments/hypothesis_64x64.py [experiment_name]

    experiment_name: uniform | cppn | scaling | kolmogorov | phase | all
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    nested_sampling_with_prior,
    order_multiplicative,
    order_kolmogorov_proxy,
    ORDER_METRICS,
    CPPN,
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_order(d):
    """Get order value from dead point (handles both dict and dataclass)."""
    if hasattr(d, 'order_value'):
        return d.order_value
    return d.get('order', d.get('order_value', 0))


def get_log_X(d):
    """Get log_X from dead point (handles both dict and dataclass)."""
    if hasattr(d, 'log_X'):
        return d.log_X
    return d.get('log_X', 0)


def compute_bits_to_threshold(dead_points, threshold, n_live):
    """
    Compute bits needed to find images above threshold using log_X crossing.

    B(T) = -log_X(T) / ln(2) where log_X(T) is the prior volume coordinate
    when threshold T is first reached.

    Returns: (bits, reached) tuple
    """
    for d in dead_points:
        if get_order(d) >= threshold:
            return -get_log_X(d) / np.log(2), True

    # Threshold not reached - return lower bound
    max_bits_probed = len(dead_points) / (n_live * np.log(2))
    return max_bits_probed, False


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output_dir = Path("results/hypothesis_64x64")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filepath}")


# ============================================================================
# EXPERIMENT 1: UNIFORM BASELINE AT 64×64
# ============================================================================

def experiment_uniform_baseline(n_live=50, n_iterations=2000, image_size=64):
    """
    Measure bits required for uniform random sampling at 64×64.

    Hypothesis: B >> 100 bits (effectively intractable)

    For 64×64 (4096 bits), uniform random sampling should require enormous
    numbers of samples to find any structure. This establishes the baseline
    for measuring how much structure neural priors provide.
    """
    print("=" * 70)
    print("EXPERIMENT 1: UNIFORM BASELINE AT 64×64")
    print("=" * 70)
    print()
    print("Hypothesis: Structured images are exponentially rare under uniform")
    print("           sampling. Expect B >> 100 bits (2^100+ samples needed)")
    print()

    results = {
        'experiment': 'uniform_baseline',
        'image_size': image_size,
        'n_bits': image_size * image_size,
        'n_live': n_live,
        'n_iterations': n_iterations,
        'timestamp': datetime.now().isoformat(),
        'thresholds': {},
    }

    print(f"Running uniform prior nested sampling...")
    print(f"  Image size: {image_size}×{image_size} = {image_size**2} bits")
    print(f"  n_live: {n_live}, n_iterations: {n_iterations}")
    print()

    dead_points, _, _ = nested_sampling_with_prior(
        prior_type='uniform',
        order_fn=order_multiplicative,
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        verbose=False
    )

    # Maximum bits we could probe
    max_bits_probed = n_iterations / (n_live * np.log(2))
    results['max_bits_probed'] = max_bits_probed

    print(f"Max bits probed: {max_bits_probed:.1f}")
    print()

    # Check various thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]

    print(f"{'Threshold':<12} {'Bits (B)':<15} {'Samples (2^B)':<20} {'Status'}")
    print("-" * 70)

    for t in thresholds:
        bits, reached = compute_bits_to_threshold(dead_points, t, n_live)
        results['thresholds'][str(t)] = {
            'bits': bits,
            'reached': reached,
            'samples_needed': 2**bits if bits < 100 else f"2^{bits:.1f}"
        }

        if not reached:
            print(f"{t:<12.2f} {'≥' + f'{bits:.1f}':<15} {'≥2^' + f'{bits:.0f}':<20} {'NOT REACHED'}")
        else:
            print(f"{t:<12.2f} {bits:<15.1f} {f'2^{bits:.0f}':<20} {'Reached'}")

    print()

    # Best order achieved
    best_order = max(get_order(d) for d in dead_points) if dead_points else 0
    results['best_order_achieved'] = best_order
    print(f"Best order achieved: {best_order:.4f}")

    # Interpretation
    print()
    print("INTERPRETATION:")
    print("-" * 70)
    if not any(results['thresholds'][str(t)]['reached'] for t in thresholds):
        print("  ✓ HYPOTHESIS SUPPORTED: No meaningful structure found")
        print(f"  Probed {max_bits_probed:.1f} bits without finding order > {thresholds[0]}")
        print(f"  This suggests B >> {max_bits_probed:.0f} bits for structured images")
        results['hypothesis_supported'] = True
    else:
        lowest_reached = min(t for t in thresholds if results['thresholds'][str(t)]['reached'])
        print(f"  Threshold {lowest_reached} reached (unexpected for uniform)")
        results['hypothesis_supported'] = False

    save_results(results, 'uniform_baseline_64x64.json')
    return results


# ============================================================================
# EXPERIMENT 2: CPPN COMPARISON AT 64×64
# ============================================================================

def experiment_cppn_comparison(n_live=50, n_iterations=500, image_size=64):
    """
    Measure bits required for CPPN prior at 64×64.

    Hypothesis: B ~ 2-5 bits (extrapolated from 32×32 results)

    CPPNs should easily find structured images due to their smooth, coordinate-
    based generation. The bit savings compared to uniform quantifies how much
    structure the CPPN architecture provides.
    """
    print("=" * 70)
    print("EXPERIMENT 2: CPPN COMPARISON AT 64×64")
    print("=" * 70)
    print()
    print("Hypothesis: CPPN reaches structure easily. Expect B ~ 2-5 bits")
    print()

    results = {
        'experiment': 'cppn_comparison',
        'image_size': image_size,
        'n_live': n_live,
        'n_iterations': n_iterations,
        'timestamp': datetime.now().isoformat(),
        'thresholds': {},
    }

    print(f"Running CPPN nested sampling at {image_size}×{image_size}...")

    # Run with both multiplicative and kolmogorov metrics
    for metric_name in ['multiplicative', 'kolmogorov']:
        print(f"\n--- Metric: {metric_name} ---")
        order_fn = ORDER_METRICS[metric_name]

        dead_points, live_points, best_img = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            order_fn=order_fn,
            seed=42
        )

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        print(f"{'Threshold':<12} {'Bits (B)':<15} {'Status'}")
        print("-" * 40)

        metric_results = {}
        for t in thresholds:
            bits, reached = compute_bits_to_threshold(dead_points, t, n_live)
            metric_results[str(t)] = {'bits': bits, 'reached': reached}

            if reached:
                print(f"{t:<12.2f} {bits:<15.2f} {'Reached'}")
            else:
                print(f"{t:<12.2f} {'≥' + f'{bits:.1f}':<15} {'Not reached'}")

        results['thresholds'][metric_name] = metric_results

        # Best order
        best_order = max(get_order(d) for d in dead_points)
        results[f'best_order_{metric_name}'] = best_order
        print(f"Best {metric_name} order: {best_order:.4f}")

    # Calculate bit savings vs uniform baseline
    print()
    print("BIT SAVINGS (vs uniform baseline ≥72 bits at 32×32):")
    print("-" * 70)

    for metric_name in ['multiplicative', 'kolmogorov']:
        if '0.1' in results['thresholds'].get(metric_name, {}):
            cppn_bits = results['thresholds'][metric_name]['0.1']['bits']
            if results['thresholds'][metric_name]['0.1']['reached']:
                # Extrapolate: uniform at 64×64 should be even worse than 72 bits at 32×32
                uniform_bits_estimate = 72 * (64/32)**2  # Rough scaling
                delta_bits = uniform_bits_estimate - cppn_bits
                speedup = 2 ** delta_bits
                print(f"  {metric_name}: CPPN needs ~{cppn_bits:.1f} bits")
                print(f"    Estimated ΔB ≥ {delta_bits:.0f} bits → speedup ≥ 2^{delta_bits:.0f}")

    save_results(results, 'cppn_comparison_64x64.json')
    return results


# ============================================================================
# EXPERIMENT 3: SCALING ANALYSIS
# ============================================================================

def experiment_scaling_analysis(sizes=[8, 16, 32, 64], n_live=50, n_iterations=500):
    """
    Test how bits-to-threshold scales with image dimension.

    Hypothesis:
    - Uniform: Bits scale roughly linearly with image size (exponential rarity)
    - CPPN: Bits remain roughly constant (structure is dimension-independent)

    This validates the theoretical prediction that structured images become
    exponentially rarer as dimension increases for uniform priors, but not
    for structured priors like CPPN.
    """
    print("=" * 70)
    print("EXPERIMENT 3: SCALING ANALYSIS")
    print("=" * 70)
    print()
    print("Hypothesis: Bits scale ~linearly with image dimension for uniform,")
    print("           but stay ~constant for CPPN")
    print()

    results = {
        'experiment': 'scaling_analysis',
        'sizes': sizes,
        'n_live': n_live,
        'n_iterations': n_iterations,
        'timestamp': datetime.now().isoformat(),
        'cppn': {},
        'uniform': {},
    }

    threshold = 0.1

    print(f"Measuring bits to reach order > {threshold} at each scale:")
    print()

    # CPPN scaling
    print("CPPN Prior:")
    print(f"{'Size':<10} {'Bits (B)':<15} {'Status'}")
    print("-" * 40)

    for size in sizes:
        dead_points, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=size,
            order_fn=order_multiplicative,
            seed=42
        )

        bits, reached = compute_bits_to_threshold(dead_points, threshold, n_live)
        results['cppn'][str(size)] = {'bits': bits, 'reached': reached, 'n_pixels': size**2}

        status = "Reached" if reached else "Not reached"
        bits_str = f"{bits:.2f}" if reached else f"≥{bits:.1f}"
        print(f"{size}×{size:<6} {bits_str:<15} {status}")

    print()

    # Uniform scaling (run shorter since we expect failure)
    print("Uniform Prior:")
    print(f"{'Size':<10} {'Bits (B)':<15} {'Status'}")
    print("-" * 40)

    for size in sizes:
        # For uniform, use fewer iterations since we expect intractability
        iter_count = min(n_iterations, 200 * (32 // size + 1))

        dead_points, _, _ = nested_sampling_with_prior(
            prior_type='uniform',
            order_fn=order_multiplicative,
            n_live=n_live,
            n_iterations=iter_count,
            image_size=size,
            verbose=False
        )

        bits, reached = compute_bits_to_threshold(dead_points, threshold, n_live)
        results['uniform'][str(size)] = {'bits': bits, 'reached': reached, 'n_pixels': size**2}

        status = "Reached" if reached else "Not reached"
        bits_str = f"{bits:.2f}" if reached else f"≥{bits:.1f}"
        print(f"{size}×{size:<6} {bits_str:<15} {status}")

    print()
    print("INTERPRETATION:")
    print("-" * 70)

    # Analyze scaling
    cppn_bits = [results['cppn'][str(s)]['bits'] for s in sizes]
    uniform_bits = [results['uniform'][str(s)]['bits'] for s in sizes]

    cppn_range = max(cppn_bits) - min(cppn_bits)
    uniform_range = max(uniform_bits) - min(uniform_bits)

    print(f"  CPPN bits range: {min(cppn_bits):.1f} - {max(cppn_bits):.1f} (spread: {cppn_range:.1f})")
    print(f"  Uniform bits range: {min(uniform_bits):.1f} - {max(uniform_bits):.1f} (spread: {uniform_range:.1f})")

    if cppn_range < 5 and uniform_range > 20:
        print("  ✓ HYPOTHESIS SUPPORTED: CPPN constant, Uniform scales")
        results['hypothesis_supported'] = True
    else:
        print("  Results inconclusive or unexpected")
        results['hypothesis_supported'] = False

    save_results(results, 'scaling_analysis.json')
    return results


# ============================================================================
# EXPERIMENT 4: KOLMOGOROV VALIDATION
# ============================================================================

def experiment_kolmogorov_validation(n_samples=10000, image_size=64):
    """
    Validate Kolmogorov complexity theoretical bounds empirically.

    Theoretical bound: Fraction of images with K(x) < K is at most 2^(K-N)
    where N = image_size^2 bits.

    For 64×64 (N=4096):
    - 10% compression (K < 3686) → fraction < 2^(-410)
    - 5% compression (K < 3891) → fraction < 2^(-205)
    - 1% compression (K < 4055) → fraction < 2^(-41)

    We test by:
    1. Generating random images
    2. Measuring their compression ratios
    3. Comparing observed frequencies to theoretical bounds
    """
    print("=" * 70)
    print("EXPERIMENT 4: KOLMOGOROV COMPLEXITY VALIDATION")
    print("=" * 70)
    print()
    print(f"Testing theoretical bounds for {image_size}×{image_size} ({image_size**2} bits)")
    print()

    N = image_size ** 2  # Total bits

    results = {
        'experiment': 'kolmogorov_validation',
        'image_size': image_size,
        'n_bits': N,
        'n_samples': n_samples,
        'timestamp': datetime.now().isoformat(),
        'compressions': [],
    }

    print(f"Generating {n_samples} random images and measuring compression...")

    compressions = []
    for i in range(n_samples):
        img = np.random.randint(0, 2, (image_size, image_size), dtype=np.uint8)
        comp = order_kolmogorov_proxy(img)
        compressions.append(comp)

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    compressions = np.array(compressions)
    results['compression_stats'] = {
        'mean': float(np.mean(compressions)),
        'std': float(np.std(compressions)),
        'min': float(np.min(compressions)),
        'max': float(np.max(compressions)),
    }

    print()
    print(f"Compression statistics:")
    print(f"  Mean: {np.mean(compressions):.4f}")
    print(f"  Std:  {np.std(compressions):.4f}")
    print(f"  Range: [{np.min(compressions):.4f}, {np.max(compressions):.4f}]")
    print()

    # Check various compression thresholds
    print("THEORETICAL vs EMPIRICAL BOUNDS:")
    print("-" * 70)
    print(f"{'Compression':<15} {'Theoretical Max':<20} {'Observed':<15} {'Result'}")
    print("-" * 70)

    thresholds_to_test = [0.01, 0.02, 0.05, 0.10, 0.20]

    for comp_threshold in thresholds_to_test:
        # Theoretical: K < (1-comp)*N implies fraction < 2^((1-comp)*N - N) = 2^(-comp*N)
        theoretical_log2 = -comp_threshold * N
        theoretical_frac = 2 ** theoretical_log2

        # Empirical
        observed_count = np.sum(compressions >= comp_threshold)
        observed_frac = observed_count / n_samples

        results[f'threshold_{comp_threshold}'] = {
            'theoretical_log2_frac': theoretical_log2,
            'theoretical_frac': theoretical_frac,
            'observed_count': int(observed_count),
            'observed_frac': float(observed_frac),
        }

        # Format output
        if theoretical_frac < 1e-100:
            theory_str = f"<2^{theoretical_log2:.0f}"
        else:
            theory_str = f"{theoretical_frac:.2e}"

        observed_str = f"{observed_count}/{n_samples}" if observed_count > 0 else "0"

        if observed_count == 0:
            result_str = "✓ Consistent"
        elif observed_frac > theoretical_frac:
            result_str = "✗ Exceeds bound"
        else:
            result_str = "✓ Within bound"

        print(f"{comp_threshold*100:>10.0f}%     {theory_str:<20} {observed_str:<15} {result_str}")

    print()
    print("INTERPRETATION:")
    print("-" * 70)
    print("  The theoretical bounds from Kolmogorov complexity are:")
    print("  - Extremely conservative (upper bounds)")
    print("  - Our zlib proxy underestimates true compressibility")
    print("  - Random images should have ~0 compression")
    print()

    if np.mean(compressions) < 0.05:
        print("  ✓ Random images show near-zero compression as expected")
        results['hypothesis_supported'] = True
    else:
        print("  ✗ Unexpected compression in random images")
        results['hypothesis_supported'] = False

    save_results(results, 'kolmogorov_validation.json')
    return results


# ============================================================================
# EXPERIMENT 5: PHASE TRANSITION DETECTION
# ============================================================================

def experiment_phase_transition(n_live=100, n_iterations=1000, image_size=64):
    """
    Detect phase transition between order and disorder.

    Hypothesis: There exists a critical order value T_c where:
    - T < T_c: Images appear noise-like (disordered phase)
    - T > T_c: Images show clear structure (ordered phase)

    The transition should be sharp, with high variance (susceptibility)
    at the critical point.
    """
    print("=" * 70)
    print("EXPERIMENT 5: PHASE TRANSITION DETECTION")
    print("=" * 70)
    print()
    print("Looking for sharp transition between disorder and order...")
    print()

    results = {
        'experiment': 'phase_transition',
        'image_size': image_size,
        'n_live': n_live,
        'n_iterations': n_iterations,
        'timestamp': datetime.now().isoformat(),
    }

    # Run nested sampling and collect live point statistics
    dead_points, live_points, best_img = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Analyze trajectory for phase transition signatures
    orders = [get_order(d) for d in dead_points]
    log_X = [get_log_X(d) for d in dead_points]

    # Compute running statistics
    window_size = 20
    running_means = []
    running_vars = []

    for i in range(window_size, len(orders)):
        window = orders[i-window_size:i]
        running_means.append(np.mean(window))
        running_vars.append(np.var(window))

    # Find variance peak (susceptibility maximum)
    if running_vars:
        peak_idx = np.argmax(running_vars)
        peak_log_X = log_X[peak_idx + window_size] if peak_idx + window_size < len(log_X) else log_X[-1]
        peak_order = running_means[peak_idx] if peak_idx < len(running_means) else orders[-1]
        peak_variance = running_vars[peak_idx]

        results['variance_peak'] = {
            'log_X': float(peak_log_X),
            'order': float(peak_order),
            'variance': float(peak_variance),
        }

        print(f"VARIANCE PEAK (susceptibility maximum):")
        print(f"  Location: log(X) ≈ {peak_log_X:.2f}")
        print(f"  Order at peak: {peak_order:.4f}")
        print(f"  Variance: {peak_variance:.6f}")
        print()

    # Find steepest slope (transition region)
    if len(orders) > 10:
        slopes = np.diff(orders) / (np.diff(log_X) + 1e-10)
        steepest_idx = np.argmax(np.abs(slopes))
        transition_log_X = log_X[steepest_idx]
        transition_order = orders[steepest_idx]

        results['steepest_slope'] = {
            'log_X': float(transition_log_X),
            'order': float(transition_order),
            'slope': float(slopes[steepest_idx]),
        }

        print(f"STEEPEST SLOPE (transition region):")
        print(f"  Location: log(X) ≈ {transition_log_X:.2f}")
        print(f"  Order: {transition_order:.4f}")
        print(f"  Slope: {slopes[steepest_idx]:.4f}")
        print()

    # Order statistics
    final_order = orders[-1] if orders else 0
    results['final_order'] = float(final_order)
    results['order_trajectory'] = {
        'min': float(min(orders)) if orders else 0,
        'max': float(max(orders)) if orders else 0,
        'final': float(final_order),
    }

    print(f"ORDER TRAJECTORY:")
    print(f"  Initial: {orders[0]:.4f}" if orders else "  No data")
    print(f"  Final: {final_order:.4f}")
    print(f"  Range: [{min(orders):.4f}, {max(orders):.4f}]" if orders else "")
    print()

    print("INTERPRETATION:")
    print("-" * 70)
    if running_vars and max(running_vars) > 0.001:
        print("  ✓ Clear variance peak detected - suggests phase transition")
        print(f"    Critical region near order ≈ {peak_order:.3f}")
        results['phase_transition_detected'] = True
    else:
        print("  Smooth transition (no sharp phase boundary)")
        results['phase_transition_detected'] = False

    save_results(results, 'phase_transition.json')
    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_experiments():
    """Run all hypothesis testing experiments."""
    print("\n" + "=" * 70)
    print("EXPONENTIAL RARITY HYPOTHESIS: FULL TEST SUITE")
    print("=" * 70)
    print()
    print("Testing: Structured images are exponentially rare in 64×64 space")
    print()

    results = {}

    # Run experiments
    print("\n[1/5] Uniform Baseline...")
    results['uniform'] = experiment_uniform_baseline(n_iterations=500)  # Shorter for demo

    print("\n[2/5] CPPN Comparison...")
    results['cppn'] = experiment_cppn_comparison()

    print("\n[3/5] Scaling Analysis...")
    results['scaling'] = experiment_scaling_analysis(sizes=[16, 32, 64])  # Skip 8 for speed

    print("\n[4/5] Kolmogorov Validation...")
    results['kolmogorov'] = experiment_kolmogorov_validation(n_samples=5000)

    print("\n[5/5] Phase Transition...")
    results['phase'] = experiment_phase_transition(n_iterations=500)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    supported = sum([
        results.get('uniform', {}).get('hypothesis_supported', False),
        results.get('scaling', {}).get('hypothesis_supported', False),
        results.get('kolmogorov', {}).get('hypothesis_supported', False),
    ])

    print(f"\nHypothesis support: {supported}/3 key experiments")
    print()

    if supported >= 2:
        print("CONCLUSION: Strong evidence that structured images are")
        print("           exponentially rare under uniform sampling.")
    else:
        print("CONCLUSION: Results inconclusive or unexpected.")

    # Save combined results
    save_results(results, 'all_experiments.json')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp = sys.argv[1].lower()
        if exp == 'uniform':
            experiment_uniform_baseline()
        elif exp == 'cppn':
            experiment_cppn_comparison()
        elif exp == 'scaling':
            experiment_scaling_analysis()
        elif exp == 'kolmogorov':
            experiment_kolmogorov_validation()
        elif exp == 'phase':
            experiment_phase_transition()
        elif exp == 'all':
            run_all_experiments()
        else:
            print(f"Unknown experiment: {exp}")
            print("Options: uniform | cppn | scaling | kolmogorov | phase | all")
    else:
        # Default: run quick demo
        print("Running quick demo (use 'all' for full suite)")
        print()
        experiment_cppn_comparison(n_iterations=200)
