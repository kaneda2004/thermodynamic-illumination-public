#!/usr/bin/env python3
"""
Critical Phenomena in Image Space

Hunting for universal critical exponents in the order/disorder phase transition.

In statistical physics, near a phase transition:
- Order parameter: m ~ (T_c - T)^β
- Susceptibility (variance): χ ~ |T - T_c|^(-γ)
- Correlation length: ξ ~ |T - T_c|^(-ν)

If we find these scaling laws in image space, it would connect
machine learning to statistical mechanics in a profound way.

Key questions:
1. Is there a sharp critical point T_c?
2. What are the critical exponents?
3. Are they UNIVERSAL across different metrics/scales?
"""

import sys
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    order_multiplicative,
    order_kolmogorov_proxy,
    ORDER_METRICS,
)


# ============================================================================
# CRITICAL EXPONENT MODELS
# ============================================================================

def power_law(x, a, b):
    """Simple power law: y = a * x^b"""
    return a * np.power(np.abs(x) + 1e-10, b)


def critical_scaling(t, t_c, beta, a):
    """Order parameter near critical point: m = a * (t_c - t)^beta for t < t_c"""
    delta = t_c - t
    return np.where(delta > 0, a * np.power(delta, beta), 0)


def susceptibility_scaling(t, t_c, gamma, a):
    """Susceptibility divergence: χ = a * |t - t_c|^(-gamma)"""
    delta = np.abs(t - t_c) + 1e-6
    return a * np.power(delta, -gamma)


# ============================================================================
# EXPERIMENT 1: HIGH-RESOLUTION PHASE MAPPING
# ============================================================================

def experiment_phase_mapping(n_live=200, n_iterations=2000, image_size=32):
    """
    High-resolution mapping of the phase transition.

    We need dense sampling to extract critical exponents accurately.
    """
    print("=" * 70)
    print("EXPERIMENT: HIGH-RESOLUTION PHASE MAPPING")
    print("=" * 70)
    print()
    print(f"Parameters: n_live={n_live}, n_iter={n_iterations}, size={image_size}")
    print("This will take a while...")
    print()

    # Run nested sampling with high resolution
    dead_points, live_points, best_img = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Extract trajectory data
    orders = np.array([d.order_value for d in dead_points])
    log_X = np.array([d.log_X for d in dead_points])

    # Compute running statistics with fine windows
    window_sizes = [10, 20, 50, 100]
    stats = {}

    for ws in window_sizes:
        means = []
        variances = []
        positions = []

        for i in range(ws, len(orders)):
            window = orders[i-ws:i]
            means.append(np.mean(window))
            variances.append(np.var(window))
            positions.append(log_X[i])

        stats[ws] = {
            'means': np.array(means),
            'variances': np.array(variances),
            'log_X': np.array(positions)
        }

    # Find critical point candidates
    ws = 20
    var_peak_idx = np.argmax(stats[ws]['variances'])
    critical_log_X = stats[ws]['log_X'][var_peak_idx]
    critical_order = stats[ws]['means'][var_peak_idx]
    peak_variance = stats[ws]['variances'][var_peak_idx]

    print(f"CRITICAL POINT CANDIDATE:")
    print(f"  log(X)_c ≈ {critical_log_X:.3f}")
    print(f"  Order at critical point: {critical_order:.4f}")
    print(f"  Peak variance (susceptibility): {peak_variance:.6f}")
    print()

    return {
        'orders': orders,
        'log_X': log_X,
        'stats': stats,
        'critical_log_X': critical_log_X,
        'critical_order': critical_order,
        'peak_variance': peak_variance,
    }


# ============================================================================
# EXPERIMENT 2: CRITICAL EXPONENT EXTRACTION
# ============================================================================

def experiment_extract_exponents(phase_data):
    """
    Extract critical exponents β and γ from the phase transition data.

    Near critical point T_c:
    - Order parameter: m ~ (T_c - T)^β  (β ≈ 0.5 for mean-field, 0.326 for 3D Ising)
    - Susceptibility: χ ~ |T - T_c|^(-γ) (γ ≈ 1.0 for mean-field, 1.237 for 3D Ising)
    """
    print("=" * 70)
    print("EXPERIMENT: CRITICAL EXPONENT EXTRACTION")
    print("=" * 70)
    print()

    orders = phase_data['orders']
    log_X = phase_data['log_X']
    critical_log_X = phase_data['critical_log_X']

    stats = phase_data['stats'][20]
    means = stats['means']
    variances = stats['variances']
    positions = stats['log_X']

    results = {}

    # --- Extract β (order parameter exponent) ---
    print("BETA (Order Parameter Exponent):")
    print("-" * 50)

    # Use region below critical point (ordered phase approaching transition)
    # t = -log_X (so larger -log_X = deeper into ordered phase)
    # We want: order ~ (t_c - t)^β where t < t_c

    mask_below = positions < critical_log_X - 0.5  # Below critical, with margin
    if np.sum(mask_below) > 10:
        t_below = positions[mask_below]
        m_below = means[mask_below]

        # Transform: log(m) = log(a) + β * log(t_c - t)
        delta_t = critical_log_X - t_below
        valid = (delta_t > 0.01) & (m_below > 0.001)

        if np.sum(valid) > 5:
            log_delta = np.log(delta_t[valid])
            log_m = np.log(m_below[valid])

            slope, intercept, r, p, se = linregress(log_delta, log_m)
            beta = slope
            results['beta'] = beta
            results['beta_r2'] = r**2
            results['beta_se'] = se

            print(f"  β = {beta:.3f} ± {se:.3f}")
            print(f"  R² = {r**2:.4f}")
            print()

            # Compare to known values
            print("  Reference exponents:")
            print(f"    Mean-field: β = 0.5")
            print(f"    2D Ising:   β = 0.125")
            print(f"    3D Ising:   β = 0.326")
            print()
    else:
        print("  Insufficient data below critical point")
        print()

    # --- Extract γ (susceptibility exponent) ---
    print("GAMMA (Susceptibility Exponent):")
    print("-" * 50)

    # Susceptibility (variance) should diverge as χ ~ |t - t_c|^(-γ)
    # Near the peak, fit power law

    # Use region around critical point
    margin = 2.0
    mask_near = np.abs(positions - critical_log_X) < margin
    mask_near &= np.abs(positions - critical_log_X) > 0.1  # Exclude exact critical

    if np.sum(mask_near) > 10:
        t_near = positions[mask_near]
        chi_near = variances[mask_near]

        delta_t = np.abs(t_near - critical_log_X)
        valid = (delta_t > 0.05) & (chi_near > 1e-8)

        if np.sum(valid) > 5:
            log_delta = np.log(delta_t[valid])
            log_chi = np.log(chi_near[valid])

            slope, intercept, r, p, se = linregress(log_delta, log_chi)
            gamma = -slope  # Negative because χ diverges (increases as delta decreases)
            results['gamma'] = gamma
            results['gamma_r2'] = r**2
            results['gamma_se'] = se

            print(f"  γ = {gamma:.3f} ± {se:.3f}")
            print(f"  R² = {r**2:.4f}")
            print()

            print("  Reference exponents:")
            print(f"    Mean-field: γ = 1.0")
            print(f"    2D Ising:   γ = 1.75")
            print(f"    3D Ising:   γ = 1.237")
            print()
    else:
        print("  Insufficient data near critical point")
        print()

    # --- Scaling relation check ---
    if 'beta' in results and 'gamma' in results:
        print("SCALING RELATION CHECK:")
        print("-" * 50)
        # Rushbrooke inequality: α + 2β + γ ≥ 2
        # For second-order transitions: α + 2β + γ = 2
        # Assuming α ≈ 0 (no specific heat divergence): 2β + γ ≈ 2

        scaling_sum = 2 * results['beta'] + results['gamma']
        print(f"  2β + γ = {scaling_sum:.3f}")
        print(f"  (Should be ≈ 2 for standard critical phenomena)")
        print()

        if 1.5 < scaling_sum < 2.5:
            print("  ✓ CONSISTENT with critical scaling!")
        else:
            print("  ✗ Deviates from standard scaling")

        results['scaling_sum'] = scaling_sum

    return results


# ============================================================================
# EXPERIMENT 3: UNIVERSALITY TEST
# ============================================================================

def experiment_universality(image_sizes=[16, 32, 64], n_live=100, n_iterations=1000):
    """
    Test if critical exponents are UNIVERSAL across:
    1. Different image sizes (finite-size scaling)
    2. Different order metrics

    Universality would be a WOW signal - it means image structure
    belongs to a well-defined universality class!
    """
    print("=" * 70)
    print("EXPERIMENT: UNIVERSALITY TEST")
    print("=" * 70)
    print()

    results = {'by_size': {}, 'by_metric': {}}

    # Test across image sizes
    print("FINITE-SIZE SCALING:")
    print("-" * 50)

    for size in image_sizes:
        print(f"\n  Testing {size}×{size}...")

        dead_points, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=size,
            order_fn=order_multiplicative,
            seed=42
        )

        orders = np.array([d.order_value for d in dead_points])
        log_X = np.array([d.log_X for d in dead_points])

        # Find variance peak
        ws = min(20, len(orders) // 10)
        variances = []
        for i in range(ws, len(orders)):
            variances.append(np.var(orders[i-ws:i]))

        if variances:
            peak_idx = np.argmax(variances)
            critical_log_X = log_X[peak_idx + ws]
            peak_var = variances[peak_idx]

            results['by_size'][size] = {
                'critical_log_X': float(critical_log_X),
                'peak_variance': float(peak_var),
                'n_pixels': size * size
            }

            print(f"    Critical log(X): {critical_log_X:.3f}")
            print(f"    Peak variance: {peak_var:.6f}")

    # Finite-size scaling: T_c(L) - T_c(∞) ~ L^(-1/ν)
    print("\n  Finite-size scaling analysis:")
    sizes = list(results['by_size'].keys())
    if len(sizes) >= 2:
        critical_points = [results['by_size'][s]['critical_log_X'] for s in sizes]
        log_L = np.log(sizes)

        # Look for power-law scaling
        slope, intercept, r, p, se = linregress(log_L, critical_points)
        print(f"    T_c shift exponent: {slope:.3f} (R²={r**2:.3f})")
        print(f"    (1/ν ≈ {-slope:.3f} if this is finite-size scaling)")

    # Test across metrics
    print("\n\nMETRIC UNIVERSALITY:")
    print("-" * 50)

    metrics_to_test = ['multiplicative', 'symmetry', 'spectral', 'ising']

    for metric_name in metrics_to_test:
        if metric_name not in ORDER_METRICS:
            continue

        print(f"\n  Testing metric: {metric_name}...")

        order_fn = ORDER_METRICS[metric_name]

        dead_points, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=32,
            order_fn=order_fn,
            seed=42
        )

        orders = np.array([d.order_value for d in dead_points])
        log_X = np.array([d.log_X for d in dead_points])

        ws = 20
        variances = []
        for i in range(ws, len(orders)):
            variances.append(np.var(orders[i-ws:i]))

        if variances:
            peak_idx = np.argmax(variances)
            critical_log_X = log_X[peak_idx + ws]
            peak_var = variances[peak_idx]

            results['by_metric'][metric_name] = {
                'critical_log_X': float(critical_log_X),
                'peak_variance': float(peak_var)
            }

            print(f"    Critical log(X): {critical_log_X:.3f}")
            print(f"    Peak variance: {peak_var:.6f}")

    # Check universality
    print("\n\nUNIVERSALITY ASSESSMENT:")
    print("-" * 50)

    if results['by_metric']:
        critical_points = [v['critical_log_X'] for v in results['by_metric'].values()]
        spread = max(critical_points) - min(critical_points)
        mean_cp = np.mean(critical_points)

        print(f"  Critical point spread across metrics: {spread:.3f}")
        print(f"  Mean critical point: {mean_cp:.3f}")

        if spread < 1.0:
            print("\n  ✓ POSSIBLE UNIVERSALITY: Critical points cluster!")
            print("    Different metrics may belong to same universality class.")
        else:
            print("\n  ✗ No clear universality: Critical points vary significantly.")
            print("    Each metric may define a different transition.")

    return results


# ============================================================================
# EXPERIMENT 4: ORDER PARAMETER COLLAPSE
# ============================================================================

def experiment_data_collapse(image_sizes=[16, 32, 64], n_live=100, n_iterations=1000):
    """
    Test for data collapse under scaling.

    Near critical point, different system sizes should collapse onto
    a universal curve when properly rescaled:

    m * L^(β/ν) = f((T - T_c) * L^(1/ν))

    This is the strongest test of critical behavior.
    """
    print("=" * 70)
    print("EXPERIMENT: DATA COLLAPSE (Scaling Collapse)")
    print("=" * 70)
    print()
    print("If curves collapse, we have genuine critical phenomena!")
    print()

    all_data = {}

    for size in image_sizes:
        print(f"Running {size}×{size}...")

        dead_points, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=size,
            order_fn=order_multiplicative,
            seed=42
        )

        orders = np.array([d.order_value for d in dead_points])
        log_X = np.array([d.log_X for d in dead_points])

        all_data[size] = {'orders': orders, 'log_X': log_X}

    # Try different scaling exponents
    print("\nTesting scaling collapse with different exponents...")
    print()

    # Estimate T_c from largest system
    largest = max(image_sizes)
    orders_L = all_data[largest]['orders']
    log_X_L = all_data[largest]['log_X']

    ws = 20
    variances = [np.var(orders_L[max(0,i-ws):i]) for i in range(ws, len(orders_L))]
    T_c = log_X_L[np.argmax(variances) + ws]

    print(f"Estimated T_c from L={largest}: {T_c:.3f}")
    print()

    # Try standard exponents
    test_exponents = [
        ('Mean-field', 0.5, 0.5),   # β=0.5, ν=0.5
        ('2D Ising', 0.125, 1.0),    # β=0.125, ν=1.0
        ('3D Ising', 0.326, 0.63),   # β=0.326, ν=0.63
        ('Percolation', 0.14, 0.88), # β≈0.14, ν≈0.88 (2D)
    ]

    print(f"{'Universality Class':<20} {'Collapse Quality':<20} {'β':<8} {'ν':<8}")
    print("-" * 60)

    best_collapse = None
    best_quality = float('inf')

    for name, beta, nu in test_exponents:
        # Compute rescaled data for each size
        rescaled_data = []

        for size in image_sizes:
            orders = all_data[size]['orders']
            log_X = all_data[size]['log_X']

            # Rescale: x = (T - T_c) * L^(1/ν), y = m * L^(β/ν)
            x_scaled = (log_X - T_c) * (size ** (1/nu))
            y_scaled = orders * (size ** (beta/nu))

            rescaled_data.append((x_scaled, y_scaled, size))

        # Measure collapse quality by variance in y at similar x values
        # (Lower variance = better collapse)
        x_all = np.concatenate([d[0] for d in rescaled_data])
        y_all = np.concatenate([d[1] for d in rescaled_data])

        # Bin and compute variance
        x_bins = np.linspace(x_all.min(), x_all.max(), 20)
        bin_vars = []
        for i in range(len(x_bins) - 1):
            mask = (x_all >= x_bins[i]) & (x_all < x_bins[i+1])
            if np.sum(mask) > 2:
                bin_vars.append(np.var(y_all[mask]))

        collapse_quality = np.mean(bin_vars) if bin_vars else float('inf')

        print(f"{name:<20} {collapse_quality:<20.6f} {beta:<8.3f} {nu:<8.3f}")

        if collapse_quality < best_quality:
            best_quality = collapse_quality
            best_collapse = (name, beta, nu)

    print()
    if best_collapse:
        print(f"BEST COLLAPSE: {best_collapse[0]}")
        print(f"  β = {best_collapse[1]}, ν = {best_collapse[2]}")
        print()

        if best_collapse[0] != 'Mean-field':
            print("  ✓ WOW: Non-mean-field exponents suggest genuine criticality!")
        else:
            print("  Mean-field behavior (expected for high-dimensional systems)")

    return {
        'T_c': T_c,
        'best_collapse': best_collapse,
        'best_quality': best_quality,
        'all_data': {k: {'orders': v['orders'].tolist()[:100],
                        'log_X': v['log_X'].tolist()[:100]}
                    for k, v in all_data.items()}
    }


# ============================================================================
# MAIN
# ============================================================================

def run_all_critical():
    """Run full critical phenomena investigation."""
    print("\n" + "=" * 70)
    print("CRITICAL PHENOMENA INVESTIGATION")
    print("Hunting for Universal Exponents in Image Space")
    print("=" * 70 + "\n")

    results = {}

    # 1. High-resolution phase mapping
    print("\n[1/4] High-Resolution Phase Mapping...")
    phase_data = experiment_phase_mapping(n_live=100, n_iterations=1000, image_size=32)
    results['phase_mapping'] = {
        'critical_log_X': phase_data['critical_log_X'],
        'critical_order': phase_data['critical_order'],
        'peak_variance': phase_data['peak_variance']
    }

    # 2. Extract critical exponents
    print("\n[2/4] Critical Exponent Extraction...")
    exponents = experiment_extract_exponents(phase_data)
    results['exponents'] = exponents

    # 3. Universality test
    print("\n[3/4] Universality Test...")
    universality = experiment_universality(n_live=50, n_iterations=500)
    results['universality'] = universality

    # 4. Data collapse
    print("\n[4/4] Data Collapse Test...")
    collapse = experiment_data_collapse(n_live=50, n_iterations=500)
    results['collapse'] = {
        'T_c': collapse['T_c'],
        'best_collapse': collapse['best_collapse'],
        'best_quality': collapse['best_quality']
    }

    # Summary
    print("\n" + "=" * 70)
    print("CRITICAL PHENOMENA SUMMARY")
    print("=" * 70)
    print()

    if 'beta' in exponents:
        print(f"Order parameter exponent β = {exponents['beta']:.3f}")
    if 'gamma' in exponents:
        print(f"Susceptibility exponent γ = {exponents['gamma']:.3f}")
    if 'scaling_sum' in exponents:
        print(f"Scaling relation 2β + γ = {exponents['scaling_sum']:.3f} (should be ≈2)")

    if collapse['best_collapse']:
        print(f"\nBest universality class: {collapse['best_collapse'][0]}")

    # WOW assessment
    print("\n" + "-" * 70)
    wow_signals = 0

    if exponents.get('beta_r2', 0) > 0.8:
        print("✓ Clean power-law scaling for order parameter")
        wow_signals += 1

    if exponents.get('gamma_r2', 0) > 0.8:
        print("✓ Clean divergence in susceptibility")
        wow_signals += 1

    if 1.5 < exponents.get('scaling_sum', 0) < 2.5:
        print("✓ Scaling relations satisfied")
        wow_signals += 1

    if collapse['best_collapse'] and collapse['best_collapse'][0] != 'Mean-field':
        print("✓ Non-trivial universality class!")
        wow_signals += 2

    print()
    if wow_signals >= 3:
        print("🎯 STRONG WOW SIGNAL: Genuine critical phenomena detected!")
    elif wow_signals >= 1:
        print("📊 Interesting signals, but more investigation needed")
    else:
        print("📉 No clear critical behavior (may need more data)")

    # Save results
    output_dir = Path("results/critical_phenomena")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "critical_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/critical_analysis.json")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp = sys.argv[1].lower()
        if exp == 'phase':
            experiment_phase_mapping()
        elif exp == 'universality':
            experiment_universality()
        elif exp == 'collapse':
            experiment_data_collapse()
        elif exp == 'all':
            run_all_critical()
        else:
            print(f"Unknown: {exp}. Options: phase | universality | collapse | all")
    else:
        run_all_critical()
