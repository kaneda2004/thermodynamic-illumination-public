"""
Finite-Size Scaling Experiment for Phase Transitions

Hypothesis: The critical point location (variance peak in log_X) shifts
systematically with image size according to finite-size scaling theory:
    T_c(L) - T_c(inf) ~ L^(-1/nu)

This extends RES-006's inconclusive phase transition results by testing
a specific, measurable prediction of criticality.

Null hypothesis: No systematic relationship between image size and
critical point location.

Generated via 9-turn deliberation.
Entry: RES-010
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    order_multiplicative,
)


def find_critical_point(orders: np.ndarray, log_X: np.ndarray, window_size: int = 20) -> tuple:
    """
    Find the critical point as the location of maximum variance.

    Returns: (critical_log_X, peak_variance, critical_order)
    """
    if len(orders) < window_size * 2:
        return (np.nan, np.nan, np.nan)

    variances = []
    means = []
    positions = []

    for i in range(window_size, len(orders)):
        window = orders[i-window_size:i]
        variances.append(np.var(window))
        means.append(np.mean(window))
        positions.append(log_X[i])

    if not variances:
        return (np.nan, np.nan, np.nan)

    peak_idx = np.argmax(variances)
    return (positions[peak_idx], variances[peak_idx], means[peak_idx])


def power_law_fit(L, a, nu_inv):
    """Power law: T_c(L) = T_c_inf + a * L^(-1/nu)"""
    return a * np.power(L, -nu_inv)


def run_finite_size_scaling(
    sizes: list = None,
    n_seeds: int = 5,
    n_live: int = 100,
    n_iterations: int = 1000,
    base_seed: int = 42
) -> dict:
    """
    Test finite-size scaling of the critical point.

    For each image size, run multiple nested sampling runs to estimate
    the critical point location with uncertainty.
    """
    if sizes is None:
        sizes = [16, 24, 32, 48]

    print("=" * 60)
    print("FINITE-SIZE SCALING EXPERIMENT")
    print("=" * 60)
    print(f"\nSizes: {sizes}")
    print(f"Seeds per size: {n_seeds}")
    print(f"n_live={n_live}, n_iterations={n_iterations}")
    print()

    results_by_size = {}

    for size in sizes:
        print(f"\n--- Testing size {size}x{size} ---")
        critical_points = []
        peak_variances = []

        for seed in range(n_seeds):
            actual_seed = base_seed + seed * 100 + size

            dead_points, _, _ = nested_sampling_v3(
                n_live=n_live,
                n_iterations=n_iterations,
                image_size=size,
                order_fn=order_multiplicative,
                seed=actual_seed
            )

            orders = np.array([d.order_value for d in dead_points])
            log_X = np.array([d.log_X for d in dead_points])

            # Adjust window size for smaller images
            ws = max(10, min(20, len(orders) // 20))

            crit_log_X, peak_var, crit_order = find_critical_point(orders, log_X, ws)

            if not np.isnan(crit_log_X):
                critical_points.append(crit_log_X)
                peak_variances.append(peak_var)

        if critical_points:
            mean_crit = np.mean(critical_points)
            se_crit = np.std(critical_points) / np.sqrt(len(critical_points))
            mean_var = np.mean(peak_variances)

            results_by_size[size] = {
                'critical_log_X_mean': float(mean_crit),
                'critical_log_X_se': float(se_crit),
                'critical_log_X_values': [float(x) for x in critical_points],
                'peak_variance_mean': float(mean_var),
                'n_valid': len(critical_points)
            }

            print(f"  T_c(L={size}) = {mean_crit:.3f} +/- {se_crit:.3f}")
            print(f"  Peak variance = {mean_var:.6f}")
        else:
            print(f"  WARNING: No valid critical points found")

    # --- Fit finite-size scaling ---
    print("\n" + "=" * 60)
    print("FINITE-SIZE SCALING ANALYSIS")
    print("=" * 60)

    valid_sizes = [s for s in sizes if s in results_by_size]
    if len(valid_sizes) < 3:
        print("ERROR: Need at least 3 valid sizes for scaling analysis")
        return {'status': 'failed', 'reason': 'insufficient_data'}

    L_values = np.array(valid_sizes)
    T_c_values = np.array([results_by_size[s]['critical_log_X_mean'] for s in valid_sizes])
    T_c_errors = np.array([results_by_size[s]['critical_log_X_se'] for s in valid_sizes])

    # Fit T_c(L) = T_c_inf + a * L^(-1/nu)
    # For linear regression on log scale: log(T_c(L) - T_c_inf) = log(a) + (-1/nu) * log(L)
    # But T_c_inf is unknown. Instead test: T_c vs L^(-x) for various x

    # First, simple linear regression on log-log
    log_L = np.log(L_values)

    # Test if T_c shifts systematically with L
    slope, intercept, r_value, p_value, se_slope = stats.linregress(log_L, T_c_values)

    print(f"\n1. Log-linear regression: T_c vs log(L)")
    print(f"   Slope = {slope:.4f} +/- {se_slope:.4f}")
    print(f"   R^2 = {r_value**2:.4f}")
    print(f"   p-value = {p_value:.6f}")

    # The slope should be negative if larger systems have lower (more negative) T_c
    # In finite-size scaling: T_c(L) approaches T_c(inf) from above as L -> inf
    # So we expect slope < 0

    # --- Bootstrap CI for slope ---
    n_bootstrap = 1000
    bootstrap_slopes = []

    for _ in range(n_bootstrap):
        # Resample across sizes with replacement
        indices = np.random.choice(len(L_values), len(L_values), replace=True)
        boot_log_L = log_L[indices]
        boot_T_c = T_c_values[indices]

        # Add noise based on standard errors
        boot_T_c = boot_T_c + np.random.randn(len(boot_T_c)) * T_c_errors[indices]

        try:
            boot_slope, _, _, _, _ = stats.linregress(boot_log_L, boot_T_c)
            bootstrap_slopes.append(boot_slope)
        except Exception:
            pass

    if len(bootstrap_slopes) > 100:
        ci_low, ci_high = np.percentile(bootstrap_slopes, [2.5, 97.5])
        bootstrap_se = np.std(bootstrap_slopes)
        print(f"\n2. Bootstrap CI (n={n_bootstrap})")
        print(f"   95% CI for slope: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"   Bootstrap SE: {bootstrap_se:.4f}")
    else:
        ci_low, ci_high = np.nan, np.nan
        bootstrap_se = np.nan

    # --- Effect size (Cohen's d) ---
    # Compare observed slope to null hypothesis (slope = 0)
    if bootstrap_se > 0:
        cohens_d = abs(slope) / bootstrap_se
        print(f"\n3. Effect size")
        print(f"   Cohen's d = {cohens_d:.3f}")
    else:
        cohens_d = np.nan

    # --- Try to extract 1/nu exponent ---
    # If T_c(L) = T_c_inf + a * L^(-1/nu), and we assume T_c_inf < min(T_c_values)
    # Try different T_c_inf values and fit

    print(f"\n4. Exponent extraction (1/nu)")

    best_nu_inv = None
    best_r2 = 0
    T_c_inf_estimates = []

    # Try fitting with the shift: T_c(L) - T_c_inf ~ L^(-1/nu)
    # Use the fact that largest L should be closest to T_c_inf
    T_c_largest = results_by_size[max(valid_sizes)]['critical_log_X_mean']

    # Fit log(T_c - T_c_inf) = log(a) - (1/nu) * log(L)
    # Try T_c_inf slightly below the largest-L value
    for offset in np.linspace(0.5, 3.0, 20):
        T_c_inf_try = T_c_largest - offset

        shifts = T_c_values - T_c_inf_try
        if np.all(shifts > 0):
            log_shifts = np.log(shifts)
            try:
                slope_nu, intercept_nu, r_nu, _, _ = stats.linregress(log_L, log_shifts)
                nu_inv = -slope_nu  # Because T_c - T_c_inf ~ L^(-1/nu)

                if r_nu**2 > best_r2 and 0 < nu_inv < 3:
                    best_r2 = r_nu**2
                    best_nu_inv = nu_inv
                    T_c_inf_estimates.append((T_c_inf_try, nu_inv, r_nu**2))
            except Exception:
                pass

    if best_nu_inv is not None:
        nu_estimate = 1 / best_nu_inv
        print(f"   Best fit: 1/nu = {best_nu_inv:.3f} (nu = {nu_estimate:.3f})")
        print(f"   Best R^2 = {best_r2:.4f}")

        # Reference exponents
        print(f"\n   Reference critical exponents:")
        print(f"     Mean-field: nu = 0.5 (1/nu = 2.0)")
        print(f"     2D Ising:   nu = 1.0 (1/nu = 1.0)")
        print(f"     3D Ising:   nu = 0.63 (1/nu = 1.59)")
    else:
        nu_estimate = np.nan
        print(f"   Could not extract stable 1/nu exponent")

    # --- Final assessment ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Success criteria
    sig_p = p_value < 0.01
    sig_d = cohens_d > 0.5 if not np.isnan(cohens_d) else False
    sig_ci = ci_high < 0 if not np.isnan(ci_high) else False  # CI excludes zero (negative slope)
    sig_r2 = r_value**2 > 0.8

    print(f"\n   p < 0.01: {sig_p} (p = {p_value:.6f})")
    print(f"   Cohen's d > 0.5: {sig_d} (d = {cohens_d:.3f})")
    print(f"   95% CI excludes zero: {sig_ci} (CI = [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"   R^2 > 0.8: {sig_r2} (R^2 = {r_value**2:.4f})")

    validated = sig_p and sig_d and sig_r2

    if validated:
        status = 'validated'
        print(f"\n   STATUS: VALIDATED")
        print(f"   Critical point shifts systematically with system size!")
    elif sig_r2 and (sig_p or sig_d):
        status = 'inconclusive'
        print(f"\n   STATUS: INCONCLUSIVE")
        print(f"   Strong trend but statistical criteria not fully met")
    else:
        status = 'refuted'
        print(f"\n   STATUS: REFUTED")
        print(f"   No clear finite-size scaling relationship")

    # Compile results
    results = {
        'experiment': 'finite_size_scaling',
        'hypothesis': 'Critical point location shifts with image size following finite-size scaling',
        'null_hypothesis': 'No systematic relationship between size and critical point',
        'status': status,
        'parameters': {
            'sizes': sizes,
            'n_seeds': n_seeds,
            'n_live': n_live,
            'n_iterations': n_iterations
        },
        'by_size': results_by_size,
        'scaling_analysis': {
            'slope': float(slope),
            'slope_se': float(se_slope),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'bootstrap_ci_95': [float(ci_low), float(ci_high)] if not np.isnan(ci_low) else None,
            'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else None,
            'nu_estimate': float(nu_estimate) if not np.isnan(nu_estimate) else None,
            'nu_inv_estimate': float(best_nu_inv) if best_nu_inv else None,
            'nu_fit_r2': float(best_r2) if best_r2 > 0 else None
        },
        'success_criteria': {
            'p_lt_001': bool(sig_p),
            'cohens_d_gt_05': bool(sig_d),
            'ci_excludes_zero': bool(sig_ci),
            'r2_gt_08': bool(sig_r2)
        }
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'finite_size_scaling'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to: {results_dir / 'scaling_results.json'}")

    return results


if __name__ == "__main__":
    results = run_finite_size_scaling(
        sizes=[16, 24, 32, 48],
        n_seeds=5,
        n_live=100,
        n_iterations=1000,
        base_seed=42
    )

    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    print(json.dumps(results['scaling_analysis'], indent=2))
