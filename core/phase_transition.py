"""
Phase Transition Detection for Thermodynamic Illumination

This module provides clean APIs for detecting and analyzing phase transitions
in the nested sampling trajectory, connecting to the hypothesis that there
exists a sharp boundary between structured and random-like images.

Key concepts:
- Variance peak (susceptibility): Maximum in order variance indicates critical point
- Steepest slope: Transition region where order changes most rapidly
- Entropy collapse: Diversity drop when entering ordered phase
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PhaseTransitionResult:
    """Results from phase transition detection."""

    # Variance peak (susceptibility maximum)
    variance_peak_log_X: float
    variance_peak_order: float
    variance_peak_value: float

    # Steepest slope (transition region)
    transition_log_X: float
    transition_order: float
    transition_slope: float

    # Overall trajectory statistics
    initial_order: float
    final_order: float
    order_range: tuple[float, float]

    # Phase transition detected?
    detected: bool
    sharpness: float  # Higher = sharper transition


def detect_phase_transition(
    dead_points: list,
    window_size: int = 20,
    variance_threshold: float = 0.001
) -> PhaseTransitionResult:
    """
    Detect phase transition in nested sampling trajectory.

    The hypothesis predicts a sharp boundary between:
    - Disordered phase: Images appear random-like (T < T_c)
    - Ordered phase: Images show clear structure (T > T_c)

    We detect this by looking for:
    1. Variance peak (susceptibility maximum) - critical point
    2. Steepest slope - transition region
    3. Entropy/diversity collapse

    Args:
        dead_points: List of dead points from nested sampling, each with
                     'order' and 'log_X' keys
        window_size: Rolling window size for computing statistics
        variance_threshold: Minimum variance to consider a "sharp" transition

    Returns:
        PhaseTransitionResult with transition characteristics
    """
    if not dead_points or len(dead_points) < window_size * 2:
        return PhaseTransitionResult(
            variance_peak_log_X=0, variance_peak_order=0, variance_peak_value=0,
            transition_log_X=0, transition_order=0, transition_slope=0,
            initial_order=0, final_order=0, order_range=(0, 0),
            detected=False, sharpness=0
        )

    orders = np.array([d['order'] for d in dead_points])
    log_X = np.array([d['log_X'] for d in dead_points])

    # Compute running statistics
    running_means = []
    running_vars = []
    running_log_X = []

    for i in range(window_size, len(orders)):
        window = orders[i-window_size:i]
        running_means.append(np.mean(window))
        running_vars.append(np.var(window))
        running_log_X.append(log_X[i])

    running_vars = np.array(running_vars)
    running_means = np.array(running_means)
    running_log_X = np.array(running_log_X)

    # Find variance peak (susceptibility maximum)
    peak_idx = np.argmax(running_vars)
    variance_peak_log_X = running_log_X[peak_idx]
    variance_peak_order = running_means[peak_idx]
    variance_peak_value = running_vars[peak_idx]

    # Find steepest slope (transition region)
    if len(orders) > 10:
        slopes = np.diff(orders) / (np.diff(log_X) + 1e-10)
        steepest_idx = np.argmax(np.abs(slopes))
        transition_log_X = log_X[steepest_idx]
        transition_order = orders[steepest_idx]
        transition_slope = slopes[steepest_idx]
    else:
        transition_log_X = variance_peak_log_X
        transition_order = variance_peak_order
        transition_slope = 0

    # Determine if transition is "sharp"
    detected = variance_peak_value > variance_threshold

    # Compute sharpness metric (normalized)
    # Higher variance relative to order range = sharper transition
    order_range_val = max(orders) - min(orders)
    if order_range_val > 0:
        sharpness = np.sqrt(variance_peak_value) / order_range_val
    else:
        sharpness = 0

    return PhaseTransitionResult(
        variance_peak_log_X=float(variance_peak_log_X),
        variance_peak_order=float(variance_peak_order),
        variance_peak_value=float(variance_peak_value),
        transition_log_X=float(transition_log_X),
        transition_order=float(transition_order),
        transition_slope=float(transition_slope),
        initial_order=float(orders[0]),
        final_order=float(orders[-1]),
        order_range=(float(min(orders)), float(max(orders))),
        detected=detected,
        sharpness=float(sharpness)
    )


def compute_rarity_curve(
    dead_points: list,
    n_live: int,
    thresholds: Optional[list] = None
) -> dict:
    """
    Compute the rarity curve: bits required to reach each threshold.

    B(T) = -log₂(p(T)) where p(T) = Pr[order ≥ T]

    This directly tests the exponential rarity hypothesis:
    - For uniform prior: B should scale with image dimension
    - For structured priors (CPPN): B should be small (~2-5 bits)

    Args:
        dead_points: List of dead points from nested sampling
        n_live: Number of live points used
        thresholds: Order thresholds to evaluate (default: [0.1, 0.2, 0.3, 0.4, 0.5])

    Returns:
        Dictionary mapping threshold → (bits, reached)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = {}

    for t in thresholds:
        bits = None
        reached = False

        for d in dead_points:
            if d['order'] >= t:
                bits = -d['log_X'] / np.log(2)
                reached = True
                break

        if bits is None:
            # Threshold not reached - return lower bound
            bits = len(dead_points) / (n_live * np.log(2))

        results[t] = {
            'bits': bits,
            'reached': reached,
            'samples_needed': 2**bits,
            'log2_samples': bits
        }

    return results


def compare_priors_rarity(
    prior_results: dict,
    threshold: float = 0.1
) -> dict:
    """
    Compare bit requirements across different priors.

    Args:
        prior_results: Dict mapping prior_name → rarity_curve results
        threshold: Threshold to compare at

    Returns:
        Comparison statistics including bit savings and speedup factors
    """
    comparison = {}

    for prior_name, rarity in prior_results.items():
        if threshold in rarity:
            comparison[prior_name] = rarity[threshold]

    # Find baseline (usually uniform)
    baseline_bits = None
    if 'uniform' in comparison:
        baseline_bits = comparison['uniform']['bits']
    else:
        # Use maximum as baseline
        baseline_bits = max(c['bits'] for c in comparison.values())

    # Compute bit savings and speedup for each prior
    for prior_name in comparison:
        bits = comparison[prior_name]['bits']
        delta_bits = baseline_bits - bits
        speedup = 2 ** delta_bits

        comparison[prior_name]['delta_bits'] = delta_bits
        comparison[prior_name]['speedup_factor'] = speedup
        comparison[prior_name]['speedup_log10'] = delta_bits * np.log10(2)

    return comparison


def theoretical_kolmogorov_bound(
    image_size: int,
    compression_fraction: float
) -> dict:
    """
    Compute theoretical Kolmogorov complexity bounds.

    For N-bit images, the fraction with K(x) < K is at most 2^(K-N).

    If compression_fraction = 1 - K/N, then:
    - K = (1 - compression_fraction) * N
    - Fraction < 2^(K-N) = 2^(-compression_fraction * N)

    Args:
        image_size: Image dimension (assumes square binary images)
        compression_fraction: Target compression (e.g., 0.1 for 10% compression)

    Returns:
        Theoretical bounds
    """
    N = image_size ** 2  # Total bits for binary image
    K = (1 - compression_fraction) * N  # Kolmogorov complexity

    log2_fraction = K - N  # = -compression_fraction * N
    fraction = 2 ** log2_fraction

    return {
        'n_bits': N,
        'kolmogorov_complexity': K,
        'compression_fraction': compression_fraction,
        'log2_fraction': log2_fraction,
        'fraction': fraction,
        'one_in_N': 1 / fraction if fraction > 0 else float('inf'),
        'bits_to_find': -log2_fraction
    }


def format_phase_report(result: PhaseTransitionResult) -> str:
    """Format phase transition result as human-readable report."""
    lines = [
        "=" * 60,
        "PHASE TRANSITION ANALYSIS",
        "=" * 60,
        "",
        "1. VARIANCE PEAK (Susceptibility Maximum)",
        "-" * 40,
        f"   Location: log(X) ≈ {result.variance_peak_log_X:.2f}",
        f"   Order at peak: {result.variance_peak_order:.4f}",
        f"   Variance: {result.variance_peak_value:.6f}",
        "",
        "2. STEEPEST SLOPE (Transition Region)",
        "-" * 40,
        f"   Location: log(X) ≈ {result.transition_log_X:.2f}",
        f"   Order: {result.transition_order:.4f}",
        f"   Slope: {result.transition_slope:.4f}",
        "",
        "3. ORDER TRAJECTORY",
        "-" * 40,
        f"   Initial: {result.initial_order:.4f}",
        f"   Final: {result.final_order:.4f}",
        f"   Range: [{result.order_range[0]:.4f}, {result.order_range[1]:.4f}]",
        "",
        "4. PHASE TRANSITION ASSESSMENT",
        "-" * 40,
    ]

    if result.detected:
        lines.extend([
            f"   ✓ Phase transition DETECTED",
            f"   Sharpness: {result.sharpness:.3f}",
            f"   Critical order ≈ {result.variance_peak_order:.3f}",
        ])
    else:
        lines.extend([
            f"   No sharp phase transition detected",
            f"   Transition appears smooth/continuous",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)
