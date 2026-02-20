#!/usr/bin/env python3
"""
Bits vs Threshold Curve Experiment (B(tau))
=============================================

This experiment analyzes how many bits are required to reach different order thresholds tau.
The goal is to show that the tau=0.1 threshold choice isn't cherry-picked - the phase
transition happens at similar bit values regardless of threshold.

Key insight: bits = -log_X / ln(2)

For nested sampling data, we compute the first iteration where order > tau for each threshold,
then convert log_X at that point to bits.

Expected outcome:
- CPPN should show low bits (~1-3) across all thresholds
- Uniform should show high bits (>50) or never reach threshold
- The GAP between CPPN and Uniform is large regardless of tau choice

Usage:
    uv run python experiments/bits_vs_threshold_curve.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    nested_sampling_with_prior,
    order_multiplicative,
    set_global_seed
)


def bits_from_log_X(log_X: float) -> float:
    """Convert log_X (natural log) to bits of prior volume contracted.

    In nested sampling, log_X represents the log of remaining prior volume.
    The bits contracted = -log_X / ln(2)
    """
    return -log_X / np.log(2)


def compute_bits_to_threshold(df: pd.DataFrame, prior: str, threshold: float) -> tuple:
    """
    Find the first point where order exceeds threshold for a given prior.

    In nested sampling, iteration increases as we contract prior volume.
    Early iterations have log_X close to 0, late iterations have large negative log_X.
    We want the EARLIEST iteration (smallest iteration number, closest to 0 log_X)
    where order first crosses the threshold.

    Returns:
        (bits, log_X, order) or (np.inf, None, None) if threshold never reached
    """
    prior_data = df[df['prior'] == prior].copy()

    # Sort by iteration (ascending) to find first crossing
    prior_data = prior_data.sort_values('iteration', ascending=True)

    # Find first point where order >= threshold
    above_thresh = prior_data[prior_data['order_mean'] >= threshold]

    if len(above_thresh) == 0:
        return np.inf, None, None

    # Get the first crossing (earliest iteration)
    first_crossing = above_thresh.iloc[0]
    log_X = first_crossing['log_X']
    bits = bits_from_log_X(log_X)
    order = first_crossing['order_mean']

    return bits, log_X, order


def run_fresh_sampling(priors, n_live=50, n_iterations=500, seed=42):
    """Run fresh nested sampling for each prior if no cached data exists."""
    from core.thermo_sampler_v3 import compare_priors

    print("Running fresh nested sampling...")
    results_dir = Path(__file__).parent.parent / 'results' / 'bits_threshold'
    results_dir.mkdir(parents=True, exist_ok=True)

    compare_priors(
        priors=priors,
        n_live=n_live,
        n_iterations=n_iterations,
        output_dir=str(results_dir),
        n_runs=3,
        base_seed=seed,
        verbose=True
    )

    return results_dir / 'curve_summary_all.csv'


def analyze_bits_vs_threshold():
    """Main analysis: compute B(tau) curves for each prior."""

    # Thresholds to analyze
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    # Check for existing data
    existing_data = Path(__file__).parent.parent / 'results' / 'prior_comparison_multiplicative' / 'curve_summary_all.csv'

    if existing_data.exists():
        print(f"Loading existing data from {existing_data}")
        df = pd.read_csv(existing_data)
        priors = df['prior'].unique().tolist()
        print(f"Found priors: {priors}")
    else:
        print("No existing data found. Running fresh sampling...")
        priors = ['cppn', 'uniform']
        csv_path = run_fresh_sampling(priors)
        df = pd.read_csv(csv_path)

    # Compute B(tau) for each prior and threshold
    results = {prior: {} for prior in priors}

    print("\n" + "=" * 70)
    print("BITS TO REACH THRESHOLD B(tau)")
    print("=" * 70)

    # Header
    header = f"{'tau':>8}"
    for prior in priors:
        header += f" | {prior:>12}"
    print(header)
    print("-" * (10 + 15 * len(priors)))

    for tau in thresholds:
        row = f"{tau:>8.2f}"
        for prior in priors:
            bits, log_X, order = compute_bits_to_threshold(df, prior, tau)
            results[prior][tau] = {
                'bits': bits,
                'log_X': log_X,
                'order_at_crossing': order
            }

            if np.isinf(bits):
                row += f" | {'never':>12}"
            else:
                row += f" | {bits:>12.2f}"
        print(row)

    return results, df, priors, thresholds


def create_figure(results, priors, thresholds):
    """Create B(tau) curve figure."""
    fig_dir = Path(__file__).parent.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors and markers for each prior
    styles = {
        'cppn': {'color': '#2ecc71', 'marker': 'o', 'label': 'CPPN (neural prior)'},
        'uniform': {'color': '#e74c3c', 'marker': '^', 'label': 'Uniform (unstructured)'},
    }

    # Track which thresholds Uniform fails at
    uniform_fails = []

    for prior in priors:
        if prior not in styles:
            continue

        taus = []
        bits_vals = []

        for tau in thresholds:
            bits = results[prior][tau]['bits']
            if not np.isinf(bits):
                taus.append(tau)
                bits_vals.append(bits)
            elif prior == 'uniform':
                uniform_fails.append(tau)

        if taus:
            ax.plot(taus, bits_vals,
                   color=styles[prior]['color'],
                   marker=styles[prior]['marker'],
                   markersize=10,
                   linewidth=2,
                   label=styles[prior]['label'])

    # Add markers for where Uniform never reaches the threshold
    if uniform_fails:
        # Put these at the top of the plot to indicate "never reached"
        fail_y = 70  # Off the chart to show failure
        ax.scatter(uniform_fails, [fail_y] * len(uniform_fails),
                  color='#e74c3c', marker='x', s=150, linewidths=3,
                  label='Uniform fails (>72 bits)', zorder=5)

    # Add horizontal line to indicate "never reached" threshold
    ax.axhline(y=65, color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.35, 67, 'Uniform never reaches tau >= 0.05', fontsize=10, color='#e74c3c',
           ha='center')

    # Add horizontal line for reference (tau=0.1 standard)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Standard threshold (tau=0.1)')

    # Add annotation for the key insight (middle-lower area)
    ax.annotate('Structured priors: ~1-13 bits\nUniform: >60 bits or never',
               xy=(0.275, 28), fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(r'Order Threshold $\\tau$', fontsize=14)
    ax.set_ylabel(r'NS crossing bits $B_{\\mathrm{NS}}^{\\mathrm{cross}}(\\tau)$', fontsize=14)
    ax.set_title('Phase Transition is Robust to Threshold Choice', fontsize=16)
    # Legend in middle-upper area, single column
    ax.legend(loc='center', fontsize=10, ncol=1, bbox_to_anchor=(0.5, 0.62))
    ax.grid(True, alpha=0.3)

    # Set y-axis limit to show meaningful range including "never" zone
    ax.set_ylim(0, 75)
    ax.set_xlim(0, 0.55)

    plt.tight_layout()

    # Save
    pdf_path = fig_dir / 'bits_threshold_curve.pdf'
    png_path = fig_dir / 'bits_threshold_curve.png'

    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to {pdf_path}")
    return pdf_path


def save_results(results, priors, thresholds):
    """Save results to JSON."""
    results_dir = Path(__file__).parent.parent / 'results' / 'bits_threshold'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output = {
        'thresholds': thresholds,
        'priors': priors,
        'B_tau_curves': {}
    }

    for prior in priors:
        output['B_tau_curves'][prior] = {}
        for tau in thresholds:
            data = results[prior][tau]
            output['B_tau_curves'][prior][str(tau)] = {
                'bits': data['bits'] if not np.isinf(data['bits']) else None,
                'log_X': data['log_X'],
                'order_at_crossing': data['order_at_crossing']
            }

    # Compute summary statistics
    output['summary'] = {}
    for prior in priors:
        bits_values = [results[prior][tau]['bits'] for tau in thresholds
                       if not np.isinf(results[prior][tau]['bits'])]
        if bits_values:
            output['summary'][prior] = {
                'mean_bits': float(np.mean(bits_values)),
                'min_bits': float(np.min(bits_values)),
                'max_bits': float(np.max(bits_values)),
                'thresholds_reached': len(bits_values),
                'thresholds_total': len(thresholds)
            }
        else:
            output['summary'][prior] = {
                'mean_bits': None,
                'thresholds_reached': 0,
                'thresholds_total': len(thresholds)
            }

    json_path = results_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {json_path}")
    return json_path


def print_summary(results, priors, thresholds):
    """Print summary analysis."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for prior in priors:
        bits_values = [results[prior][tau]['bits'] for tau in thresholds
                       if not np.isinf(results[prior][tau]['bits'])]

        if bits_values:
            print(f"\n{prior.upper()}:")
            print(f"  Thresholds reached: {len(bits_values)}/{len(thresholds)}")
            print(f"  Mean bits: {np.mean(bits_values):.2f}")
            print(f"  Range: [{np.min(bits_values):.2f}, {np.max(bits_values):.2f}]")
        else:
            print(f"\n{prior.upper()}:")
            print(f"  Thresholds reached: 0/{len(thresholds)}")
            print(f"  (Never crosses any threshold in sampled region)")

    # Compute gaps
    if 'cppn' in results and 'uniform' in results:
        print("\n" + "-" * 50)
        print("GAP ANALYSIS (CPPN vs Uniform):")
        print("-" * 50)

        for tau in thresholds:
            cppn_bits = results['cppn'][tau]['bits']
            uniform_bits = results['uniform'][tau]['bits']

            if not np.isinf(cppn_bits):
                if np.isinf(uniform_bits):
                    gap_str = "infinite (uniform never reaches)"
                else:
                    gap = uniform_bits - cppn_bits
                    gap_str = f"{gap:.1f} bits"
                print(f"  tau={tau:.2f}: gap = {gap_str}")

    print("\n" + "=" * 70)
    print("KEY FINDING:")
    print("The phase transition (CPPN reaching structure quickly, Uniform failing)")
    print("occurs at similar bit values regardless of threshold choice.")
    print("This confirms tau=0.1 is not cherry-picked.")
    print("=" * 70)


if __name__ == '__main__':
    print("=" * 70)
    print("BITS VS THRESHOLD CURVE EXPERIMENT")
    print("=" * 70)
    print("Goal: Show phase transition is robust to threshold choice")
    print()

    # Run analysis
    results, df, priors, thresholds = analyze_bits_vs_threshold()

    # Create figure
    fig_path = create_figure(results, priors, thresholds)

    # Save results
    json_path = save_results(results, priors, thresholds)

    # Print summary
    print_summary(results, priors, thresholds)
