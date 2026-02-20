#!/usr/bin/env python3
"""
Visualization Suite for Exponential Rarity Hypothesis

Generates figures demonstrating:
1. Bits vs Threshold curves (log-log scale)
2. Scaling analysis plots
3. Phase transition diagrams
4. Image galleries at different order levels
5. Kolmogorov complexity scatter plots

Usage:
    python figures/generate_hypothesis_figures.py [figure_name]

    figure_name: bits_curve | scaling | phase | gallery | kolmogorov | all
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    nested_sampling_with_prior,
    order_multiplicative,
    order_kolmogorov_proxy,
    CPPN,
)
from core.phase_transition import (
    detect_phase_transition,
    theoretical_kolmogorov_bound,
    format_phase_report,
)


# ============================================================================
# ASCII VISUALIZATION (Terminal-friendly)
# ============================================================================

def ascii_bar_chart(data: dict, title: str, width: int = 50):
    """Create ASCII horizontal bar chart."""
    lines = [
        "=" * (width + 20),
        title.center(width + 20),
        "=" * (width + 20),
        ""
    ]

    max_val = max(abs(v) for v in data.values()) if data else 1

    for label, value in data.items():
        bar_len = int(width * abs(value) / max_val)
        bar = "█" * bar_len
        lines.append(f"{label:>12} | {bar} {value:.2f}")

    lines.append("")
    return "\n".join(lines)


def ascii_line_plot(x_data: list, y_data: list, title: str,
                    x_label: str = "X", y_label: str = "Y",
                    width: int = 60, height: int = 20):
    """Create ASCII line plot."""
    if not x_data or not y_data:
        return "No data to plot"

    x = np.array(x_data)
    y = np.array(y_data)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for xi, yi in zip(x, y):
        col = int((xi - x_min) / (x_max - x_min + 1e-10) * (width - 1))
        row = int((1 - (yi - y_min) / (y_max - y_min + 1e-10)) * (height - 1))
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        grid[row][col] = '█'

    # Build output
    lines = [
        "=" * (width + 15),
        title.center(width + 15),
        "=" * (width + 15),
        f"{y_label}",
        f"  {y_max:.2f} ┬" + "─" * width,
    ]

    for row in grid:
        lines.append("        │" + "".join(row))

    lines.extend([
        f"  {y_min:.2f} ┴" + "─" * width,
        " " * 9 + f"{x_min:.2f}" + " " * (width - 10) + f"{x_max:.2f}",
        " " * ((width + 15) // 2 - len(x_label) // 2) + x_label,
        ""
    ])

    return "\n".join(lines)


def ascii_image(img: np.ndarray, width: int = 32) -> str:
    """Convert binary image to ASCII art."""
    chars = " ░▒▓█"
    h, w = img.shape

    # Downsample if needed
    if w > width:
        scale = width / w
        new_h = int(h * scale)
        new_w = width

        # Simple downsampling
        result = []
        for i in range(new_h):
            row = ""
            for j in range(new_w):
                # Average over the corresponding region
                y0 = int(i / scale)
                y1 = min(h, int((i + 1) / scale))
                x0 = int(j / scale)
                x1 = min(w, int((j + 1) / scale))

                if y1 > y0 and x1 > x0:
                    avg = np.mean(img[y0:y1, x0:x1])
                else:
                    avg = img[y0, x0]

                idx = int(avg * (len(chars) - 1))
                row += chars[idx]
            result.append(row)
        return "\n".join(result)
    else:
        result = []
        for row in img:
            line = ""
            for val in row:
                idx = int(val * (len(chars) - 1))
                line += chars[idx]
            result.append(line)
        return "\n".join(result)


# ============================================================================
# FIGURE 1: BITS VS THRESHOLD CURVE
# ============================================================================

def figure_bits_curve(n_live=50, n_iterations=500, image_size=32):
    """
    Generate bits vs threshold curves for different priors.

    This is the core visualization showing exponential rarity:
    - Uniform: Steep curve (many bits needed)
    - CPPN: Flat curve (few bits needed)
    """
    print("=" * 70)
    print("FIGURE 1: BITS VS THRESHOLD CURVE")
    print("=" * 70)
    print()
    print("Generating rarity curves for CPPN and Uniform priors...")
    print(f"Image size: {image_size}×{image_size}")
    print()

    results = {}

    # Run CPPN
    print("Running CPPN...")
    dead_cppn, _, _ = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Extract curve data
    cppn_orders = [d['order'] for d in dead_cppn]
    cppn_bits = [-d['log_X'] / np.log(2) for d in dead_cppn]
    results['cppn'] = {'orders': cppn_orders, 'bits': cppn_bits}

    # Run Uniform (shorter since we expect failure)
    print("Running Uniform...")
    dead_uniform, _, _ = nested_sampling_with_prior(
        prior_type='uniform',
        order_fn=order_multiplicative,
        n_live=n_live,
        n_iterations=min(200, n_iterations),
        image_size=image_size,
        verbose=False
    )

    uniform_orders = [d['order'] for d in dead_uniform]
    uniform_bits = [-d['log_X'] / np.log(2) for d in dead_uniform]
    results['uniform'] = {'orders': uniform_orders, 'bits': uniform_bits}

    # Print ASCII plot
    print()
    print("CPPN Rarity Curve (Bits vs Order):")
    print(ascii_line_plot(
        cppn_orders, cppn_bits,
        "CPPN: Bits Required vs Order Threshold",
        x_label="Order", y_label="Bits"
    ))

    print()
    print("Uniform Rarity Curve (Bits vs Order):")
    print(ascii_line_plot(
        uniform_orders, uniform_bits,
        "Uniform: Bits Required vs Order Threshold",
        x_label="Order", y_label="Bits"
    ))

    # Summary comparison
    print()
    print("COMPARISON AT KEY THRESHOLDS:")
    print("-" * 50)
    print(f"{'Threshold':<12} {'CPPN Bits':<15} {'Uniform Bits':<15}")
    print("-" * 50)

    for t in [0.01, 0.05, 0.1]:
        cppn_bit = next((b for o, b in zip(cppn_orders, cppn_bits) if o >= t), None)
        uniform_bit = next((b for o, b in zip(uniform_orders, uniform_bits) if o >= t), None)

        cppn_str = f"{cppn_bit:.2f}" if cppn_bit else "Not reached"
        uniform_str = f"{uniform_bit:.2f}" if uniform_bit else f"≥{max(uniform_bits):.1f}"

        print(f"{t:<12.2f} {cppn_str:<15} {uniform_str:<15}")

    return results


# ============================================================================
# FIGURE 2: SCALING ANALYSIS
# ============================================================================

def figure_scaling(sizes=[8, 16, 32], n_live=50, n_iterations=300):
    """
    Show how bits scale with image dimension.

    Hypothesis:
    - Uniform: Bits increase with image size (exponential rarity)
    - CPPN: Bits stay roughly constant (dimension-independent structure)
    """
    print("=" * 70)
    print("FIGURE 2: SCALING ANALYSIS")
    print("=" * 70)
    print()

    threshold = 0.1
    results = {'cppn': {}, 'uniform': {}}

    print(f"Measuring bits to reach order > {threshold} at each scale:")
    print()

    for size in sizes:
        print(f"  Testing {size}×{size}...")

        # CPPN
        dead, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=size,
            order_fn=order_multiplicative,
            seed=42
        )
        for d in dead:
            if d['order'] >= threshold:
                results['cppn'][size] = -d['log_X'] / np.log(2)
                break
        else:
            results['cppn'][size] = len(dead) / (n_live * np.log(2))

        # Uniform (short run)
        dead, _, _ = nested_sampling_with_prior(
            prior_type='uniform',
            order_fn=order_multiplicative,
            n_live=n_live,
            n_iterations=100,
            image_size=size,
            verbose=False
        )
        for d in dead:
            if d['order'] >= threshold:
                results['uniform'][size] = -d['log_X'] / np.log(2)
                break
        else:
            results['uniform'][size] = len(dead) / (n_live * np.log(2))

    # Print results
    print()
    print("SCALING RESULTS:")
    print("-" * 50)
    print(f"{'Size':<12} {'CPPN Bits':<15} {'Uniform Bits':<15}")
    print("-" * 50)

    for size in sizes:
        cppn_bits = results['cppn'].get(size, 0)
        uniform_bits = results['uniform'].get(size, 0)
        print(f"{size}×{size:<8} {cppn_bits:<15.2f} ≥{uniform_bits:<14.2f}")

    # ASCII bar chart
    print()
    cppn_data = {f"{s}×{s}": results['cppn'].get(s, 0) for s in sizes}
    print(ascii_bar_chart(cppn_data, "CPPN: Bits vs Image Size"))

    return results


# ============================================================================
# FIGURE 3: PHASE TRANSITION DIAGRAM
# ============================================================================

def figure_phase_transition(n_live=100, n_iterations=500, image_size=32):
    """
    Visualize the phase transition between disorder and order.
    """
    print("=" * 70)
    print("FIGURE 3: PHASE TRANSITION DIAGRAM")
    print("=" * 70)
    print()

    # Run nested sampling
    dead, _, _ = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Detect phase transition
    result = detect_phase_transition(dead)

    # Print report
    print(format_phase_report(result))

    # Plot order vs log_X
    orders = [d['order'] for d in dead]
    log_X = [d['log_X'] for d in dead]

    print()
    print("ORDER vs LOG(X) TRAJECTORY:")
    print(ascii_line_plot(
        log_X, orders,
        "Nested Sampling Trajectory",
        x_label="log(X) [Prior Volume]",
        y_label="Order"
    ))

    return result


# ============================================================================
# FIGURE 4: IMAGE GALLERY
# ============================================================================

def figure_image_gallery(n_live=50, n_iterations=300, image_size=32):
    """
    Show images at different order levels to visualize noise→structure transition.
    """
    print("=" * 70)
    print("FIGURE 4: IMAGE GALLERY (Noise → Structure)")
    print("=" * 70)
    print()

    # Run nested sampling and collect samples at different thresholds
    dead, live, best = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=42
    )

    # Find images at different order levels
    targets = [0.01, 0.05, 0.1, 0.15]
    gallery = {}

    for target in targets:
        for d in dead:
            if d['order'] >= target and target not in gallery:
                gallery[target] = d
                break

    # Display gallery
    for target in targets:
        if target in gallery:
            d = gallery[target]
            print(f"ORDER ≈ {d['order']:.3f} (log_X ≈ {d['log_X']:.1f}):")
            print("-" * 40)
            if 'image' in d:
                print(ascii_image(d['image'], width=32))
            else:
                print("  [Image not stored in dead point]")
            print()

    # Show best image
    print(f"BEST IMAGE (Order = {max(d['order'] for d in dead):.3f}):")
    print("-" * 40)
    if best is not None:
        print(ascii_image(best, width=32))
    print()

    return gallery


# ============================================================================
# FIGURE 5: KOLMOGOROV COMPLEXITY SCATTER
# ============================================================================

def figure_kolmogorov_scatter(n_samples=1000, image_size=32):
    """
    Scatter plot of compression ratio vs order metric.

    Shows correlation between compressibility and structure.
    """
    print("=" * 70)
    print("FIGURE 5: KOLMOGOROV COMPLEXITY vs ORDER")
    print("=" * 70)
    print()

    # Generate CPPN samples
    compressions = []
    orders = []

    print(f"Generating {n_samples} CPPN samples...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)

        comp = order_kolmogorov_proxy(img)
        order = order_multiplicative(img)

        compressions.append(comp)
        orders.append(order)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples}")

    compressions = np.array(compressions)
    orders = np.array(orders)

    # Statistics
    correlation = np.corrcoef(compressions, orders)[0, 1]

    print()
    print("STATISTICS:")
    print("-" * 50)
    print(f"Correlation (compression vs order): {correlation:.3f}")
    print(f"Mean compression: {np.mean(compressions):.3f}")
    print(f"Mean order: {np.mean(orders):.3f}")
    print()

    # ASCII scatter plot
    print("SCATTER PLOT:")
    print(ascii_line_plot(
        compressions.tolist(), orders.tolist(),
        f"Compression vs Order (r={correlation:.2f})",
        x_label="Compression (Kolmogorov proxy)",
        y_label="Order (Multiplicative)"
    ))

    # Theoretical bounds
    print()
    print("THEORETICAL KOLMOGOROV BOUNDS:")
    print("-" * 50)

    for comp_frac in [0.1, 0.2, 0.5]:
        bound = theoretical_kolmogorov_bound(image_size, comp_frac)
        print(f"  {comp_frac*100:.0f}% compression: 1 in 2^{-bound['log2_fraction']:.0f} images")

    return {'compressions': compressions.tolist(), 'orders': orders.tolist(), 'correlation': correlation}


# ============================================================================
# MAIN
# ============================================================================

def generate_all_figures():
    """Generate all hypothesis figures."""
    print("\n" + "=" * 70)
    print("GENERATING ALL HYPOTHESIS FIGURES")
    print("=" * 70 + "\n")

    results = {}

    print("\n[1/5] Bits vs Threshold Curve...")
    results['bits_curve'] = figure_bits_curve()

    print("\n[2/5] Scaling Analysis...")
    results['scaling'] = figure_scaling()

    print("\n[3/5] Phase Transition Diagram...")
    results['phase'] = figure_phase_transition()

    print("\n[4/5] Image Gallery...")
    results['gallery'] = figure_image_gallery()

    print("\n[5/5] Kolmogorov Scatter...")
    results['kolmogorov'] = figure_kolmogorov_scatter(n_samples=500)

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print("=" * 70)

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fig = sys.argv[1].lower()
        if fig == 'bits_curve':
            figure_bits_curve()
        elif fig == 'scaling':
            figure_scaling()
        elif fig == 'phase':
            figure_phase_transition()
        elif fig == 'gallery':
            figure_image_gallery()
        elif fig == 'kolmogorov':
            figure_kolmogorov_scatter()
        elif fig == 'all':
            generate_all_figures()
        else:
            print(f"Unknown figure: {fig}")
            print("Options: bits_curve | scaling | phase | gallery | kolmogorov | all")
    else:
        # Default: quick demo
        print("Running quick visualization demo...")
        print()
        figure_image_gallery(n_iterations=200)
