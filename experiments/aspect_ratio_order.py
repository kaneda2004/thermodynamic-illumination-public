"""
RES-058: Aspect Ratio Effect on Achievable Order

Hypothesis: Elongated images (e.g., 64x16) achieve lower order than square images
(32x32) for fixed pixel count due to reduced spatial coherence opportunities.

FINDING: Direction of effect is inconsistent - wide > square > tall, suggesting
CPPN coordinate asymmetry rather than pure aspect ratio effect.

Method:
- Generate CPPNs and render to multiple aspect ratios with same pixel count (1024)
- Measure order using multiplicative metric
- Compare distributions statistically
- Test whether effect is due to aspect ratio or CPPN x/y asymmetry
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, compute_spectral_coherence
from scipy import stats


def render_cppn_aspect(cppn, height, width):
    """Render CPPN to arbitrary aspect ratio."""
    coords_y = np.linspace(-1, 1, height)
    coords_x = np.linspace(-1, 1, width)
    x, y = np.meshgrid(coords_x, coords_y)
    return (cppn.activate(x, y) > 0.5).astype(np.uint8)


def run_experiment(n_samples=500, seed=42):
    """Test aspect ratio effect on order."""
    np.random.seed(seed)

    # Aspect ratios: all with 1024 pixels
    # Format: (height, width, name)
    aspect_configs = [
        (32, 32, "1:1"),      # Square
        (16, 64, "1:4"),      # Wide
        (64, 16, "4:1"),      # Tall
        (8, 128, "1:16"),     # Very wide
        (128, 8, "16:1"),     # Very tall
    ]

    results = {name: [] for _, _, name in aspect_configs}
    coherence_results = {name: [] for _, _, name in aspect_configs}

    print(f"Generating {n_samples} CPPNs across {len(aspect_configs)} aspect ratios...")
    print("="*60)

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"Progress: {i}/{n_samples}")

        # Generate one CPPN
        cppn = CPPN()

        # Render to all aspect ratios
        for h, w, name in aspect_configs:
            img = render_cppn_aspect(cppn, h, w)
            order = order_multiplicative(img)
            coherence = compute_spectral_coherence(img)
            results[name].append(order)
            coherence_results[name].append(coherence)

    print("\n" + "="*60)
    print("RESULTS: Order by Aspect Ratio")
    print("="*60)

    # Summary statistics
    for h, w, name in aspect_configs:
        orders = results[name]
        coh = coherence_results[name]
        print(f"\n{name} ({h}x{w}):")
        print(f"  Order: mean={np.mean(orders):.4f}, std={np.std(orders):.4f}")
        print(f"  Coherence: mean={np.mean(coh):.4f}, std={np.std(coh):.4f}")
        print(f"  Order > 0.1: {np.mean(np.array(orders) > 0.1)*100:.1f}%")

    # Statistical comparison: square vs elongated
    print("\n" + "="*60)
    print("STATISTICAL TESTS (vs Square 1:1)")
    print("="*60)

    square = np.array(results["1:1"])
    p_values = []
    effect_sizes = []
    test_names = []

    for h, w, name in aspect_configs:
        if name == "1:1":
            continue
        elongated = np.array(results[name])

        # Paired t-test (same CPPN, different renders)
        t_stat, p_val = stats.ttest_rel(square, elongated)

        # Effect size (Cohen's d for paired samples)
        diff = square - elongated
        d = np.mean(diff) / np.std(diff)

        print(f"\n{name} vs 1:1:")
        print(f"  Mean diff: {np.mean(diff):.4f}")
        print(f"  t-stat: {t_stat:.2f}")
        print(f"  p-value: {p_val:.2e}")
        print(f"  Cohen's d: {d:.3f}")

        p_values.append(p_val)
        effect_sizes.append(d)
        test_names.append(name)

    # Bonferroni correction
    n_tests = len(p_values)
    bonferroni_alpha = 0.01 / n_tests
    print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")

    # Test for asymmetry: wide vs tall
    print("\n" + "="*60)
    print("ASYMMETRY TEST: Wide vs Tall")
    print("="*60)

    wide_1_4 = np.array(results["1:4"])
    tall_4_1 = np.array(results["4:1"])
    wide_1_16 = np.array(results["1:16"])
    tall_16_1 = np.array(results["16:1"])

    # Wide vs Tall at 1:4
    t_asym, p_asym = stats.ttest_rel(wide_1_4, tall_4_1)
    d_asym = np.mean(wide_1_4 - tall_4_1) / np.std(wide_1_4 - tall_4_1)
    print(f"\n1:4 (wide) vs 4:1 (tall):")
    print(f"  Mean diff: {np.mean(wide_1_4 - tall_4_1):.4f}")
    print(f"  t-stat: {t_asym:.2f}")
    print(f"  p-value: {p_asym:.2e}")
    print(f"  Cohen's d: {d_asym:.3f}")

    # Wide vs Tall at 1:16
    t_asym2, p_asym2 = stats.ttest_rel(wide_1_16, tall_16_1)
    d_asym2 = np.mean(wide_1_16 - tall_16_1) / np.std(wide_1_16 - tall_16_1)
    print(f"\n1:16 (wide) vs 16:1 (tall):")
    print(f"  Mean diff: {np.mean(wide_1_16 - tall_16_1):.4f}")
    print(f"  t-stat: {t_asym2:.2f}")
    print(f"  p-value: {p_asym2:.2e}")
    print(f"  Cohen's d: {d_asym2:.3f}")

    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Key findings:
    # 1. Wide images (1:4, 1:16) have HIGHER order than square
    # 2. Tall images (4:1, 16:1) have LOWER order than square
    # 3. This suggests CPPN x/y asymmetry, not pure aspect ratio effect

    # The original hypothesis (elongation reduces order) is REFUTED
    # because the direction depends on orientation (wide vs tall)

    # Check if asymmetry is significant
    asymmetry_significant = p_asym < bonferroni_alpha and p_asym2 < bonferroni_alpha

    if asymmetry_significant:
        status = "REFUTED"
        summary = ("Original hypothesis refuted: aspect ratio effect depends on orientation. "
                   f"Wide images achieve {np.mean(wide_1_4 - tall_4_1):.3f} higher order than "
                   "equivalent tall images, suggesting CPPN coordinate asymmetry.")
    else:
        status = "INCONCLUSIVE"
        summary = "Mixed results - no clear relationship between aspect ratio and order."

    print(f"Status: {status}")
    print(f"Summary: {summary}")

    # Return metrics for log
    max_effect = max(abs(e) for e in effect_sizes)
    return {
        'status': status,
        'summary': summary,
        'effect_size': float(max_effect),
        'p_value': float(min(p_values)),
        'asymmetry_d': float(d_asym),
        'asymmetry_p': float(p_asym),
        'mean_order_square': float(np.mean(square)),
        'mean_order_wide_1_4': float(np.mean(wide_1_4)),
        'mean_order_tall_4_1': float(np.mean(tall_4_1)),
    }


if __name__ == "__main__":
    results = run_experiment(n_samples=500, seed=42)

    print("\n" + "="*60)
    print("METRICS FOR LOG")
    print("="*60)
    for k, v in results.items():
        print(f"{k}: {v}")
