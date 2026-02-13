"""
RES-119: Generator Sampling Efficiency Comparison

Compare how efficiently different generators produce high-order images.
CPPN vs Perlin noise vs sinusoidal vs Gabor wavelets.

Metric: Fraction of random samples exceeding order threshold T.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import order_multiplicative, CPPN


def generate_cppn_image(size=64, seed=None):
    """Generate CPPN image with random weights using proper CPPN class."""
    if seed is not None:
        np.random.seed(seed)

    # Create a fresh CPPN with random weights from the prior
    cppn = CPPN()

    # Render to binary image
    img = cppn.render(size=size)
    return img.astype(float)


def generate_perlin_image(size=64, seed=None):
    """Generate Perlin-like noise image."""
    if seed is not None:
        np.random.seed(seed)

    # Multi-octave noise (simplified)
    result = np.zeros((size, size))

    for octave in range(4):
        scale = 2 ** octave
        freq = (2 + scale) / size

        # Generate random phase and amplitude
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        amplitude = 1.0 / scale

        x = np.linspace(0, 2 * np.pi * freq * size, size)
        y = np.linspace(0, 2 * np.pi * freq * size, size)
        xx, yy = np.meshgrid(x, y)

        # Random direction gradients
        theta = np.random.rand() * 2 * np.pi
        result += amplitude * np.sin(xx * np.cos(theta) + yy * np.sin(theta) + phase_x)

    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_sinusoidal_image(size=64, seed=None):
    """Generate sinusoidal pattern with random frequencies/phases."""
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Random number of sine components (1-5)
    n_components = np.random.randint(1, 6)
    result = np.zeros((size, size))

    for _ in range(n_components):
        freq_x = np.random.uniform(1, 10)
        freq_y = np.random.uniform(1, 10)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        result += amplitude * np.sin(freq_x * xx * np.pi + freq_y * yy * np.pi + phase)

    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_gabor_image(size=64, seed=None):
    """Generate Gabor wavelet pattern."""
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Random Gabor parameters
    theta = np.random.uniform(0, np.pi)  # orientation
    sigma = np.random.uniform(0.1, 0.5)  # envelope width
    freq = np.random.uniform(2, 8)  # frequency
    phase = np.random.uniform(0, 2 * np.pi)

    # Rotate coordinates
    xx_rot = xx * np.cos(theta) + yy * np.sin(theta)
    yy_rot = -xx * np.sin(theta) + yy * np.cos(theta)

    # Gabor = Gaussian * sinusoid
    gaussian = np.exp(-(xx_rot**2 + yy_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * freq * xx_rot + phase)

    result = gaussian * sinusoid
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_random_image(size=64, seed=None):
    """Generate uniform random image (baseline)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(size, size)


def measure_order(image):
    """Measure order of an image (thresholded to binary)."""
    # Image may already be binary (0/1), or continuous [0,1]
    if image.max() > 1:
        binary = (image > 127).astype(np.uint8)
    else:
        binary = (image > 0.5).astype(np.uint8)
    return order_multiplicative(binary)


def run_experiment(n_samples=500, order_thresholds=[0.3, 0.5, 0.7]):
    """
    Generate samples from each generator and measure what fraction
    exceeds each order threshold.
    """
    generators = {
        'cppn': generate_cppn_image,
        'perlin': generate_perlin_image,
        'sinusoidal': generate_sinusoidal_image,
        'gabor': generate_gabor_image,
        'random': generate_random_image
    }

    results = {name: {'orders': [], 'images': []} for name in generators}

    print(f"Generating {n_samples} samples from each generator...")

    for name, gen_func in generators.items():
        print(f"  {name}...", end=" ", flush=True)
        for i in range(n_samples):
            img = gen_func(size=64, seed=i*1000 + hash(name) % 10000)
            order = measure_order(img)
            results[name]['orders'].append(order)
        print(f"done (mean order: {np.mean(results[name]['orders']):.3f})")

    # Compute success rates at each threshold
    success_rates = {thresh: {} for thresh in order_thresholds}

    for thresh in order_thresholds:
        for name in generators:
            orders = np.array(results[name]['orders'])
            success_rate = np.mean(orders >= thresh)
            success_rates[thresh][name] = success_rate

    # Compute effect sizes (CPPN vs others)
    effect_sizes = {}
    p_values = {}

    cppn_orders = np.array(results['cppn']['orders'])

    for name in ['perlin', 'sinusoidal', 'gabor', 'random']:
        other_orders = np.array(results[name]['orders'])

        # Cohen's d
        pooled_std = np.sqrt((cppn_orders.std()**2 + other_orders.std()**2) / 2)
        d = (cppn_orders.mean() - other_orders.mean()) / pooled_std
        effect_sizes[f'cppn_vs_{name}'] = d

        # Mann-Whitney U test (non-parametric)
        stat, p = stats.mannwhitneyu(cppn_orders, other_orders, alternative='greater')
        p_values[f'cppn_vs_{name}'] = p

    # Compute efficiency ratios at threshold 0.5
    thresh = 0.5
    efficiency_ratios = {}
    cppn_rate = success_rates[thresh]['cppn']
    for name in ['perlin', 'sinusoidal', 'gabor', 'random']:
        other_rate = success_rates[thresh][name]
        if other_rate > 0:
            ratio = cppn_rate / other_rate
        else:
            ratio = float('inf')
        efficiency_ratios[f'cppn_vs_{name}'] = ratio

    return {
        'results': {name: {'orders': results[name]['orders']} for name in generators},
        'success_rates': success_rates,
        'effect_sizes': effect_sizes,
        'p_values': p_values,
        'efficiency_ratios': efficiency_ratios,
        'order_stats': {
            name: {
                'mean': float(np.mean(results[name]['orders'])),
                'std': float(np.std(results[name]['orders'])),
                'median': float(np.median(results[name]['orders'])),
                'min': float(np.min(results[name]['orders'])),
                'max': float(np.max(results[name]['orders']))
            }
            for name in generators
        }
    }


if __name__ == "__main__":
    np.random.seed(42)

    results = run_experiment(n_samples=500)

    print("\n" + "="*60)
    print("RESULTS: Generator Efficiency Comparison (RES-119)")
    print("="*60)

    print("\nOrder Statistics by Generator:")
    print("-" * 50)
    for name, stats_dict in results['order_stats'].items():
        print(f"  {name:12s}: mean={stats_dict['mean']:.3f}, std={stats_dict['std']:.3f}, "
              f"range=[{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")

    print("\nSuccess Rates (fraction >= threshold):")
    print("-" * 50)
    for thresh in [0.3, 0.5, 0.7]:
        print(f"\n  Threshold T={thresh}:")
        for name in ['cppn', 'perlin', 'sinusoidal', 'gabor', 'random']:
            rate = results['success_rates'][thresh][name]
            print(f"    {name:12s}: {rate:.3f} ({rate*100:.1f}%)")

    print("\nEffect Sizes (Cohen's d, CPPN vs others):")
    print("-" * 50)
    for comparison, d in results['effect_sizes'].items():
        p = results['p_values'][comparison]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {comparison:20s}: d={d:+.2f}, p={p:.2e} {sig}")

    print("\nEfficiency Ratios at T=0.5 (CPPN success rate / other):")
    print("-" * 50)
    for comparison, ratio in results['efficiency_ratios'].items():
        if ratio == float('inf'):
            print(f"  {comparison:20s}: inf (other has 0% success)")
        else:
            print(f"  {comparison:20s}: {ratio:.1f}x")

    # Summary for validation
    print("\n" + "="*60)
    print("VALIDATION CHECK:")
    print("="*60)

    # Check if hypothesis is validated
    all_significant = all(p < 0.01 for p in results['p_values'].values())
    all_large_effect = all(d > 0.5 for d in results['effect_sizes'].values())

    max_effect = max(results['effect_sizes'].values())
    min_p = min(results['p_values'].values())

    print(f"  All p < 0.01: {all_significant}")
    print(f"  All d > 0.5: {all_large_effect}")
    print(f"  Max effect size: d={max_effect:.2f}")
    print(f"  Min p-value: p={min_p:.2e}")

    if all_significant and all_large_effect:
        print("\n  STATUS: VALIDATED")
    else:
        print("\n  STATUS: REFUTED or INCONCLUSIVE")
