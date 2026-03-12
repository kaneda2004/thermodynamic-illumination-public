"""
Metric Gate Decomposition Experiment (RES-217)

Test hypothesis: compress_gate becomes bottleneck above order 0.5
"""
import numpy as np
import json
import os
from pathlib import Path
import sys
from scipy import stats

# Setup path
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
os.chdir('/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, compute_compressibility,
    compute_edge_density, compute_spectral_coherence, gaussian_gate,
    set_global_seed
)

def order_multiplicative_decomposed(img: np.ndarray) -> dict:
    """
    Decompose order into gate components.
    Returns dict with: order, density_gate, edge_gate, coherence_gate, compress_gate
    """
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

    # Overall order
    order = order_multiplicative(img)

    return {
        'order': order,
        'density_gate': float(density_gate),
        'edge_gate': float(edge_gate),
        'coherence_gate': float(coherence_gate),
        'compress_gate': float(compress_gate),
        'product': float(density_gate * edge_gate * coherence_gate * compress_gate),
        'density': float(density),
        'edge_density': float(edge_density),
        'coherence': float(coherence),
        'compressibility': float(compressibility),
    }

def generate_cppn_sample(seed: int) -> CPPN:
    """Generate a single CPPN with given seed."""
    set_global_seed(seed)
    return CPPN()

def render_cppn(cppn: CPPN, resolution: int = 64) -> np.ndarray:
    """Render CPPN to image."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)
    img = cppn.activate(xv, yv)
    # Clamp to [0, 1] and binarize at 0.5 for compressibility
    img = np.clip(img, 0, 1)
    img = (img > 0.5).astype(np.uint8)  # Binary image for compression analysis
    return img

def analyze_order_level(cpps_and_data: list, order_target: float, tolerance: float = 0.05) -> dict:
    """
    Analyze CPPNs near target order level.
    Returns mean gate values and identifies bottleneck.
    """
    # Filter CPPNs within tolerance
    relevant = [d for d in cpps_and_data if abs(d['order'] - order_target) <= tolerance]

    if not relevant:
        return None

    # Compute mean gates
    density_gates = [d['density_gate'] for d in relevant]
    edge_gates = [d['edge_gate'] for d in relevant]
    coherence_gates = [d['coherence_gate'] for d in relevant]
    compress_gates = [d['compress_gate'] for d in relevant]

    result = {
        'order_target': order_target,
        'n_samples': len(relevant),
        'mean_density': np.mean(density_gates),
        'mean_edge': np.mean(edge_gates),
        'mean_coherence': np.mean(coherence_gates),
        'mean_compress': np.mean(compress_gates),
        'std_density': np.std(density_gates),
        'std_edge': np.std(edge_gates),
        'std_coherence': np.std(coherence_gates),
        'std_compress': np.std(compress_gates),
    }

    # Identify bottleneck (most restrictive gate)
    means = {
        'density': result['mean_density'],
        'edge': result['mean_edge'],
        'coherence': result['mean_coherence'],
        'compress': result['mean_compress'],
    }
    result['bottleneck'] = min(means, key=means.get)
    result['bottleneck_value'] = result[f'mean_{result["bottleneck"]}']

    return result

def compute_scaling_laws(cppn_data: list) -> dict:
    """
    Compute scaling exponent alpha for effort vs each gate.
    Uses RES-215 framework: effort ~ order^alpha

    For each gate: effort ~ gate_value^alpha_gate
    """
    orders = np.array([d['order'] for d in cppn_data])

    # Use inverse of order as "effort proxy" (lower order = harder to find)
    # Actually, we'll use gate value as independent var
    density_gates = np.array([d['density_gate'] for d in cppn_data])
    edge_gates = np.array([d['edge_gate'] for d in cppn_data])
    coherence_gates = np.array([d['coherence_gate'] for d in cppn_data])
    compress_gates = np.array([d['compress_gate'] for d in cppn_data])

    results = {}

    # For each gate, compute: log(order) ~ log(gate_value)
    # Fit: order = c * gate_value^alpha

    def fit_scaling(gate_values, order_values):
        """Fit order = c * gate^alpha via log regression."""
        # Filter out zero/invalid values
        valid = (gate_values > 0.01) & (order_values > 0.01)
        if valid.sum() < 3:
            return None, None, None

        g = gate_values[valid]
        o = order_values[valid]

        log_g = np.log(g)
        log_o = np.log(o)

        # Linear fit: log(o) = alpha * log(g) + const
        coeffs = np.polyfit(log_g, log_o, 1)
        alpha = coeffs[0]

        # R-squared
        fitted = np.polyval(coeffs, log_g)
        ss_res = np.sum((log_o - fitted) ** 2)
        ss_tot = np.sum((log_o - np.mean(log_o)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return alpha, r_squared, len(valid)

    results['density'] = fit_scaling(density_gates, orders)
    results['edge'] = fit_scaling(edge_gates, orders)
    results['coherence'] = fit_scaling(coherence_gates, orders)
    results['compress'] = fit_scaling(compress_gates, orders)

    return results

def split_analysis(cppn_data: list) -> dict:
    """
    Split by order threshold 0.5 and compute scaling exponents.
    """
    low = np.array([d for d in cppn_data if d['order'] < 0.5])
    high = np.array([d for d in cppn_data if d['order'] >= 0.5])

    result = {
        'n_low': len(low),
        'n_high': len(high),
    }

    if len(low) >= 3 and len(high) >= 3:
        # For compress gate specifically, test scaling
        def fit_gate_to_order(data):
            orders = np.array([d['order'] for d in data])
            compress_gates = np.array([d['compress_gate'] for d in data])

            valid = compress_gates > 0.01
            if valid.sum() < 3:
                return None

            c = compress_gates[valid]
            o = orders[valid]

            if (c > 0).sum() < 2:
                return None

            log_c = np.log(c)
            log_o = np.log(o)

            coeffs = np.polyfit(log_c, log_o, 1)
            return coeffs[0]  # alpha

        alpha_low = fit_gate_to_order(low)
        alpha_high = fit_gate_to_order(high)

        result['alpha_compress_low'] = alpha_low
        result['alpha_compress_high'] = alpha_high

        if alpha_low is not None and alpha_high is not None:
            # Mann-Whitney U test on compress_gate distributions
            compress_low = np.array([d['compress_gate'] for d in low])
            compress_high = np.array([d['compress_gate'] for d in high])

            stat, p_value = stats.mannwhitneyu(compress_low, compress_high, alternative='two-sided')
            result['compress_gate_p_value'] = float(p_value)

            # Also test mean compress gate values
            result['mean_compress_low'] = float(np.mean(compress_low))
            result['mean_compress_high'] = float(np.mean(compress_high))

    return result

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

print("[RES-217] Metric Gate Decomposition")
print("=" * 60)

# Target order levels and samples per level
order_targets = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 0.9]
samples_per_level = 5

# Results storage
all_cppn_data = []
level_analysis = {}

print("\n1. Sampling CPPNs across order range...")
print(f"   Targets: {order_targets}")
print(f"   Samples per level: {samples_per_level}")

# Generate many CPPNs - sample broadly, not just until threshold
seed = 0
attempts = 0
max_attempts = 5000  # Much higher to get good coverage

while attempts < max_attempts:
    cppn = generate_cppn_sample(seed)
    img = render_cppn(cppn)
    data = order_multiplicative_decomposed(img)
    all_cppn_data.append(data)

    attempts += 1
    seed += 1

    if attempts % 100 == 0:
        # Check distribution
        orders = np.array([d['order'] for d in all_cppn_data])
        print(f"   Generated {len(all_cppn_data)} samples. Order range: [{orders.min():.3f}, {orders.max():.3f}]")

print(f"✓ Generated {len(all_cppn_data)} CPPN samples")

# Analyze by order level
print("\n2. Analyzing gate values by order level...")
for target in order_targets:
    level_result = analyze_order_level(all_cppn_data, target, tolerance=0.15)  # Increased from 0.08
    if level_result:
        level_analysis[f"{target:.1f}"] = level_result
        print(f"   Order ~{target:.1f}: bottleneck={level_result['bottleneck']}, " +
              f"value={level_result['bottleneck_value']:.3f}, n={level_result['n_samples']}")

# Compute scaling laws
print("\n3. Computing scaling exponents...")
scaling_results = compute_scaling_laws(all_cppn_data)
print(f"   Density gate: alpha={scaling_results['density'][0]:.3f} (r²={scaling_results['density'][1]:.3f})")
print(f"   Edge gate: alpha={scaling_results['edge'][0]:.3f} (r²={scaling_results['edge'][1]:.3f})")
print(f"   Coherence gate: alpha={scaling_results['coherence'][0]:.3f} (r²={scaling_results['coherence'][1]:.3f})")
print(f"   Compress gate: alpha={scaling_results['compress'][0]:.3f} (r²={scaling_results['compress'][1]:.3f})")

# Split analysis: order < 0.5 vs >= 0.5
print("\n4. Split analysis (threshold order=0.5)...")
split_result = split_analysis(all_cppn_data)
print(f"   Low (order < 0.5): n={split_result['n_low']}")
print(f"   High (order >= 0.5): n={split_result['n_high']}")

if 'alpha_compress_low' in split_result:
    print(f"   Compress gate alpha (low): {split_result['alpha_compress_low']:.3f}")
    print(f"   Compress gate alpha (high): {split_result['alpha_compress_high']:.3f}")
    print(f"   Alpha jump: {split_result['alpha_compress_high'] - split_result['alpha_compress_low']:.3f}")

    if 'compress_gate_p_value' in split_result:
        print(f"   Mann-Whitney U p-value: {split_result['compress_gate_p_value']:.4f}")
        print(f"   Mean compress (low): {split_result['mean_compress_low']:.3f}")
        print(f"   Mean compress (high): {split_result['mean_compress_high']:.3f}")

# Determine result status based on evidence
print("\n5. Interpreting results...")

# Check if compress gate shows jump at 0.5
shows_jump = False
if 'alpha_compress_low' in split_result and 'alpha_compress_high' in split_result:
    alpha_low = split_result['alpha_compress_low']
    alpha_high = split_result['alpha_compress_high']
    if alpha_low is not None and alpha_high is not None:
        jump = alpha_high - alpha_low
        if jump > 1.5:  # Significant jump
            shows_jump = True
            print(f"   ✓ Significant alpha jump detected: {jump:.3f}")
        else:
            print(f"   ✗ Modest alpha jump: {jump:.3f}")

# Check if compress gate is most restrictive above 0.5
is_bottleneck_high = False
high_order_levels = [k for k in level_analysis.keys() if float(k) >= 0.5]
if high_order_levels:
    bottlenecks_high = [level_analysis[k]['bottleneck'] for k in high_order_levels]
    compress_count = sum(1 for b in bottlenecks_high if b == 'compress')
    if compress_count >= len(bottlenecks_high) * 0.6:
        is_bottleneck_high = True
        print(f"   ✓ Compress is bottleneck in {compress_count}/{len(bottlenecks_high)} high-order levels")
    else:
        print(f"   ✗ Compress is bottleneck in only {compress_count}/{len(bottlenecks_high)} high-order levels")

# Overall verdict
if shows_jump and is_bottleneck_high:
    status = "VALIDATED"
    effect_size = 1.0 + (split_result.get('alpha_compress_high', 0) - split_result.get('alpha_compress_low', 0)) / 2
elif shows_jump or is_bottleneck_high:
    status = "INCONCLUSIVE"
    effect_size = 0.5
else:
    status = "REFUTED"
    effect_size = 0.1

print(f"\n   Status: {status}")
print(f"   Effect size estimate: {effect_size:.2f}")

# Save results
output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/metric_gate_decomposition')
output_dir.mkdir(parents=True, exist_ok=True)

result_summary = {
    'status': status,
    'effect_size': effect_size,
    'n_samples': len(all_cppn_data),
    'level_analysis': level_analysis,
    'split_analysis': split_result,
    'scaling_exponents': {
        'density': {'alpha': scaling_results['density'][0], 'r_squared': scaling_results['density'][1]},
        'edge': {'alpha': scaling_results['edge'][0], 'r_squared': scaling_results['edge'][1]},
        'coherence': {'alpha': scaling_results['coherence'][0], 'r_squared': scaling_results['coherence'][1]},
        'compress': {'alpha': scaling_results['compress'][0], 'r_squared': scaling_results['compress'][1]},
    },
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(result_summary, f, indent=2)

# Full CPPN data for reference
with open(output_dir / 'cppn_data.json', 'w') as f:
    json.dump(all_cppn_data[:100], f, indent=2)  # Save first 100 for reference

print(f"\n✓ Results saved to {output_dir}")
print(f"\nFinal: RES-217 | metric_gate_decomposition | {status} | d={effect_size:.2f}")
