"""
RES-100: Jacobian Singular Value Analysis of CPPNs

Hypothesis: Jacobian singular values of CPPNs correlate with image order -
high-order images have distinct spectral properties in d(output)/d(x,y).

Theory: The Jacobian J = [dO/dx, dO/dy] captures how the CPPN output changes
with spatial coordinates. High-order images should have:
- Lower Jacobian magnitude (smoother variations)
- More balanced singular values (isotropic structure)
- Lower effective rank (simpler spatial dependence)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats

def compute_jacobian_at_point(cppn: CPPN, x: float, y: float, eps: float = 1e-5) -> np.ndarray:
    """Compute Jacobian [dO/dx, dO/dy] at a single point using finite differences."""
    x_arr = np.array([x])
    y_arr = np.array([y])

    # Central differences
    o_xp = cppn.activate(np.array([x + eps]), y_arr)[0]
    o_xm = cppn.activate(np.array([x - eps]), y_arr)[0]
    o_yp = cppn.activate(x_arr, np.array([y + eps]))[0]
    o_ym = cppn.activate(x_arr, np.array([y - eps]))[0]

    dO_dx = (o_xp - o_xm) / (2 * eps)
    dO_dy = (o_yp - o_ym) / (2 * eps)

    return np.array([dO_dx, dO_dy])

def compute_jacobian_field(cppn: CPPN, size: int = 16, eps: float = 1e-5) -> tuple:
    """
    Compute Jacobian field over the image domain.

    Returns:
    - jacobians: (size, size, 2) array of [dO/dx, dO/dy] at each point
    - magnitudes: (size, size) array of ||J||
    - singular_values: list of 2D SVD results for each point
    """
    coords = np.linspace(-1, 1, size)
    jacobians = np.zeros((size, size, 2))

    for i, y in enumerate(coords):
        for j, x in enumerate(coords):
            jacobians[i, j] = compute_jacobian_at_point(cppn, x, y, eps)

    # Compute magnitudes
    magnitudes = np.sqrt(jacobians[:, :, 0]**2 + jacobians[:, :, 1]**2)

    return jacobians, magnitudes

def analyze_jacobian_spectrum(jacobians: np.ndarray) -> dict:
    """
    Analyze the singular value spectrum of the Jacobian field.

    Since J is 2D at each point, we can:
    1. Look at the distribution of gradient magnitudes
    2. Compute SVD of the entire Jacobian field reshaped as a matrix
    3. Compute anisotropy (ratio of directional gradients)
    """
    size = jacobians.shape[0]

    # Flatten to (N, 2) matrix where each row is [dO/dx, dO/dy] at a pixel
    J_flat = jacobians.reshape(-1, 2)

    # SVD of the Jacobian field (how much variation in each direction)
    U, S, Vt = np.linalg.svd(J_flat, full_matrices=False)

    # Individual gradient magnitudes
    magnitudes = np.sqrt(J_flat[:, 0]**2 + J_flat[:, 1]**2)

    # Anisotropy: how directional is the gradient field?
    # Ratio of singular values: 1 = isotropic, high = directional
    anisotropy = S[0] / (S[1] + 1e-10)

    # Effective rank (using normalized singular values)
    S_norm = S / (np.sum(S) + 1e-10)
    effective_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))

    # Gradient magnitude statistics
    mean_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)
    max_magnitude = np.max(magnitudes)

    # Gradient field entropy (how uniform is the gradient distribution)
    # Bin the magnitudes and compute entropy
    hist, _ = np.histogram(magnitudes, bins=20, density=True)
    hist = hist / (np.sum(hist) + 1e-10)
    gradient_entropy = -np.sum(hist * np.log(hist + 1e-10))

    return {
        'singular_values': S,
        'anisotropy': anisotropy,
        'effective_rank': effective_rank,
        'mean_magnitude': mean_magnitude,
        'std_magnitude': std_magnitude,
        'max_magnitude': max_magnitude,
        'gradient_entropy': gradient_entropy,
        'sv_ratio': S[0] / (S[1] + 1e-10),
    }

def sample_cppns_with_order(n_samples: int = 200, seed: int = 42) -> list:
    """Sample CPPNs and compute their order along with Jacobian properties."""
    set_global_seed(seed)
    results = []

    for i in range(n_samples):
        # Create random CPPN
        cppn = CPPN()

        # Render image and compute order
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        # Compute Jacobian field (use smaller grid for efficiency)
        jacobians, magnitudes = compute_jacobian_field(cppn, size=16)

        # Analyze spectrum
        spectrum_analysis = analyze_jacobian_spectrum(jacobians)

        results.append({
            'order': order,
            **spectrum_analysis
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n_samples} CPPNs")

    return results

def run_experiment():
    """Main experiment: correlate Jacobian properties with order."""
    print("=" * 60)
    print("RES-100: Jacobian Singular Value Analysis")
    print("=" * 60)

    # Sample CPPNs
    print("\nSampling CPPNs and computing Jacobian properties...")
    results = sample_cppns_with_order(n_samples=300, seed=42)

    # Extract arrays
    orders = np.array([r['order'] for r in results])
    anisotropies = np.array([r['anisotropy'] for r in results])
    effective_ranks = np.array([r['effective_rank'] for r in results])
    mean_magnitudes = np.array([r['mean_magnitude'] for r in results])
    max_magnitudes = np.array([r['max_magnitude'] for r in results])
    gradient_entropies = np.array([r['gradient_entropy'] for r in results])
    sv_ratios = np.array([r['sv_ratio'] for r in results])

    print(f"\nOrder distribution: mean={orders.mean():.3f}, std={orders.std():.3f}")
    print(f"  Low order (< 0.1): {np.sum(orders < 0.1)} samples")
    print(f"  Mid order (0.1-0.5): {np.sum((orders >= 0.1) & (orders < 0.5))} samples")
    print(f"  High order (>= 0.5): {np.sum(orders >= 0.5)} samples")

    # Compute correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlations = {}

    for name, values in [
        ('anisotropy', anisotropies),
        ('effective_rank', effective_ranks),
        ('mean_magnitude', mean_magnitudes),
        ('max_magnitude', max_magnitudes),
        ('gradient_entropy', gradient_entropies),
        ('sv_ratio', sv_ratios),
    ]:
        # Pearson correlation
        r, p = stats.pearsonr(orders, values)
        correlations[name] = {'r': r, 'p': p}

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{name:20s}: r={r:+.3f}, p={p:.2e} {sig}")

    # Compare high vs low order groups
    print("\n" + "=" * 60)
    print("GROUP COMPARISON: Low (O<0.1) vs High (O>=0.5)")
    print("=" * 60)

    low_mask = orders < 0.1
    high_mask = orders >= 0.5

    if np.sum(high_mask) < 10:
        # Adjust threshold if not enough high-order samples
        high_threshold = np.percentile(orders, 80)
        high_mask = orders >= high_threshold
        print(f"(Adjusted high threshold to {high_threshold:.3f}, top 20%)")

    group_comparisons = {}

    for name, values in [
        ('anisotropy', anisotropies),
        ('effective_rank', effective_ranks),
        ('mean_magnitude', mean_magnitudes),
        ('gradient_entropy', gradient_entropies),
    ]:
        low_vals = values[low_mask]
        high_vals = values[high_mask]

        if len(low_vals) > 5 and len(high_vals) > 5:
            # Mann-Whitney U test (non-parametric)
            stat, p = stats.mannwhitneyu(low_vals, high_vals, alternative='two-sided')

            # Effect size (Cohen's d approximation)
            pooled_std = np.sqrt((np.var(low_vals) + np.var(high_vals)) / 2)
            effect_size = (np.mean(high_vals) - np.mean(low_vals)) / (pooled_std + 1e-10)

            group_comparisons[name] = {
                'low_mean': np.mean(low_vals),
                'high_mean': np.mean(high_vals),
                'effect_size': effect_size,
                'p_value': p
            }

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:20s}: low={np.mean(low_vals):.3f}, high={np.mean(high_vals):.3f}, "
                  f"d={effect_size:+.2f}, p={p:.2e} {sig}")

    # Primary hypothesis test: mean_magnitude correlation with order
    print("\n" + "=" * 60)
    print("PRIMARY HYPOTHESIS TEST")
    print("=" * 60)

    r_primary = correlations['mean_magnitude']['r']
    p_primary = correlations['mean_magnitude']['p']

    # Effect size from group comparison
    if 'mean_magnitude' in group_comparisons:
        d_primary = group_comparisons['mean_magnitude']['effect_size']
    else:
        d_primary = 0.0

    print(f"\nHypothesis: Jacobian magnitude correlates with order")
    print(f"Correlation r = {r_primary:.3f}, p = {p_primary:.2e}")
    print(f"Group effect size d = {d_primary:.3f}")

    # Validation criteria
    is_significant = p_primary < 0.01
    has_effect = abs(d_primary) > 0.5

    if is_significant and has_effect:
        status = "VALIDATED"
        direction = "negative" if r_primary < 0 else "positive"
        summary = (f"Jacobian magnitude shows {direction} correlation with order "
                  f"(r={r_primary:.3f}, d={d_primary:.2f}). High-order images have "
                  f"{'lower' if r_primary < 0 else 'higher'} gradient magnitudes.")
    elif is_significant:
        status = "INCONCLUSIVE"
        summary = f"Correlation is significant (p={p_primary:.2e}) but effect size weak (d={d_primary:.2f})."
    else:
        status = "REFUTED"
        summary = f"No significant correlation found (r={r_primary:.3f}, p={p_primary:.2e})."

    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Return results for logging
    return {
        'status': status.lower(),
        'summary': summary,
        'metrics': {
            'r_magnitude': float(r_primary),
            'p_magnitude': float(p_primary),
            'effect_size': float(d_primary),
            'n_samples': len(orders),
            'n_high_order': int(np.sum(high_mask)),
            'correlations': {k: {'r': float(v['r']), 'p': float(v['p'])}
                           for k, v in correlations.items()}
        }
    }

if __name__ == "__main__":
    results = run_experiment()
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    for k, v in results['metrics'].items():
        if isinstance(v, dict):
            continue
        print(f"  {k}: {v}")
