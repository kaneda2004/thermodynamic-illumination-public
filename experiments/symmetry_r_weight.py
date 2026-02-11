"""
RES-171: Test if larger |weight| on r-input produces more rotational symmetry in CPPN outputs.

Hypothesis: The r (radius) input is inherently rotationally symmetric (r = sqrt(x^2 + y^2)),
so CPPNs with higher weight magnitude on the r-input should produce outputs with more
rotational symmetry (rot90, rot180).

Approach:
1. Generate 500 random CPPNs
2. For each: extract |weight| on r-input connection
3. Compute rotational symmetry metrics (rot90, rot180)
4. Correlate r-weight magnitude with rotational symmetry
5. Compare high vs low r-weight groups

Success criteria: p < 0.01, Cohen's d > 0.5
"""

import numpy as np
import json
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, set_global_seed

def compute_rot90_symmetry(img):
    """Fraction of pixels matching after 90-degree rotation."""
    rotated = np.rot90(img)
    return np.mean(img == rotated)

def compute_rot180_symmetry(img):
    """Fraction of pixels matching after 180-degree rotation."""
    rotated = np.rot90(img, 2)
    return np.mean(img == rotated)

def compute_reflection_symmetry(img):
    """Average of horizontal and vertical reflection symmetry."""
    h_sym = np.mean(img == np.fliplr(img))
    v_sym = np.mean(img == np.flipud(img))
    return (h_sym + v_sym) / 2

def get_r_weight(cppn):
    """Get the weight on the r-input (node id 2) to output connection."""
    for conn in cppn.connections:
        if conn.from_id == 2 and conn.to_id == cppn.output_id and conn.enabled:
            return conn.weight
    return 0.0

def get_all_weights(cppn):
    """Get weights for x (0), y (1), r (2), bias (3) inputs."""
    weights = {}
    for conn in cppn.connections:
        if conn.to_id == cppn.output_id and conn.enabled:
            weights[conn.from_id] = conn.weight
    return weights

def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)

def main():
    set_global_seed(42)
    n_samples = 500

    results = {
        'r_weights': [],
        'abs_r_weights': [],
        'x_weights': [],
        'y_weights': [],
        'rot90_sym': [],
        'rot180_sym': [],
        'refl_sym': [],
    }

    print(f"Generating {n_samples} CPPNs...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=32)

        weights = get_all_weights(cppn)
        r_weight = weights.get(2, 0.0)
        x_weight = weights.get(0, 0.0)
        y_weight = weights.get(1, 0.0)

        rot90 = compute_rot90_symmetry(img)
        rot180 = compute_rot180_symmetry(img)
        refl = compute_reflection_symmetry(img)

        results['r_weights'].append(r_weight)
        results['abs_r_weights'].append(abs(r_weight))
        results['x_weights'].append(x_weight)
        results['y_weights'].append(y_weight)
        results['rot90_sym'].append(rot90)
        results['rot180_sym'].append(rot180)
        results['refl_sym'].append(refl)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Combined rotational symmetry
    rot_sym = (results['rot90_sym'] + results['rot180_sym']) / 2

    # Correlations
    r_rot90_corr, r_rot90_p = stats.pearsonr(results['abs_r_weights'], results['rot90_sym'])
    r_rot180_corr, r_rot180_p = stats.pearsonr(results['abs_r_weights'], results['rot180_sym'])
    r_rot_corr, r_rot_p = stats.pearsonr(results['abs_r_weights'], rot_sym)
    r_refl_corr, r_refl_p = stats.pearsonr(results['abs_r_weights'], results['refl_sym'])

    # Comparison: x/y weights vs rotational symmetry
    xy_mag = np.sqrt(results['x_weights']**2 + results['y_weights']**2)
    xy_rot_corr, xy_rot_p = stats.pearsonr(xy_mag, rot_sym)

    # Split into high vs low r-weight groups
    median_r = np.median(results['abs_r_weights'])
    high_r_mask = results['abs_r_weights'] > median_r
    low_r_mask = ~high_r_mask

    high_r_rot = rot_sym[high_r_mask]
    low_r_rot = rot_sym[low_r_mask]

    d = cohens_d(high_r_rot, low_r_rot)
    t_stat, t_p = stats.ttest_ind(high_r_rot, low_r_rot)

    # Print results
    print("\n=== RESULTS ===")
    print(f"\n|r-weight| vs rot90 symmetry: r={r_rot90_corr:.3f}, p={r_rot90_p:.2e}")
    print(f"|r-weight| vs rot180 symmetry: r={r_rot180_corr:.3f}, p={r_rot180_p:.2e}")
    print(f"|r-weight| vs rotational symmetry (avg): r={r_rot_corr:.3f}, p={r_rot_p:.2e}")
    print(f"|r-weight| vs reflection symmetry: r={r_refl_corr:.3f}, p={r_refl_p:.2e}")
    print(f"|xy-weight| vs rotational symmetry: r={xy_rot_corr:.3f}, p={xy_rot_p:.2e}")

    print(f"\nHigh |r-weight| group (n={sum(high_r_mask)}): mean rot_sym={np.mean(high_r_rot):.4f}")
    print(f"Low |r-weight| group (n={sum(low_r_mask)}): mean rot_sym={np.mean(low_r_rot):.4f}")
    print(f"Cohen's d={d:.3f}, t-test p={t_p:.2e}")

    # Check relative importance of r vs x/y
    # Ratio of r-weight to total weight magnitude
    total_weights = np.sqrt(results['x_weights']**2 + results['y_weights']**2 + results['abs_r_weights']**2)
    r_ratio = results['abs_r_weights'] / (total_weights + 1e-10)
    r_ratio_corr, r_ratio_p = stats.pearsonr(r_ratio, rot_sym)
    print(f"\nr-weight ratio (|r|/total) vs rotational symmetry: r={r_ratio_corr:.3f}, p={r_ratio_p:.2e}")

    # Additional analysis: r-dominance vs x/y-dominance
    r_dominant = results['abs_r_weights'] > xy_mag
    r_dom_rot = rot_sym[r_dominant]
    xy_dom_rot = rot_sym[~r_dominant]
    d_dominance = cohens_d(r_dom_rot, xy_dom_rot)
    t_dom, p_dom = stats.ttest_ind(r_dom_rot, xy_dom_rot)

    print(f"\n--- R-dominance analysis ---")
    print(f"R-dominant CPPNs (|r| > |xy|): n={sum(r_dominant)}, mean rot_sym={np.mean(r_dom_rot):.4f}")
    print(f"XY-dominant CPPNs (|xy| > |r|): n={sum(~r_dominant)}, mean rot_sym={np.mean(xy_dom_rot):.4f}")
    print(f"Cohen's d={d_dominance:.3f}, p={p_dom:.2e}")

    # Use the dominance comparison for validation
    if abs(d_dominance) > 0.5 and p_dom < 0.01:
        status = 'validated'
    elif p_dom > 0.1 or abs(d_dominance) < 0.2:
        status = 'refuted'
    else:
        status = 'inconclusive'

    print(f"\n=== STATUS: {status.upper()} ===")

    # Save results
    output = {
        'hypothesis': 'Larger |weight| on r-input produces more rotational symmetry in CPPN outputs',
        'domain': 'symmetry_mechanisms',
        'n_samples': n_samples,
        'correlations': {
            'abs_r_vs_rot90': {'r': float(r_rot90_corr), 'p': float(r_rot90_p)},
            'abs_r_vs_rot180': {'r': float(r_rot180_corr), 'p': float(r_rot180_p)},
            'abs_r_vs_rot_avg': {'r': float(r_rot_corr), 'p': float(r_rot_p)},
            'abs_r_vs_reflection': {'r': float(r_refl_corr), 'p': float(r_refl_p)},
            'xy_mag_vs_rot': {'r': float(xy_rot_corr), 'p': float(xy_rot_p)},
            'r_ratio_vs_rot': {'r': float(r_ratio_corr), 'p': float(r_ratio_p)},
        },
        'group_comparison': {
            'high_r_mean_rot': float(np.mean(high_r_rot)),
            'low_r_mean_rot': float(np.mean(low_r_rot)),
            'cohens_d': float(d),
            't_test_p': float(t_p),
        },
        'dominance_analysis': {
            'r_dominant_n': int(sum(r_dominant)),
            'xy_dominant_n': int(sum(~r_dominant)),
            'r_dominant_mean_rot': float(np.mean(r_dom_rot)),
            'xy_dominant_mean_rot': float(np.mean(xy_dom_rot)),
            'cohens_d': float(d_dominance),
            'p_value': float(p_dom),
        },
        'status': status,
    }

    with open('/Users/matt/Development/monochrome_noise_converger/results/symmetry_r_weight/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to results/symmetry_r_weight/results.json")

    return output

if __name__ == '__main__':
    main()
