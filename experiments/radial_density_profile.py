"""
RES-178: Radial Density Profile Analysis

Hypothesis: High-order CPPNs have more uniform radial density than low-order CPPNs

Theory: CPPNs use radial input r = sqrt(x^2 + y^2). High-order images may show
more uniform density across radius because structured patterns need balanced
foreground/background at all scales, whereas low-order patterns might
concentrate pixels in center or edges.

Metrics:
- Radial density profile: mean pixel density in concentric annuli
- Radial uniformity: std of density across annuli (lower = more uniform)
- Center-edge ratio: density at center vs edge
"""

import numpy as np
import json
import os
from pathlib import Path
from scipy import stats

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_radial_density_profile(img: np.ndarray, n_bins: int = 8) -> dict:
    """
    Compute density in concentric annuli from center.

    Returns dict with:
    - profile: density values for each annulus (center to edge)
    - uniformity: std of profile (lower = more uniform)
    - center_edge_ratio: center_density / edge_density
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Compute radius for each pixel
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)

    # Bin into annuli
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    profile = []

    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.sum() > 0:
            profile.append(float(np.mean(img[mask])))
        else:
            profile.append(0.5)  # Default if no pixels

    profile = np.array(profile)
    uniformity = float(np.std(profile))

    # Center-edge ratio (avoid division by zero)
    center_density = profile[0] if profile[0] > 0.01 else 0.01
    edge_density = profile[-1] if profile[-1] > 0.01 else 0.01
    center_edge_ratio = center_density / edge_density

    return {
        'profile': profile.tolist(),
        'uniformity': uniformity,
        'center_edge_ratio': float(center_edge_ratio),
        'center_density': float(profile[0]),
        'edge_density': float(profile[-1])
    }


def run_experiment(n_samples: int = 500, seed: int = 42):
    """Run radial density profile experiment."""
    set_global_seed(seed)

    results = []

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization
        img = cppn.render(32)
        order = order_multiplicative(img)

        radial = compute_radial_density_profile(img)

        results.append({
            'order': order,
            'uniformity': radial['uniformity'],
            'center_edge_ratio': radial['center_edge_ratio'],
            'center_density': radial['center_density'],
            'edge_density': radial['edge_density'],
            'profile': radial['profile']
        })

    return results


def analyze_results(results: list) -> dict:
    """Analyze correlation between order and radial density patterns."""
    orders = np.array([r['order'] for r in results])
    uniformities = np.array([r['uniformity'] for r in results])
    center_edge_ratios = np.array([r['center_edge_ratio'] for r in results])

    # Split into high/low order groups
    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = orders <= median_order

    # Correlations
    r_uniformity, p_uniformity = stats.pearsonr(orders, uniformities)
    rho_uniformity, p_uniformity_rho = stats.spearmanr(orders, uniformities)

    # T-test between high and low order groups
    high_uniformity = uniformities[high_mask]
    low_uniformity = uniformities[low_mask]
    t_stat, p_ttest = stats.ttest_ind(high_uniformity, low_uniformity)

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(high_uniformity) - 1) * np.std(high_uniformity, ddof=1)**2 +
         (len(low_uniformity) - 1) * np.std(low_uniformity, ddof=1)**2) /
        (len(high_uniformity) + len(low_uniformity) - 2)
    )
    cohens_d = (np.mean(high_uniformity) - np.mean(low_uniformity)) / pooled_std if pooled_std > 0 else 0

    # Center-edge analysis
    r_ce, p_ce = stats.pearsonr(orders, center_edge_ratios)

    # Per-annulus analysis
    n_bins = len(results[0]['profile'])
    annulus_correlations = []
    for bin_idx in range(n_bins):
        densities = np.array([r['profile'][bin_idx] for r in results])
        r_ann, p_ann = stats.pearsonr(orders, densities)
        annulus_correlations.append({
            'bin': bin_idx,
            'r': float(r_ann),
            'p': float(p_ann)
        })

    return {
        'n_samples': len(results),
        'uniformity': {
            'pearson_r': float(r_uniformity),
            'spearman_rho': float(rho_uniformity),
            'p_pearson': float(p_uniformity),
            'p_spearman': float(p_uniformity_rho),
            'high_order_mean': float(np.mean(high_uniformity)),
            'low_order_mean': float(np.mean(low_uniformity)),
            'cohens_d': float(cohens_d),
            't_stat': float(t_stat),
            'p_ttest': float(p_ttest)
        },
        'center_edge': {
            'pearson_r': float(r_ce),
            'p_value': float(p_ce),
            'high_order_mean': float(np.mean(center_edge_ratios[high_mask])),
            'low_order_mean': float(np.mean(center_edge_ratios[low_mask]))
        },
        'annulus_correlations': annulus_correlations,
        'order_stats': {
            'mean': float(np.mean(orders)),
            'std': float(np.std(orders)),
            'median': float(median_order),
            'min': float(np.min(orders)),
            'max': float(np.max(orders))
        }
    }


def main():
    print("RES-178: Radial Density Profile Analysis")
    print("=" * 50)
    print("Hypothesis: High-order CPPNs have more uniform radial density")
    print()

    # Run experiment
    print("Running experiment with 500 samples...")
    results = run_experiment(n_samples=500, seed=42)

    # Analyze
    analysis = analyze_results(results)

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "radial_density_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'experiment_id': 'RES-178',
            'hypothesis': 'High-order CPPNs have more uniform radial density than low-order CPPNs',
            'analysis': analysis
        }, f, indent=2)

    # Print summary
    u = analysis['uniformity']
    print(f"\nUniformity (std of radial profile):")
    print(f"  Pearson r with order: {u['pearson_r']:.3f} (p={u['p_pearson']:.2e})")
    print(f"  Spearman rho: {u['spearman_rho']:.3f} (p={u['p_spearman']:.2e})")
    print(f"  High-order mean: {u['high_order_mean']:.4f}")
    print(f"  Low-order mean: {u['low_order_mean']:.4f}")
    print(f"  Cohen's d: {u['cohens_d']:.3f}")
    print(f"  T-test p: {u['p_ttest']:.2e}")

    ce = analysis['center_edge']
    print(f"\nCenter-Edge Ratio:")
    print(f"  Pearson r: {ce['pearson_r']:.3f} (p={ce['p_value']:.2e})")

    print(f"\nPer-annulus correlations with order:")
    for ann in analysis['annulus_correlations']:
        sig = "*" if ann['p'] < 0.01 else ""
        print(f"  Bin {ann['bin']}: r={ann['r']:.3f} {sig}")

    # Determine status
    effect_size = abs(u['cohens_d'])
    p_value = u['p_ttest']

    print(f"\n{'='*50}")
    if effect_size >= 0.5 and p_value < 0.01:
        if u['cohens_d'] < 0:
            status = "validated"
            print("STATUS: VALIDATED - High-order CPPNs have significantly more uniform radial density")
        else:
            status = "refuted"
            print("STATUS: REFUTED - High-order CPPNs have LESS uniform radial density (opposite direction)")
    else:
        if p_value >= 0.01:
            status = "refuted"
            print(f"STATUS: REFUTED - Effect not significant (p={p_value:.2e} >= 0.01)")
        else:
            status = "refuted"
            print(f"STATUS: REFUTED - Effect size too small (d={effect_size:.3f} < 0.5)")

    return analysis, status


if __name__ == "__main__":
    analysis, status = main()
