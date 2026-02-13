"""
RES-023: Boundary Effects on Order Metric

Hypothesis: Boundary pixels contribute disproportionately MORE to the order metric
than interior pixels of equal count, due to edge detection artifacts, spectral
edge effects, and connected component boundary interactions.

Method:
- Generate N CPPN images
- For each image, compute baseline order
- Create boundary-masked version (set boundary ring to neutral 0.5, then binarize)
- Create interior-masked version (set equivalent # of interior pixels to 0.5)
- Compare |order_change| for boundary mask vs interior mask
- Test with Wilcoxon signed-rank (paired), Cohen's d

Null Hypothesis: Boundary masking produces the same |order_change| as interior
masking of equivalent pixel count - boundary pixels have no special contribution.

Builds on: RES-020 (scale dependence shows 8x8 characteristic scale)
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def mask_boundary(img: np.ndarray, ring_width: int = 2) -> np.ndarray:
    """
    Mask boundary pixels by setting them to neutral gray (0), then binarizing.

    Since our images are binary, we need a different approach:
    We'll set boundary pixels to 0 (black) to neutralize the boundary.
    This is equivalent to extending the "void" at edges.
    """
    result = img.copy()
    h, w = img.shape

    # Set boundary ring to 0
    result[:ring_width, :] = 0  # Top
    result[-ring_width:, :] = 0  # Bottom
    result[:, :ring_width] = 0  # Left
    result[:, -ring_width:] = 0  # Right

    return result


def mask_interior_equivalent(img: np.ndarray, ring_width: int = 2, seed: int = None) -> np.ndarray:
    """
    Mask an equivalent number of interior pixels randomly.

    The boundary ring contains: 2 * ring_width * (H + W) - 4 * ring_width^2 pixels
    We mask the same number of interior pixels.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    result = img.copy()
    h, w = img.shape

    # Calculate boundary pixel count
    boundary_pixels = 2 * ring_width * (h + w) - 4 * ring_width ** 2

    # Get interior pixel indices
    interior_mask = np.ones_like(img, dtype=bool)
    interior_mask[:ring_width, :] = False
    interior_mask[-ring_width:, :] = False
    interior_mask[:, :ring_width] = False
    interior_mask[:, -ring_width:] = False

    interior_indices = np.where(interior_mask)
    n_interior = len(interior_indices[0])

    if n_interior < boundary_pixels:
        # Can't mask equivalent number - just mask all interior
        result[interior_mask] = 0
        return result

    # Randomly select interior pixels to mask
    selected = rng.choice(n_interior, size=boundary_pixels, replace=False)
    for idx in selected:
        result[interior_indices[0][idx], interior_indices[1][idx]] = 0

    return result


def run_experiment(
    n_samples: int = 500,
    image_size: int = 32,
    ring_widths: list = [1, 2, 4],  # Test multiple boundary widths
    n_interior_reps: int = 10,  # Multiple random interior masks per image
    seed: int = 42
) -> dict:
    """
    Test whether boundary pixels contribute more to order than interior pixels.
    """
    set_global_seed(seed)

    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'ring_widths': ring_widths,
        'n_interior_reps': n_interior_reps,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
    }

    # Generate CPPN images
    print(f"Generating {n_samples} CPPN images...")
    images = []
    original_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=image_size)
        order = order_multiplicative(img)
        images.append(img)
        original_orders.append(order)

    original_orders = np.array(original_orders)
    print(f"Original order: mean={np.mean(original_orders):.4f}, std={np.std(original_orders):.4f}")

    results['original_order_mean'] = float(np.mean(original_orders))
    results['original_order_std'] = float(np.std(original_orders))

    # Test each ring width
    results['by_ring_width'] = {}

    for ring_width in ring_widths:
        print(f"\n--- Ring width: {ring_width} ---")

        # Calculate number of boundary pixels
        boundary_pixels = 2 * ring_width * (image_size + image_size) - 4 * ring_width ** 2
        print(f"Boundary pixels: {boundary_pixels}")

        boundary_changes = []
        interior_changes = []
        boundary_orders = []
        interior_orders = []

        for idx, (img, orig_order) in enumerate(zip(images, original_orders)):
            # Boundary mask
            boundary_masked = mask_boundary(img, ring_width)
            boundary_order = order_multiplicative(boundary_masked)
            boundary_change = boundary_order - orig_order
            boundary_changes.append(boundary_change)
            boundary_orders.append(boundary_order)

            # Interior masks (multiple random samples)
            int_changes = []
            int_orders = []
            for rep in range(n_interior_reps):
                interior_masked = mask_interior_equivalent(img, ring_width, seed=idx*1000 + rep)
                interior_order = order_multiplicative(interior_masked)
                int_changes.append(interior_order - orig_order)
                int_orders.append(interior_order)

            interior_changes.append(np.mean(int_changes))
            interior_orders.append(np.mean(int_orders))

        boundary_changes = np.array(boundary_changes)
        interior_changes = np.array(interior_changes)
        boundary_orders = np.array(boundary_orders)
        interior_orders = np.array(interior_orders)

        # Statistics: Compare |boundary_change| vs |interior_change|
        abs_boundary = np.abs(boundary_changes)
        abs_interior = np.abs(interior_changes)

        # Paired Wilcoxon signed-rank test (two-sided first to detect any difference)
        stat_w_two, p_w_two = stats.wilcoxon(abs_boundary, abs_interior, alternative='two-sided')

        # Then test which direction (interior > boundary)
        stat_w, p_w = stats.wilcoxon(abs_interior, abs_boundary, alternative='greater')

        # Cohen's d (paired)
        diff = abs_boundary - abs_interior
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

        # Bootstrap CI for Cohen's d
        n_bootstrap = 1000
        bootstrap_d = []
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(diff), size=len(diff), replace=True)
            boot_diff = diff[boot_idx]
            bootstrap_d.append(np.mean(boot_diff) / (np.std(boot_diff) + 1e-10))
        d_ci = np.percentile(bootstrap_d, [2.5, 97.5])

        # Direction analysis
        frac_boundary_larger = np.mean(abs_boundary > abs_interior)

        # Mean changes (signed)
        mean_boundary_change = np.mean(boundary_changes)
        mean_interior_change = np.mean(interior_changes)

        print(f"\nBoundary masking: mean |change| = {np.mean(abs_boundary):.4f}")
        print(f"Interior masking: mean |change| = {np.mean(abs_interior):.4f}")
        print(f"Interior produces larger change: {(1-frac_boundary_larger)*100:.1f}% of images")
        print(f"Wilcoxon (two-sided): p = {p_w_two:.4e}")
        print(f"Wilcoxon (one-sided, interior > boundary): p = {p_w:.4e}")
        print(f"Cohen's d (boundary - interior): {cohens_d:.3f} (95% CI: [{d_ci[0]:.3f}, {d_ci[1]:.3f}])")

        # Also test signed changes (direction of effect)
        stat_signed, p_signed = stats.wilcoxon(boundary_changes, interior_changes, alternative='two-sided')

        results['by_ring_width'][ring_width] = {
            'boundary_pixels': int(boundary_pixels),
            'mean_abs_boundary_change': float(np.mean(abs_boundary)),
            'mean_abs_interior_change': float(np.mean(abs_interior)),
            'std_abs_boundary_change': float(np.std(abs_boundary)),
            'std_abs_interior_change': float(np.std(abs_interior)),
            'frac_interior_larger': float(1 - frac_boundary_larger),
            'wilcoxon_two_sided_p': float(p_w_two),
            'wilcoxon_interior_gt_boundary_p': float(p_w),
            'cohens_d': float(cohens_d),  # Negative means interior > boundary
            'd_ci_95': [float(d_ci[0]), float(d_ci[1])],
            'mean_boundary_change_signed': float(mean_boundary_change),
            'mean_interior_change_signed': float(mean_interior_change),
            'wilcoxon_signed_p': float(p_signed),
            'mean_boundary_order': float(np.mean(boundary_orders)),
            'mean_interior_order': float(np.mean(interior_orders)),
        }

    # Primary analysis: Use ring_width=2 as main result
    primary_ring = 2
    primary_result = results['by_ring_width'][primary_ring]

    results['primary_result'] = {
        'ring_width': primary_ring,
        'wilcoxon_two_sided_p': primary_result['wilcoxon_two_sided_p'],
        'wilcoxon_interior_gt_boundary_p': primary_result['wilcoxon_interior_gt_boundary_p'],
        'cohens_d': primary_result['cohens_d'],
        'd_ci_95': primary_result['d_ci_95'],
        'frac_interior_larger': primary_result['frac_interior_larger'],
    }

    # Determine status
    # Use two-sided test for significance, then look at direction
    p_threshold = 0.01
    d_threshold = 0.5

    p_val = primary_result['wilcoxon_two_sided_p']
    d_val = primary_result['cohens_d']  # Negative if interior > boundary

    if p_val < p_threshold and d_val < -d_threshold:
        # Interior contributes MORE than boundary (opposite of original hypothesis)
        results['status'] = 'refuted'
        results['conclusion'] = f"OPPOSITE finding: Interior pixels contribute MORE to order than boundary pixels (d={-d_val:.2f}, p={p_val:.4e})"
    elif p_val < p_threshold and d_val > d_threshold:
        results['status'] = 'validated'
        results['conclusion'] = f"Boundary pixels contribute MORE to order (d={d_val:.2f}, p={p_val:.4e})"
    elif p_val >= p_threshold:
        results['status'] = 'refuted'
        results['conclusion'] = f"No significant boundary effect (p={p_val:.4f})"
    else:
        results['status'] = 'inconclusive'
        results['conclusion'] = f"Significant but small effect (d={d_val:.3f})"

    return results


def main():
    results = run_experiment(
        n_samples=500,
        image_size=32,
        ring_widths=[1, 2, 4],
        n_interior_reps=10,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'boundary_effects'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'boundary_effects_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Boundary Effects on Order Metric")
    print("="*70)

    print(f"\nHypothesis: Boundary pixels contribute MORE to order than interior pixels")

    for ring_width in results['by_ring_width']:
        rw = results['by_ring_width'][ring_width]
        print(f"\nRing width {ring_width}:")
        print(f"  Boundary |change|: {rw['mean_abs_boundary_change']:.4f} +/- {rw['std_abs_boundary_change']:.4f}")
        print(f"  Interior |change|: {rw['mean_abs_interior_change']:.4f} +/- {rw['std_abs_interior_change']:.4f}")
        print(f"  Interior larger: {rw['frac_interior_larger']*100:.1f}%")
        print(f"  Wilcoxon (two-sided) p: {rw['wilcoxon_two_sided_p']:.4e}")
        print(f"  Cohen's d (boundary - interior): {rw['cohens_d']:.3f}")

    print(f"\n=== STATUS: {results['status'].upper()} ===")
    print(f"Conclusion: {results['conclusion']}")

    return results


if __name__ == '__main__':
    main()
