#!/usr/bin/env python3
"""
RES-164: Local patch perturbations vs distributed random pixel flips.

HYPOTHESIS: Local patch perturbations affect order more than distributed
random pixel flips of same magnitude.

RATIONALE: The order metric uses spatial features (edges, coherence, components).
Concentrated local changes should disrupt these spatial structures more than
diffuse changes affecting the same number of pixels.

METHODOLOGY:
1. Generate CPPN images spanning order range
2. For each image, apply two perturbation types with same # of pixels:
   a) Local patch: flip a contiguous square patch
   b) Distributed: flip same # of randomly scattered pixels
3. Measure |delta_order| for each perturbation type
4. Compare: does patch have larger effect than distributed?

SUCCESS CRITERIA:
- p < 0.01
- Cohen's d > 0.5
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative
)


def perturb_patch(img: np.ndarray, patch_size: int, seed: int = None) -> np.ndarray:
    """Flip a contiguous square patch of pixels."""
    rng = np.random.RandomState(seed)
    h, w = img.shape
    max_y = h - patch_size
    max_x = w - patch_size

    y = rng.randint(0, max_y + 1)
    x = rng.randint(0, max_x + 1)

    result = img.copy()
    result[y:y+patch_size, x:x+patch_size] = 1 - result[y:y+patch_size, x:x+patch_size]
    return result


def perturb_distributed(img: np.ndarray, num_pixels: int, seed: int = None) -> np.ndarray:
    """Flip randomly scattered pixels."""
    rng = np.random.RandomState(seed)
    h, w = img.shape
    flat = img.flatten()

    indices = rng.choice(h * w, size=num_pixels, replace=False)
    flat[indices] = 1 - flat[indices]

    return flat.reshape(h, w)


def run_experiment(
    n_images: int = 300,
    img_size: int = 64,
    patch_size: int = 8,  # 8x8 = 64 pixels
    n_perturbations_per_image: int = 10,
    seed: int = 42
):
    """Run the patch vs distributed perturbation experiment."""
    rng = np.random.RandomState(seed)
    num_pixels = patch_size * patch_size  # Same number of pixels for both methods

    patch_deltas = []
    distributed_deltas = []
    original_orders = []

    print(f"Generating {n_images} CPPN images at {img_size}x{img_size}...")
    print(f"Perturbation size: {num_pixels} pixels (patch: {patch_size}x{patch_size})")

    for i in range(n_images):
        # Generate CPPN image
        np.random.seed(rng.randint(0, 100000))
        cppn = CPPN()
        img = cppn.render(img_size)
        original_order = order_multiplicative(img)
        original_orders.append(original_order)

        # Apply multiple perturbations per image for stability
        for j in range(n_perturbations_per_image):
            p_seed = i * n_perturbations_per_image + j

            # Patch perturbation
            patch_img = perturb_patch(img, patch_size, seed=p_seed)
            patch_order = order_multiplicative(patch_img)
            patch_delta = abs(patch_order - original_order)
            patch_deltas.append(patch_delta)

            # Distributed perturbation (same # of pixels)
            dist_img = perturb_distributed(img, num_pixels, seed=p_seed + 100000)
            dist_order = order_multiplicative(dist_img)
            dist_delta = abs(dist_order - original_order)
            distributed_deltas.append(dist_delta)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_images} images...")

    # Convert to arrays
    patch_deltas = np.array(patch_deltas)
    distributed_deltas = np.array(distributed_deltas)
    original_orders = np.array(original_orders)

    # Statistics
    patch_mean = np.mean(patch_deltas)
    dist_mean = np.mean(distributed_deltas)

    # Paired t-test (same images, same # pixels, just different spatial distribution)
    t_stat, p_value = stats.ttest_rel(patch_deltas, distributed_deltas)

    # Cohen's d for paired samples
    diff = patch_deltas - distributed_deltas
    d = np.mean(diff) / np.std(diff, ddof=1)

    # Additional analysis: does effect depend on original order?
    # Group by order terciles
    order_terciles = np.percentile(original_orders, [33, 67])

    tercile_results = []
    for t_idx, (low, high) in enumerate([
        (0, order_terciles[0]),
        (order_terciles[0], order_terciles[1]),
        (order_terciles[1], 1.0)
    ]):
        mask = (original_orders >= low) & (original_orders < high) if t_idx < 2 else (original_orders >= low)
        mask_expanded = np.repeat(mask, n_perturbations_per_image)

        p_vals = patch_deltas[mask_expanded]
        d_vals = distributed_deltas[mask_expanded]

        if len(p_vals) > 0:
            t_t, p_t = stats.ttest_rel(p_vals, d_vals)
            diff_t = p_vals - d_vals
            d_t = np.mean(diff_t) / np.std(diff_t, ddof=1) if np.std(diff_t) > 0 else 0
            tercile_results.append({
                'tercile': ['low', 'mid', 'high'][t_idx],
                'n_images': int(np.sum(mask)),
                'patch_mean': float(np.mean(p_vals)),
                'distributed_mean': float(np.mean(d_vals)),
                'cohens_d': float(d_t),
                'p_value': float(p_t)
            })

    results = {
        'experiment': 'RES-164',
        'domain': 'metric_sensitivity',
        'hypothesis': 'Local patch perturbations affect order more than distributed random pixel flips',
        'parameters': {
            'n_images': n_images,
            'img_size': img_size,
            'patch_size': patch_size,
            'num_pixels_perturbed': num_pixels,
            'n_perturbations_per_image': n_perturbations_per_image,
            'total_samples': len(patch_deltas)
        },
        'results': {
            'patch_mean_delta': float(patch_mean),
            'distributed_mean_delta': float(dist_mean),
            'difference': float(patch_mean - dist_mean),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(d),
            'patch_median': float(np.median(patch_deltas)),
            'distributed_median': float(np.median(distributed_deltas)),
            'ratio_patch_to_distributed': float(patch_mean / dist_mean) if dist_mean > 0 else float('inf')
        },
        'tercile_analysis': tercile_results,
        'original_order_stats': {
            'mean': float(np.mean(original_orders)),
            'std': float(np.std(original_orders)),
            'min': float(np.min(original_orders)),
            'max': float(np.max(original_orders))
        }
    }

    # Determine status
    if p_value < 0.01 and abs(d) > 0.5:
        if d > 0:  # patch > distributed
            results['status'] = 'validated'
            results['conclusion'] = f'Patch perturbations cause {results["results"]["ratio_patch_to_distributed"]:.2f}x larger order changes than distributed (d={d:.2f}, p={p_value:.2e})'
        else:  # distributed > patch (opposite of hypothesis)
            results['status'] = 'refuted'
            results['conclusion'] = f'OPPOSITE: Distributed perturbations cause {1/results["results"]["ratio_patch_to_distributed"]:.2f}x larger order changes than patch (d={d:.2f}, p={p_value:.2e})'
    elif p_value < 0.01:
        results['status'] = 'inconclusive'
        results['conclusion'] = f'Significant (p={p_value:.2e}) but small effect (d={d:.2f})'
    else:
        results['status'] = 'refuted'
        results['conclusion'] = f'No significant difference (p={p_value:.2f}, d={d:.2f})'

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("RES-164: Patch vs Distributed Perturbation Sensitivity")
    print("=" * 60)

    results = run_experiment()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Patch perturbation mean |delta|: {results['results']['patch_mean_delta']:.4f}")
    print(f"Distributed perturbation mean |delta|: {results['results']['distributed_mean_delta']:.4f}")
    print(f"Ratio (patch/distributed): {results['results']['ratio_patch_to_distributed']:.2f}x")
    print(f"Cohen's d: {results['results']['cohens_d']:.3f}")
    print(f"p-value: {results['results']['p_value']:.2e}")
    print(f"\nStatus: {results['status'].upper()}")
    print(f"Conclusion: {results['conclusion']}")

    print("\nTercile Analysis (by original order):")
    for tr in results['tercile_analysis']:
        print(f"  {tr['tercile']}-order: d={tr['cohens_d']:.2f}, p={tr['p_value']:.2e}")

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'patch_vs_pixel_perturbation' / 'results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
