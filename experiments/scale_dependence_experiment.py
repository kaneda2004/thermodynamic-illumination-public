"""
RES-017: Scale Dependence of Order Structure

Hypothesis: Order metric sensitivity exhibits scale dependence - local perturbations
at different scales have different effects on order, revealing a characteristic
scale at which structure is most important.

Key insight from preliminary experiment:
- Low-order images gain order from shuffling (randomization creates texture)
- The gain depends on scale: 4x4 patch shuffle increases order more than 8x8 or 16x16

This experiment:
1. Generate high-order CPPN images (order > 0.2) where shuffling DECREASES order
2. Test multiple perturbation scales
3. Find the "critical scale" where perturbation has maximum effect

Method:
- Generate CPPN images, filter to high-order (>0.2)
- Apply pixel shuffling within patches of size 2, 4, 8, 16
- Measure order DROP (since these are high-order images)
- Test if effect size varies systematically with scale

Null hypothesis: Order drop is independent of perturbation scale
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def shuffle_within_patches(img: np.ndarray, patch_size: int) -> np.ndarray:
    """Shuffle pixels within each patch."""
    h, w = img.shape
    result = img.copy()

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = result[i:i_end, j:j_end].flatten()
            np.random.shuffle(patch)
            result[i:i_end, j:j_end] = patch.reshape(i_end - i, j_end - j)

    return result


def run_experiment(
    n_samples: int = 500,
    image_size: int = 32,
    patch_sizes: list = [2, 4, 8, 16],
    order_threshold: float = 0.15,
    n_shuffle_reps: int = 10,
    seed: int = 42
) -> dict:
    """
    Test scale dependence of order sensitivity.
    """
    set_global_seed(seed)

    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'patch_sizes': patch_sizes,
        'order_threshold': order_threshold,
        'n_shuffle_reps': n_shuffle_reps,
        'seed': seed,
    }

    # Generate CPPN images until we have enough high-order ones
    print(f"Generating CPPN images (filtering to order > {order_threshold})...")
    images = []
    original_orders = []
    attempts = 0

    while len(images) < n_samples and attempts < n_samples * 20:
        cppn = CPPN()
        img = cppn.render(size=image_size)
        order = order_multiplicative(img)
        attempts += 1

        if order > order_threshold:
            images.append(img)
            original_orders.append(order)

        if attempts % 500 == 0:
            print(f"  {len(images)}/{n_samples} high-order images found after {attempts} attempts")

    original_orders = np.array(original_orders)
    n_found = len(images)
    print(f"Found {n_found} images with order > {order_threshold} ({n_found/attempts*100:.1f}% yield)")
    print(f"Original order: mean={np.mean(original_orders):.4f}, std={np.std(original_orders):.4f}")

    results['n_found'] = n_found
    results['original_order_mean'] = float(np.mean(original_orders))
    results['original_order_std'] = float(np.std(original_orders))

    # Test each patch size
    order_drops = {ps: [] for ps in patch_sizes}

    for idx, (img, orig_order) in enumerate(zip(images, original_orders)):
        for patch_size in patch_sizes:
            drops = []
            for _ in range(n_shuffle_reps):
                shuffled = shuffle_within_patches(img, patch_size)
                shuffled_order = order_multiplicative(shuffled)
                drops.append(orig_order - shuffled_order)
            order_drops[patch_size].append(np.mean(drops))

    # Convert to arrays and analyze
    results['by_patch_size'] = {}

    for patch_size in patch_sizes:
        drops = np.array(order_drops[patch_size])

        results['by_patch_size'][patch_size] = {
            'mean_drop': float(np.mean(drops)),
            'std_drop': float(np.std(drops)),
            'median_drop': float(np.median(drops)),
            'iqr_drop': float(stats.iqr(drops)),
            'frac_positive': float(np.mean(drops > 0)),  # Fraction where order decreased
        }

        print(f"\nPatch size {patch_size}x{patch_size}:")
        print(f"  Order drop: {np.mean(drops):.4f} +/- {np.std(drops):.4f}")
        print(f"  Fraction where shuffling reduced order: {np.mean(drops > 0)*100:.1f}%")

    # Pairwise comparisons between adjacent scales
    results['pairwise_comparisons'] = []

    for i in range(len(patch_sizes) - 1):
        ps1, ps2 = patch_sizes[i], patch_sizes[i+1]
        drops1 = np.array(order_drops[ps1])
        drops2 = np.array(order_drops[ps2])

        # Mann-Whitney U
        stat_u, p_u = stats.mannwhitneyu(drops1, drops2, alternative='two-sided')

        # Paired Wilcoxon (same images)
        stat_w, p_w = stats.wilcoxon(drops1, drops2, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((np.std(drops1)**2 + np.std(drops2)**2) / 2)
        cohens_d = (np.mean(drops1) - np.mean(drops2)) / (pooled_std + 1e-10)

        results['pairwise_comparisons'].append({
            'scale_pair': [ps1, ps2],
            'mann_whitney_p': float(p_u),
            'wilcoxon_p': float(p_w),
            'cohens_d': float(cohens_d),
        })

        print(f"\n{ps1}x{ps1} vs {ps2}x{ps2}: Cohen's d={cohens_d:.3f}, Wilcoxon p={p_w:.4e}")

    # Kruskal-Wallis across all scales
    all_drops = [np.array(order_drops[ps]) for ps in patch_sizes]
    stat_kw, p_kw = stats.kruskal(*all_drops)

    results['kruskal_wallis_H'] = float(stat_kw)
    results['kruskal_wallis_p'] = float(p_kw)

    print(f"\nKruskal-Wallis H={stat_kw:.2f}, p={p_kw:.4e}")

    # Test for monotonic trend (Jonckheere-Terpstra proxy using Spearman)
    # Correlate patch_size with mean_drop
    mean_drops = [np.mean(order_drops[ps]) for ps in patch_sizes]
    spearman_r, spearman_p = stats.spearmanr(patch_sizes, mean_drops)

    results['scale_trend_spearman_r'] = float(spearman_r)
    results['scale_trend_spearman_p'] = float(spearman_p)

    print(f"\nScale trend: Spearman r={spearman_r:.3f}, p={spearman_p:.4e}")

    return results


def main():
    results = run_experiment(
        n_samples=300,
        image_size=32,
        patch_sizes=[2, 4, 8, 16],
        order_threshold=0.15,
        n_shuffle_reps=10,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'scale_dependence'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'scale_dependence_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Scale Dependence of Order Sensitivity")
    print("="*60)

    print(f"\nKruskal-Wallis test (is there scale dependence?):")
    print(f"  H = {results['kruskal_wallis_H']:.2f}, p = {results['kruskal_wallis_p']:.4e}")

    if results['kruskal_wallis_p'] < 0.01:
        print("  -> SIGNIFICANT: Order sensitivity depends on perturbation scale")
    else:
        print("  -> NOT SIGNIFICANT: No scale dependence detected")

    print(f"\nMonotonic trend (does effect increase/decrease with scale?):")
    print(f"  Spearman r = {results['scale_trend_spearman_r']:.3f}, p = {results['scale_trend_spearman_p']:.4e}")

    if results['scale_trend_spearman_p'] < 0.01:
        if results['scale_trend_spearman_r'] > 0:
            print("  -> VALIDATED: Larger scales cause MORE order drop")
        else:
            print("  -> VALIDATED: Smaller scales cause MORE order drop")

    # Find max effect scale
    max_drop = 0
    max_scale = 0
    for ps, data in results['by_patch_size'].items():
        if data['mean_drop'] > max_drop:
            max_drop = data['mean_drop']
            max_scale = ps

    print(f"\nCharacteristic scale (max sensitivity): {max_scale}x{max_scale} (drop={max_drop:.4f})")

    return results


if __name__ == '__main__':
    main()
