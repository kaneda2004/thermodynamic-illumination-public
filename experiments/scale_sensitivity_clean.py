"""
RES-017: Scale-Dependent Structure in Order Metric

REFINED Hypothesis: The order metric exhibits non-monotonic sensitivity to
perturbation scale, with maximum sensitivity at an intermediate "characteristic
scale" (~1/4 image size).

Key observation: Shuffling pixels at ANY scale INCREASES order for most images.
This reveals that the order metric measures "textural complexity" rather than
"preservation of original structure".

The research question becomes: At what scale does perturbation have the
STRONGEST effect (positive or negative)?

Method:
- Generate images across order spectrum
- Apply scale-specific perturbations
- Measure |order_change| = |order_after - order_before|
- Test if |order_change| depends non-monotonically on scale
- Identify characteristic scale

Null: |order_change| is independent of perturbation scale
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
    n_samples: int = 400,
    image_size: int = 32,
    patch_sizes: list = [2, 4, 8, 16],
    n_shuffle_reps: int = 10,
    seed: int = 42
) -> dict:
    """
    Test scale dependence of order metric sensitivity.
    """
    set_global_seed(seed)

    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'patch_sizes': patch_sizes,
        'n_shuffle_reps': n_shuffle_reps,
        'seed': seed,
    }

    # Generate CPPN images (full spectrum)
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
    print(f"Order range: [{np.min(original_orders):.4f}, {np.max(original_orders):.4f}]")

    results['original_order_mean'] = float(np.mean(original_orders))
    results['original_order_std'] = float(np.std(original_orders))

    # For each patch size, measure order change magnitude
    abs_order_changes = {ps: [] for ps in patch_sizes}
    order_directions = {ps: [] for ps in patch_sizes}  # +1 if increased, -1 if decreased
    final_orders = {ps: [] for ps in patch_sizes}

    for idx, (img, orig_order) in enumerate(zip(images, original_orders)):
        for patch_size in patch_sizes:
            changes = []
            finals = []
            for _ in range(n_shuffle_reps):
                shuffled = shuffle_within_patches(img.copy(), patch_size)
                shuffled_order = order_multiplicative(shuffled)
                changes.append(shuffled_order - orig_order)
                finals.append(shuffled_order)

            mean_change = np.mean(changes)
            abs_order_changes[patch_size].append(np.abs(mean_change))
            order_directions[patch_size].append(1 if mean_change > 0 else -1)
            final_orders[patch_size].append(np.mean(finals))

    # Analyze by patch size
    results['by_patch_size'] = {}

    for patch_size in patch_sizes:
        abs_changes = np.array(abs_order_changes[patch_size])
        directions = np.array(order_directions[patch_size])
        finals = np.array(final_orders[patch_size])

        results['by_patch_size'][patch_size] = {
            'mean_abs_change': float(np.mean(abs_changes)),
            'std_abs_change': float(np.std(abs_changes)),
            'frac_increase': float(np.mean(directions > 0)),
            'frac_decrease': float(np.mean(directions < 0)),
            'final_order_mean': float(np.mean(finals)),
            'final_order_std': float(np.std(finals)),
        }

        print(f"\nPatch size {patch_size}x{patch_size}:")
        print(f"  |Order change|: {np.mean(abs_changes):.4f} +/- {np.std(abs_changes):.4f}")
        print(f"  Order increased: {np.mean(directions > 0)*100:.1f}%, decreased: {np.mean(directions < 0)*100:.1f}%")
        print(f"  Final order: {np.mean(finals):.4f} +/- {np.std(finals):.4f}")

    # Kruskal-Wallis on |order_change|
    all_abs_changes = [np.array(abs_order_changes[ps]) for ps in patch_sizes]
    stat_kw, p_kw = stats.kruskal(*all_abs_changes)

    results['kruskal_wallis_H'] = float(stat_kw)
    results['kruskal_wallis_p'] = float(p_kw)

    print(f"\nKruskal-Wallis H={stat_kw:.2f}, p={p_kw:.4e}")

    # Find characteristic scale (max |change|)
    mean_changes = [results['by_patch_size'][ps]['mean_abs_change'] for ps in patch_sizes]
    max_idx = np.argmax(mean_changes)
    char_scale = patch_sizes[max_idx]

    results['characteristic_scale'] = int(char_scale)
    results['characteristic_scale_effect'] = float(mean_changes[max_idx])

    print(f"\nCharacteristic scale: {char_scale}x{char_scale} (|change|={mean_changes[max_idx]:.4f})")

    # Test if characteristic scale is significantly different from others
    char_changes = np.array(abs_order_changes[char_scale])
    other_changes = np.concatenate([np.array(abs_order_changes[ps]) for ps in patch_sizes if ps != char_scale])

    stat_u, p_u = stats.mannwhitneyu(char_changes, other_changes, alternative='greater')
    pooled_std = np.sqrt((np.std(char_changes)**2 + np.std(other_changes)**2) / 2)
    cohens_d = (np.mean(char_changes) - np.mean(other_changes)) / (pooled_std + 1e-10)

    results['char_vs_others_U'] = float(stat_u)
    results['char_vs_others_p'] = float(p_u)
    results['char_vs_others_d'] = float(cohens_d)

    print(f"Characteristic scale vs others: U={stat_u:.0f}, p={p_u:.4e}, d={cohens_d:.3f}")

    # Test pairwise adjacent scales
    results['pairwise_comparisons'] = []
    for i in range(len(patch_sizes) - 1):
        ps1, ps2 = patch_sizes[i], patch_sizes[i+1]
        changes1 = np.array(abs_order_changes[ps1])
        changes2 = np.array(abs_order_changes[ps2])

        stat_w, p_w = stats.wilcoxon(changes1, changes2, alternative='two-sided')
        pooled_std = np.sqrt((np.std(changes1)**2 + np.std(changes2)**2) / 2)
        cohens_d = (np.mean(changes1) - np.mean(changes2)) / (pooled_std + 1e-10)

        results['pairwise_comparisons'].append({
            'scale_pair': [ps1, ps2],
            'wilcoxon_p': float(p_w),
            'cohens_d': float(cohens_d),
        })

    return results


def main():
    results = run_experiment(
        n_samples=400,
        image_size=32,
        patch_sizes=[2, 4, 8, 16],
        n_shuffle_reps=10,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'scale_dependence'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'scale_sensitivity_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Scale-Dependent Structure Sensitivity")
    print("="*70)

    print(f"\n1. SCALE DEPENDENCE EXISTS:")
    print(f"   Kruskal-Wallis H = {results['kruskal_wallis_H']:.2f}, p = {results['kruskal_wallis_p']:.4e}")
    if results['kruskal_wallis_p'] < 0.01:
        print(f"   -> VALIDATED (p < 0.01)")

    char = results['characteristic_scale']
    print(f"\n2. CHARACTERISTIC SCALE: {char}x{char} pixels")
    print(f"   (1/{results['image_size']//char} of image side length)")
    print(f"   Effect size at char scale: {results['characteristic_scale_effect']:.4f}")

    print(f"\n3. CHAR SCALE VS OTHERS:")
    print(f"   Mann-Whitney p = {results['char_vs_others_p']:.4e}")
    print(f"   Cohen's d = {results['char_vs_others_d']:.3f}")
    if results['char_vs_others_p'] < 0.01 and results['char_vs_others_d'] > 0.5:
        print(f"   -> VALIDATED: Characteristic scale has larger effect")

    print(f"\n4. DIRECTION OF CHANGE:")
    for ps in results['by_patch_size']:
        data = results['by_patch_size'][ps]
        print(f"   {ps}x{ps}: {data['frac_increase']*100:.0f}% increase, {data['frac_decrease']*100:.0f}% decrease")

    return results


if __name__ == '__main__':
    main()
