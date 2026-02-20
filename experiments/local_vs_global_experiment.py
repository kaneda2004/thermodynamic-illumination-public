"""
RES-017: Local vs Global Structure Decomposition

Hypothesis: Global structure (patch arrangement) contributes more to order than
local structure (within-patch patterns). Shuffling patches globally should
destroy more order than shuffling pixels within patches.

Method:
- Generate N CPPN images
- For each image, create two perturbations:
  A) Within-patch shuffle: randomly permute pixels within each patch (destroys local)
  B) Global patch shuffle: randomly permute patch positions (destroys global)
- Measure order drop: delta_local = order(original) - order(within_shuffle)
                      delta_global = order(original) - order(patch_shuffle)
- Compare delta_global vs delta_local using Mann-Whitney U and Cohen's d

Null hypothesis: delta_global == delta_local (local and global contribute equally)
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def shuffle_within_patches(img: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Shuffle pixels within each patch, preserving global patch arrangement.
    Destroys local structure, preserves global structure.
    """
    h, w = img.shape
    result = img.copy()

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Extract patch
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = result[i:i_end, j:j_end].flatten()
            # Shuffle pixels within patch
            np.random.shuffle(patch)
            result[i:i_end, j:j_end] = patch.reshape(i_end - i, j_end - j)

    return result


def shuffle_patches_globally(img: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Shuffle patch positions globally, preserving within-patch structure.
    Destroys global structure, preserves local structure.
    """
    h, w = img.shape

    # Ensure image is divisible by patch_size (pad if needed)
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    # Extract patches
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].copy()
            patches.append(patch)

    # Shuffle patch order
    np.random.shuffle(patches)

    # Reconstruct image
    result = np.zeros((n_patches_h * patch_size, n_patches_w * patch_size), dtype=img.dtype)
    idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            result[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
            idx += 1

    return result


def run_experiment(
    n_samples: int = 300,
    image_size: int = 32,
    patch_sizes: list = [4, 8],
    n_shuffle_reps: int = 5,  # Repeat shuffles to reduce noise
    seed: int = 42
) -> dict:
    """
    Run local vs global structure experiment.

    For each patch size, we test whether global shuffling destroys more order
    than local shuffling.
    """
    set_global_seed(seed)

    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'patch_sizes': patch_sizes,
        'n_shuffle_reps': n_shuffle_reps,
        'seed': seed,
        'by_patch_size': {}
    }

    # Generate CPPN images and compute original orders
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

    # Filter to images with meaningful order (>0.05) for cleaner comparison
    meaningful_mask = original_orders > 0.05
    n_meaningful = np.sum(meaningful_mask)
    print(f"Images with order > 0.05: {n_meaningful}/{n_samples}")

    for patch_size in patch_sizes:
        print(f"\nTesting patch_size={patch_size}...")

        delta_local = []  # Order drop from within-patch shuffle
        delta_global = []  # Order drop from global patch shuffle

        for idx in range(n_samples):
            if not meaningful_mask[idx]:
                continue

            img = images[idx]
            orig_order = original_orders[idx]

            # Average over multiple shuffle repetitions
            local_orders = []
            global_orders = []

            for _ in range(n_shuffle_reps):
                # Within-patch shuffle (destroys local)
                local_shuffled = shuffle_within_patches(img, patch_size)
                local_orders.append(order_multiplicative(local_shuffled))

                # Global patch shuffle (destroys global)
                global_shuffled = shuffle_patches_globally(img, patch_size)
                global_orders.append(order_multiplicative(global_shuffled))

            # Order drop (positive means order was destroyed)
            delta_local.append(orig_order - np.mean(local_orders))
            delta_global.append(orig_order - np.mean(global_orders))

        delta_local = np.array(delta_local)
        delta_global = np.array(delta_global)

        # Statistical tests
        # Mann-Whitney U: Is delta_global > delta_local?
        stat_mw, p_mw = stats.mannwhitneyu(delta_global, delta_local, alternative='greater')

        # Cohen's d
        pooled_std = np.sqrt((np.std(delta_global)**2 + np.std(delta_local)**2) / 2)
        cohens_d = (np.mean(delta_global) - np.mean(delta_local)) / (pooled_std + 1e-10)

        # Paired t-test (since same images)
        stat_t, p_t = stats.ttest_rel(delta_global, delta_local)

        # Wilcoxon signed-rank (non-parametric paired)
        stat_w, p_w = stats.wilcoxon(delta_global, delta_local, alternative='greater')

        # Fraction where global > local
        frac_global_larger = np.mean(delta_global > delta_local)

        result = {
            'n_meaningful': int(n_meaningful),
            'delta_local_mean': float(np.mean(delta_local)),
            'delta_local_std': float(np.std(delta_local)),
            'delta_global_mean': float(np.mean(delta_global)),
            'delta_global_std': float(np.std(delta_global)),
            'mann_whitney_U': float(stat_mw),
            'mann_whitney_p': float(p_mw),
            'cohens_d': float(cohens_d),
            'paired_t_stat': float(stat_t),
            'paired_t_p': float(p_t / 2),  # one-sided
            'wilcoxon_stat': float(stat_w),
            'wilcoxon_p': float(p_w),
            'frac_global_larger': float(frac_global_larger),
        }

        results['by_patch_size'][patch_size] = result

        print(f"  Delta local:  mean={np.mean(delta_local):.4f}, std={np.std(delta_local):.4f}")
        print(f"  Delta global: mean={np.mean(delta_global):.4f}, std={np.std(delta_global):.4f}")
        print(f"  Mann-Whitney p={p_mw:.4e}, Cohen's d={cohens_d:.3f}")
        print(f"  Fraction where global > local: {frac_global_larger:.2%}")

    return results


def main():
    """Run experiment and save results."""
    results = run_experiment(
        n_samples=300,
        image_size=32,
        patch_sizes=[4, 8, 16],
        n_shuffle_reps=5,
        seed=42
    )

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'local_vs_global'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = output_dir / 'local_global_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Local vs Global Structure Contribution to Order")
    print("="*60)

    for patch_size, data in results['by_patch_size'].items():
        print(f"\nPatch size {patch_size}x{patch_size}:")
        print(f"  Order drop from local shuffle:  {data['delta_local_mean']:.4f} +/- {data['delta_local_std']:.4f}")
        print(f"  Order drop from global shuffle: {data['delta_global_mean']:.4f} +/- {data['delta_global_std']:.4f}")
        print(f"  Cohen's d (global - local): {data['cohens_d']:.3f}")
        print(f"  Mann-Whitney p-value: {data['mann_whitney_p']:.4e}")
        print(f"  Wilcoxon signed-rank p-value: {data['wilcoxon_p']:.4e}")

        if data['wilcoxon_p'] < 0.01 and data['cohens_d'] > 0.5:
            print(f"  -> VALIDATED: Global structure contributes MORE to order")
        elif data['wilcoxon_p'] < 0.01 and data['cohens_d'] < -0.5:
            print(f"  -> REVERSED: Local structure contributes MORE to order")
        elif data['wilcoxon_p'] >= 0.01:
            print(f"  -> INCONCLUSIVE: No significant difference (p >= 0.01)")
        else:
            print(f"  -> INCONCLUSIVE: Effect size too small (|d| < 0.5)")

    return results


if __name__ == '__main__':
    main()
