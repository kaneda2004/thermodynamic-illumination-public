"""
RES-160: High-order CPPN images have longer skeleton branch lengths than low-order images

Hypothesis: High-order images have more extended, coherent structures that produce
longer skeleton branches, while low-order images have fragmented patterns with shorter branches.

Tests the morphological skeleton (medial axis) properties of CPPN-generated binary images.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import ndimage
from scipy.stats import pearsonr, ttest_ind
import json
from pathlib import Path


def skeletonize(img: np.ndarray) -> np.ndarray:
    """Simple hit-or-miss skeletonization."""
    skeleton = np.zeros_like(img, dtype=bool)
    element = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)

    img_copy = img.astype(bool).copy()

    # Iterative thinning
    for _ in range(100):  # Max iterations
        eroded = ndimage.binary_erosion(img_copy, element)
        opened = ndimage.binary_dilation(eroded, element)
        skeleton = skeleton | (img_copy & ~opened)
        img_copy = eroded
        if not np.any(img_copy):
            break

    return skeleton.astype(np.uint8)


def get_skeleton_stats(img: np.ndarray) -> dict:
    """Compute skeleton morphology statistics."""
    skeleton = skeletonize(img)

    # Count skeleton pixels
    skel_pixels = np.sum(skeleton)

    if skel_pixels == 0:
        return {
            'skeleton_length': 0,
            'branch_count': 0,
            'mean_branch_length': 0,
            'endpoint_count': 0,
            'junction_count': 0,
        }

    # Find endpoints and junctions using neighbor count
    padded = np.pad(skeleton, 1, mode='constant', constant_values=0)
    neighbors = np.zeros_like(skeleton, dtype=int)

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted = padded[1+di:1+di+skeleton.shape[0], 1+dj:1+dj+skeleton.shape[1]]
            neighbors += shifted

    # Endpoints: skeleton pixels with 1 neighbor
    endpoints = skeleton & (neighbors == 1)
    endpoint_count = np.sum(endpoints)

    # Junctions: skeleton pixels with 3+ neighbors
    junctions = skeleton & (neighbors >= 3)
    junction_count = np.sum(junctions)

    # Estimate branch count from endpoints and junctions
    # branches = (endpoints + 2*junctions) / 2 for tree-like structures
    branch_count = max(1, (endpoint_count + 2 * junction_count) // 2)

    # Mean branch length = skeleton_length / branch_count
    mean_branch_length = skel_pixels / max(1, branch_count)

    return {
        'skeleton_length': int(skel_pixels),
        'branch_count': int(branch_count),
        'mean_branch_length': float(mean_branch_length),
        'endpoint_count': int(endpoint_count),
        'junction_count': int(junction_count),
    }


def main():
    np.random.seed(42)

    n_samples = 500
    results = []

    print("Generating CPPN samples and computing skeleton morphology...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        stats = get_skeleton_stats(img)
        stats['order'] = order
        results.append(stats)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples}")

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    branch_lengths = np.array([r['mean_branch_length'] for r in results])
    skeleton_lengths = np.array([r['skeleton_length'] for r in results])
    branch_counts = np.array([r['branch_count'] for r in results])

    # Remove zeros for correlation (images with no foreground)
    valid = skeleton_lengths > 0
    orders_valid = orders[valid]
    branch_lengths_valid = branch_lengths[valid]

    # Correlation: order vs mean branch length
    r_branch, p_branch = pearsonr(orders_valid, branch_lengths_valid)

    # Correlation: order vs skeleton length
    r_skel, p_skel = pearsonr(orders_valid, skeleton_lengths[valid])

    # High vs low order comparison
    median_order = np.median(orders_valid)
    high_order = orders_valid > median_order
    low_order = orders_valid <= median_order

    high_branch = branch_lengths_valid[high_order]
    low_branch = branch_lengths_valid[low_order]

    t_stat, t_pval = ttest_ind(high_branch, low_branch)

    # Cohen's d
    pooled_std = np.sqrt((np.var(high_branch) + np.var(low_branch)) / 2)
    cohens_d = (np.mean(high_branch) - np.mean(low_branch)) / (pooled_std + 1e-10)

    output = {
        'n_samples': n_samples,
        'n_valid': int(np.sum(valid)),
        'correlation_branch_length': {
            'r': float(r_branch),
            'p': float(p_branch),
        },
        'correlation_skeleton_length': {
            'r': float(r_skel),
            'p': float(p_skel),
        },
        'high_vs_low_order': {
            'high_mean_branch_length': float(np.mean(high_branch)),
            'low_mean_branch_length': float(np.mean(low_branch)),
            't_stat': float(t_stat),
            'p_value': float(t_pval),
            'cohens_d': float(cohens_d),
        },
        'summary_stats': {
            'mean_order': float(np.mean(orders)),
            'mean_branch_length': float(np.mean(branch_lengths_valid)),
            'mean_skeleton_length': float(np.mean(skeleton_lengths[valid])),
        }
    }

    # Save results
    out_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/res_160_skeleton_morphology')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("RES-160: Skeleton Branch Morphology Results")
    print("="*60)
    print(f"\nValid samples: {output['n_valid']}/{n_samples}")
    print(f"\nCorrelation (order vs branch length): r={r_branch:.3f}, p={p_branch:.2e}")
    print(f"Correlation (order vs skeleton length): r={r_skel:.3f}, p={p_skel:.2e}")
    print(f"\nHigh-order mean branch length: {np.mean(high_branch):.2f}")
    print(f"Low-order mean branch length: {np.mean(low_branch):.2f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"t-test p-value: {t_pval:.2e}")

    # Determine verdict
    if abs(cohens_d) >= 0.5 and p_branch < 0.01:
        if cohens_d > 0:
            print("\n>>> VALIDATED: High-order images have longer skeleton branches")
        else:
            print("\n>>> REFUTED: High-order images have SHORTER skeleton branches")
    elif p_branch < 0.01:
        print("\n>>> INCONCLUSIVE: Significant p but small effect size")
    else:
        print("\n>>> REFUTED: No significant correlation found")

    return output


if __name__ == '__main__':
    main()
