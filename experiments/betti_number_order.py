"""
RES-188: Betti-1 (number of holes) negatively correlates with CPPN order

Hypothesis: High-order CPPN images have fewer topological holes (Betti-1 = 0 or 1)
because they tend toward simple, connected patterns rather than fragmented ones.

Betti-0 = number of connected components (already tested in RES-113)
Betti-1 = number of "holes" (enclosed regions of background)

Domain: topology_analysis
"""

import numpy as np
from scipy import stats, ndimage
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def count_holes(img: np.ndarray) -> int:
    """
    Count Betti-1: number of holes (enclosed background regions).

    For binary images, holes are background connected components that don't
    touch the boundary. We use 4-connectivity for background.
    """
    # Pad with background (0) to detect boundary-touching components
    padded = np.pad(img, 1, mode='constant', constant_values=0)

    # Invert: background becomes foreground
    background = 1 - padded

    # Label background connected components (4-connectivity)
    labeled, n_bg_components = ndimage.label(background, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))

    # The boundary-touching component is labeled as 1 (touches the padding)
    # Count components that don't touch the boundary = holes
    boundary_label = labeled[0, 0]  # Corner is part of the infinite background

    # Count unique labels that are not 0 (no component) and not the boundary label
    unique_labels = set(labeled.flatten()) - {0, boundary_label}

    return len(unique_labels)


def euler_characteristic(img: np.ndarray) -> int:
    """
    Compute Euler characteristic: chi = V - E + F for 2D grid.
    For binary images: chi = #components - #holes (Betti-0 - Betti-1)
    """
    # Count connected components (foreground)
    labeled_fg, n_fg = ndimage.label(img, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))

    # Count holes
    n_holes = count_holes(img)

    return n_fg - n_holes


def main():
    set_global_seed(42)
    n_samples = 1000

    orders = []
    betti_1s = []  # Number of holes
    betti_0s = []  # Number of connected components
    euler_chars = []

    print("Generating CPPN samples and computing Betti numbers...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute topological features
        labeled, n_components = ndimage.label(img, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        n_holes = count_holes(img)
        euler_char = n_components - n_holes

        orders.append(order)
        betti_0s.append(n_components)
        betti_1s.append(n_holes)
        euler_chars.append(euler_char)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples}")

    orders = np.array(orders)
    betti_0s = np.array(betti_0s)
    betti_1s = np.array(betti_1s)
    euler_chars = np.array(euler_chars)

    # Split into quartiles for effect size
    q25, q75 = np.percentile(orders, [25, 75])
    low_order_mask = orders <= q25
    high_order_mask = orders >= q75

    # Correlation tests
    r_betti1, p_betti1 = stats.pearsonr(orders, betti_1s)
    rho_betti1, p_rho_betti1 = stats.spearmanr(orders, betti_1s)

    r_betti0, p_betti0 = stats.pearsonr(orders, betti_0s)
    r_euler, p_euler = stats.pearsonr(orders, euler_chars)

    # Effect size (Cohen's d) for high vs low order quartiles
    high_betti1 = betti_1s[high_order_mask]
    low_betti1 = betti_1s[low_order_mask]

    pooled_std = np.sqrt((np.var(high_betti1, ddof=1) + np.var(low_betti1, ddof=1)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(high_betti1) - np.mean(low_betti1)) / pooled_std
    else:
        cohens_d = 0

    # t-test for quartile comparison
    t_stat, p_ttest = stats.ttest_ind(high_betti1, low_betti1)

    # Summary statistics
    print("\n" + "="*60)
    print("RESULTS: RES-188")
    print("="*60)
    print(f"\nSample size: n={n_samples}")
    print(f"\nBetti-1 (holes) distribution:")
    print(f"  Overall: mean={np.mean(betti_1s):.3f}, std={np.std(betti_1s):.3f}")
    print(f"  High-order (Q4): mean={np.mean(high_betti1):.3f}, std={np.std(high_betti1):.3f}")
    print(f"  Low-order (Q1): mean={np.mean(low_betti1):.3f}, std={np.std(low_betti1):.3f}")

    print(f"\nCorrelations with order:")
    print(f"  Betti-1 (holes): r={r_betti1:.4f}, p={p_betti1:.2e}, rho={rho_betti1:.4f}")
    print(f"  Betti-0 (components): r={r_betti0:.4f}, p={p_betti0:.2e}")
    print(f"  Euler char (B0-B1): r={r_euler:.4f}, p={p_euler:.2e}")

    print(f"\nQuartile comparison (high vs low order):")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  t-test p-value: {p_ttest:.2e}")

    # Distribution of Betti-1 values
    print(f"\nBetti-1 distribution:")
    for h in range(int(max(betti_1s)) + 1):
        count = np.sum(betti_1s == h)
        pct = 100 * count / len(betti_1s)
        print(f"  {h} holes: {count} ({pct:.1f}%)")

    # Betti-1 by order quartile
    print(f"\nBetti-1 breakdown by order quartile:")
    for q, label in [(orders <= q25, "Q1 (low)"), (orders >= q75, "Q4 (high)")]:
        subset = betti_1s[q]
        for h in range(int(max(subset)) + 1):
            count = np.sum(subset == h)
            pct = 100 * count / len(subset)
            if pct > 1:
                print(f"  {label}: {h} holes = {pct:.1f}%")

    # Validation criteria
    print("\n" + "="*60)
    print("VALIDATION CHECK")
    print("="*60)
    effect_ok = abs(cohens_d) > 0.5
    pval_ok = p_betti1 < 0.01
    direction_ok = r_betti1 < 0  # Hypothesis: NEGATIVE correlation

    print(f"  Effect size |d| > 0.5: {effect_ok} (d={cohens_d:.3f})")
    print(f"  p-value < 0.01: {pval_ok} (p={p_betti1:.2e})")
    print(f"  Negative correlation: {direction_ok} (r={r_betti1:.3f})")

    if effect_ok and pval_ok and direction_ok:
        status = "validated"
    elif pval_ok and direction_ok:
        status = "inconclusive"  # Significant but weak effect
    else:
        status = "refuted"

    print(f"\n  STATUS: {status.upper()}")

    # Output for log
    print("\n" + "="*60)
    print("LOG SUMMARY")
    print("="*60)
    if status == "validated":
        summary = f"Betti-1 negatively correlates with order (r={r_betti1:.3f}, d={cohens_d:.2f}, p={p_betti1:.2e}). High-order images have {np.mean(high_betti1):.2f} holes vs {np.mean(low_betti1):.2f} for low-order. Simple topology (fewer enclosed regions) characterizes structured CPPN outputs."
    elif r_betti1 > 0:
        summary = f"Betti-1 POSITIVELY correlates with order (r={r_betti1:.3f}, d={cohens_d:.2f}, p={p_betti1:.2e}). High-order images have MORE holes ({np.mean(high_betti1):.2f}) than low-order ({np.mean(low_betti1):.2f}). Opposite to hypothesis - structure requires topological complexity, not simplicity."
    else:
        summary = f"Betti-1 shows weak/no correlation with order (r={r_betti1:.3f}, d={cohens_d:.2f}, p={p_betti1:.2e}). High-order mean={np.mean(high_betti1):.2f}, low-order mean={np.mean(low_betti1):.2f}. Topological hole count does not distinguish structured from unstructured CPPN outputs."

    print(f"Status: {status}")
    print(f"Summary: {summary}")

    return {
        'status': status,
        'r': r_betti1,
        'rho': rho_betti1,
        'd': cohens_d,
        'p': p_betti1,
        'summary': summary
    }


if __name__ == '__main__':
    main()
