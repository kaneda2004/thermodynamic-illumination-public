"""
RES-204: Context depth saturation in CPPN vs random images.

Hypothesis: CPPN images reach compression saturation at smaller context
depth than matched random images.

Rationale:
- CPPN images have local structure (RES-003: 366x higher spatial MI)
- If structure is local, small context windows should capture it
- Random images may need larger context to exploit long-range correlations

Method:
1. Generate CPPN and matched smooth random images
2. Compute compression ratio using varying context sizes (k-th order entropy)
3. Find saturation point where additional context yields <5% improvement
4. Compare saturation depth between CPPN and random at matched compressibility
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, compute_compressibility, set_global_seed


def compute_kth_order_entropy(img: np.ndarray, k: int) -> float:
    """
    Compute k-th order entropy: entropy of pixel given k previous pixels.
    Uses raster scan order (left-to-right, top-to-bottom).

    For k=0: just marginal entropy H(X)
    For k=1: H(X|X-1) conditional on previous pixel
    For k>1: H(X|X-1, X-2, ..., X-k) conditional on k previous
    """
    flat = img.flatten()
    n = len(flat)

    if k == 0:
        # Marginal entropy
        p1 = np.mean(flat)
        if p1 == 0 or p1 == 1:
            return 0.0
        return -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))

    # Count context patterns and transitions
    # Context is k previous bits, target is current bit
    context_counts = {}  # context pattern -> [count_0, count_1]

    for i in range(k, n):
        # Get context as tuple of previous k bits
        context = tuple(flat[i-k:i])
        target = flat[i]

        if context not in context_counts:
            context_counts[context] = [0, 0]
        context_counts[context][target] += 1

    # Compute conditional entropy H(X|context)
    total_samples = n - k
    weighted_entropy = 0.0

    for context, counts in context_counts.items():
        n_context = sum(counts)
        p_context = n_context / total_samples

        # Entropy given this context
        p0 = counts[0] / n_context if n_context > 0 else 0
        p1 = counts[1] / n_context if n_context > 0 else 0

        if p0 > 0 and p1 > 0:
            h_given_context = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            h_given_context = 0.0

        weighted_entropy += p_context * h_given_context

    return weighted_entropy


def compute_2d_context_entropy(img: np.ndarray, radius: int) -> float:
    """
    Compute entropy given 2D spatial context of given radius.
    Context includes all pixels in a square neighborhood.
    More natural for images than 1D raster context.
    """
    h, w = img.shape

    # For each pixel, context is the surrounding pixels within radius
    # Use available neighbors only (handles boundaries)
    context_counts = {}

    for i in range(h):
        for j in range(w):
            # Collect context pixels (exclude self)
            context_pixels = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        # Only use pixels that come "before" in some ordering
                        # Use raster order: (ni < i) or (ni == i and nj < j)
                        if ni < i or (ni == i and nj < j):
                            context_pixels.append(img[ni, nj])

            if not context_pixels:
                continue

            context = tuple(context_pixels)
            target = img[i, j]

            if context not in context_counts:
                context_counts[context] = [0, 0]
            context_counts[context][target] += 1

    # Compute weighted entropy
    total = sum(sum(c) for c in context_counts.values())
    if total == 0:
        return 0.0

    weighted_entropy = 0.0
    for context, counts in context_counts.items():
        n_context = sum(counts)
        p_context = n_context / total

        p0 = counts[0] / n_context if n_context > 0 else 0
        p1 = counts[1] / n_context if n_context > 0 else 0

        if p0 > 0 and p1 > 0:
            h_given_context = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            h_given_context = 0.0

        weighted_entropy += p_context * h_given_context

    return weighted_entropy


def find_saturation_depth(entropies: list, threshold: float = 0.05) -> int:
    """
    Find the context depth at which entropy improvement saturates.
    Saturation = when relative improvement drops below threshold.

    Returns the k at which H(k) - H(k+1) < threshold * H(0)
    """
    if len(entropies) < 2:
        return 0

    base_entropy = entropies[0]
    if base_entropy == 0:
        return 0

    for k in range(len(entropies) - 1):
        improvement = entropies[k] - entropies[k + 1]
        relative_improvement = improvement / base_entropy
        if relative_improvement < threshold:
            return k

    return len(entropies) - 1  # Never saturated


def generate_smooth_random(size: int, sigma: float) -> np.ndarray:
    """Generate smooth random binary image via Gaussian blur."""
    continuous = np.random.randn(size, size)
    smoothed = gaussian_filter(continuous, sigma=sigma)
    return (smoothed > np.median(smoothed)).astype(np.uint8)


def main():
    set_global_seed(42)
    n_samples = 200
    size = 32
    context_depths = [0, 1, 2, 4, 8]  # k values to test

    print("=" * 70)
    print("RES-204: Context Depth Saturation Analysis")
    print("=" * 70)
    print(f"\nHypothesis: CPPN images reach compression saturation at smaller")
    print(f"context depth than matched random images.")
    print(f"\nSamples: {n_samples} per type, size: {size}x{size}")
    print(f"Context depths (k): {context_depths}")

    # Generate CPPN samples - oversample to filter for compressibility range
    print("\n--- Generating CPPN samples ---")
    cppn_data = []
    attempts = 0
    target_comp_range = (0.7, 0.95)  # Target range for matching

    while len(cppn_data) < n_samples and attempts < n_samples * 10:
        cppn = CPPN()
        img = cppn.render(size)
        comp = compute_compressibility(img)
        attempts += 1

        # Compute entropy at each context depth
        entropies = [compute_kth_order_entropy(img, k) for k in context_depths]
        sat_depth = find_saturation_depth(entropies)

        cppn_data.append({
            'comp': comp,
            'entropies': entropies,
            'sat_depth': sat_depth,
            'reduction_ratio': entropies[-1] / entropies[0] if entropies[0] > 0 else 1.0
        })

        if len(cppn_data) % 50 == 0:
            print(f"  {len(cppn_data)}/{n_samples}")

    # Generate smooth random samples with wider smoothness range to overlap compressibility
    print("\n--- Generating smooth random samples (wider sigma range) ---")
    random_data = []
    # Use larger sigmas to get higher compressibility
    sigmas = np.linspace(1.0, 8.0, n_samples * 2)  # More samples, wider range

    for i, sigma in enumerate(sigmas):
        img = generate_smooth_random(size, sigma)
        comp = compute_compressibility(img)

        entropies = [compute_kth_order_entropy(img, k) for k in context_depths]
        sat_depth = find_saturation_depth(entropies)

        random_data.append({
            'comp': comp,
            'entropies': entropies,
            'sat_depth': sat_depth,
            'reduction_ratio': entropies[-1] / entropies[0] if entropies[0] > 0 else 1.0
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sigmas)}")

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    cppn_sats = [d['sat_depth'] for d in cppn_data]
    rand_sats = [d['sat_depth'] for d in random_data]

    print(f"\nCPPN:")
    print(f"  Saturation depth: {np.mean(cppn_sats):.2f} +/- {np.std(cppn_sats):.2f}")
    print(f"  Compressibility: {np.mean([d['comp'] for d in cppn_data]):.3f}")

    print(f"\nSmooth Random:")
    print(f"  Saturation depth: {np.mean(rand_sats):.2f} +/- {np.std(rand_sats):.2f}")
    print(f"  Compressibility: {np.mean([d['comp'] for d in random_data]):.3f}")

    # Statistical test (overall)
    t_stat, p_overall = stats.ttest_ind(cppn_sats, rand_sats)
    pooled_std = np.sqrt((np.var(cppn_sats) + np.var(rand_sats)) / 2)
    d_overall = (np.mean(cppn_sats) - np.mean(rand_sats)) / pooled_std if pooled_std > 0 else 0

    print(f"\nOverall comparison:")
    print(f"  t-stat: {t_stat:.3f}, p: {p_overall:.4e}")
    print(f"  Cohen's d: {d_overall:.3f}")

    # Matched compressibility analysis
    print("\n" + "=" * 70)
    print("MATCHED COMPRESSIBILITY ANALYSIS")
    print("=" * 70)

    bins = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    matched_results = []

    for lo, hi in bins:
        cppn_bin = [d for d in cppn_data if lo <= d['comp'] < hi]
        rand_bin = [d for d in random_data if lo <= d['comp'] < hi]

        print(f"\nBin [{lo}, {hi}):")
        print(f"  CPPN: n={len(cppn_bin)}, Random: n={len(rand_bin)}")

        if len(cppn_bin) < 10 or len(rand_bin) < 10:
            print("  Insufficient samples")
            continue

        cppn_sat = [d['sat_depth'] for d in cppn_bin]
        rand_sat = [d['sat_depth'] for d in rand_bin]

        t, p = stats.ttest_ind(cppn_sat, rand_sat)
        pooled = np.sqrt((np.var(cppn_sat) + np.var(rand_sat)) / 2)
        d = (np.mean(cppn_sat) - np.mean(rand_sat)) / pooled if pooled > 0 else 0

        print(f"  CPPN sat_depth: {np.mean(cppn_sat):.2f} +/- {np.std(cppn_sat):.2f}")
        print(f"  Random sat_depth: {np.mean(rand_sat):.2f} +/- {np.std(rand_sat):.2f}")
        print(f"  t={t:.3f}, p={p:.4f}, d={d:.3f}")

        matched_results.append({
            'bin': f"[{lo}, {hi})",
            'cppn_mean': np.mean(cppn_sat),
            'rand_mean': np.mean(rand_sat),
            'p': p,
            'd': d
        })

    # Entropy decay curves
    print("\n" + "=" * 70)
    print("ENTROPY DECAY CURVES (mean across samples)")
    print("=" * 70)

    cppn_curves = np.array([d['entropies'] for d in cppn_data])
    rand_curves = np.array([d['entropies'] for d in random_data])

    print("\nContext depth vs mean entropy (bits/pixel):")
    print(f"{'k':<6} {'CPPN':>12} {'Random':>12} {'Diff':>12}")
    print("-" * 44)

    for i, k in enumerate(context_depths):
        cppn_mean = np.mean(cppn_curves[:, i])
        rand_mean = np.mean(rand_curves[:, i])
        diff = cppn_mean - rand_mean
        print(f"{k:<6} {cppn_mean:>12.4f} {rand_mean:>12.4f} {diff:>12.4f}")

    # Final reduction ratio comparison
    print("\n" + "=" * 70)
    print("FINAL REDUCTION RATIO (H_final / H_0)")
    print("=" * 70)

    cppn_ratios = [d['reduction_ratio'] for d in cppn_data]
    rand_ratios = [d['reduction_ratio'] for d in random_data]

    print(f"CPPN: {np.mean(cppn_ratios):.3f} +/- {np.std(cppn_ratios):.3f}")
    print(f"Random: {np.mean(rand_ratios):.3f} +/- {np.std(rand_ratios):.3f}")

    t_ratio, p_ratio = stats.ttest_ind(cppn_ratios, rand_ratios)
    d_ratio = (np.mean(cppn_ratios) - np.mean(rand_ratios)) / np.sqrt((np.var(cppn_ratios) + np.var(rand_ratios)) / 2)

    print(f"t={t_ratio:.3f}, p={p_ratio:.4e}, d={d_ratio:.3f}")

    # Spearman correlation: saturation depth vs compressibility
    print("\n" + "=" * 70)
    print("SATURATION DEPTH vs COMPRESSIBILITY CORRELATION")
    print("=" * 70)

    all_comps = [d['comp'] for d in cppn_data + random_data]
    all_sats = [d['sat_depth'] for d in cppn_data + random_data]

    rho, p_corr = stats.spearmanr(all_comps, all_sats)
    print(f"Spearman rho: {rho:.3f}, p: {p_corr:.4e}")

    # More continuous metric: entropy reduction from k=0 to k=1 (first step captures most info)
    print("\n" + "=" * 70)
    print("FIRST-STEP ENTROPY REDUCTION (H(0) - H(1)) / H(0)")
    print("=" * 70)

    cppn_first_step = [(d['entropies'][0] - d['entropies'][1]) / d['entropies'][0]
                       if d['entropies'][0] > 0 else 0 for d in cppn_data]
    rand_first_step = [(d['entropies'][0] - d['entropies'][1]) / d['entropies'][0]
                       if d['entropies'][0] > 0 else 0 for d in random_data]

    print(f"CPPN: {np.mean(cppn_first_step):.3f} +/- {np.std(cppn_first_step):.3f}")
    print(f"Random: {np.mean(rand_first_step):.3f} +/- {np.std(rand_first_step):.3f}")

    t_fs, p_fs = stats.ttest_ind(cppn_first_step, rand_first_step)
    d_fs = (np.mean(cppn_first_step) - np.mean(rand_first_step)) / np.sqrt(
        (np.var(cppn_first_step) + np.var(rand_first_step)) / 2)
    print(f"t={t_fs:.3f}, p={p_fs:.4e}, d={d_fs:.3f}")
    print("Interpretation: Higher = more info captured by first context bit")

    # VERDICT
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if CPPN saturates earlier (lower depth)
    sig_bins = [r for r in matched_results if r['p'] < 0.01 and abs(r['d']) > 0.5]

    if sig_bins:
        cppn_lower = sum(1 for r in sig_bins if r['cppn_mean'] < r['rand_mean'])
        if cppn_lower == len(sig_bins):
            status = "validated"
            verdict = "VALIDATED: CPPN images reach compression saturation at smaller context depth"
        elif cppn_lower == 0:
            status = "refuted"
            verdict = "REFUTED: CPPN images require LARGER context depth (opposite of hypothesis)"
        else:
            status = "inconclusive"
            verdict = "INCONCLUSIVE: Mixed results across compressibility bins"
    else:
        # Check overall result
        if p_overall < 0.01 and abs(d_overall) > 0.5:
            if d_overall < 0:
                status = "validated"
                verdict = "VALIDATED: CPPN saturates earlier overall (not controlling for compressibility)"
            else:
                status = "refuted"
                verdict = "REFUTED: Random saturates earlier overall"
        else:
            status = "refuted"
            verdict = "REFUTED: No significant difference in saturation depth"

    print(f"\n{verdict}")
    print(f"\nKey metrics:")
    print(f"  Overall effect size (d): {d_overall:.3f}")
    print(f"  Overall p-value: {p_overall:.4e}")
    print(f"  Matched bin results: {len(sig_bins)} significant out of {len(matched_results)}")

    return status, d_overall, p_overall, matched_results


if __name__ == "__main__":
    status, d, p, results = main()
