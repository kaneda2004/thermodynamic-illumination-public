"""
RES-082: Pixel entropy rate analysis - does bits/pixel distinguish CPPN from random?

Hypothesis: CPPN images have lower pixel entropy rate than random images at matched order.

Method:
1. Generate CPPN and random images
2. Match by compressibility (order proxy)
3. Compute block entropy at multiple block sizes to estimate entropy rate
4. Compare entropy rates at matched compressibility levels
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, compute_compressibility, set_global_seed

def compute_block_entropy(img: np.ndarray, block_size: int = 2) -> float:
    """Compute entropy of non-overlapping blocks."""
    h, w = img.shape
    # Trim to fit blocks
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    img_trim = img[:h_trim, :w_trim]

    # Reshape into blocks and convert to integers
    blocks = img_trim.reshape(h_trim // block_size, block_size,
                               w_trim // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

    # Convert each block to an integer (base-2 number)
    block_ints = np.array([int(''.join(map(str, b.flatten())), 2) for b in blocks])

    # Compute entropy of block distribution
    _, counts = np.unique(block_ints, return_counts=True)
    probs = counts / len(block_ints)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Normalize by block size to get bits per pixel
    return entropy / (block_size ** 2)

def compute_pixel_entropy(img: np.ndarray) -> float:
    """Simple single-pixel entropy."""
    p1 = np.mean(img)
    p0 = 1 - p1
    if p1 == 0 or p1 == 1:
        return 0.0
    return -(p1 * np.log2(p1) + p0 * np.log2(p0))

def compute_conditional_entropy(img: np.ndarray) -> float:
    """Entropy of pixel given its left neighbor (horizontal conditional entropy)."""
    # Count transitions
    left = img[:, :-1]
    right = img[:, 1:]

    # Joint counts
    n00 = np.sum((left == 0) & (right == 0))
    n01 = np.sum((left == 0) & (right == 1))
    n10 = np.sum((left == 1) & (right == 0))
    n11 = np.sum((left == 1) & (right == 1))

    total = n00 + n01 + n10 + n11

    # P(left=0), P(left=1)
    p_left0 = (n00 + n01) / total
    p_left1 = (n10 + n11) / total

    # P(right|left=0), P(right|left=1)
    if n00 + n01 > 0:
        p_r0_l0 = n00 / (n00 + n01)
        p_r1_l0 = n01 / (n00 + n01)
        h_l0 = -(p_r0_l0 * np.log2(p_r0_l0 + 1e-10) + p_r1_l0 * np.log2(p_r1_l0 + 1e-10))
    else:
        h_l0 = 0

    if n10 + n11 > 0:
        p_r0_l1 = n10 / (n10 + n11)
        p_r1_l1 = n11 / (n10 + n11)
        h_l1 = -(p_r0_l1 * np.log2(p_r0_l1 + 1e-10) + p_r1_l1 * np.log2(p_r1_l1 + 1e-10))
    else:
        h_l1 = 0

    # H(right|left) = sum_l P(l) H(right|left=l)
    return p_left0 * h_l0 + p_left1 * h_l1

def generate_smooth_random(size: int = 32, sigma: float = 2.0) -> np.ndarray:
    """Generate smooth random image via Gaussian blur then threshold."""
    continuous = np.random.randn(size, size)
    smoothed = gaussian_filter(continuous, sigma=sigma)
    return (smoothed > np.median(smoothed)).astype(np.uint8)

def main():
    set_global_seed(42)
    n_samples = 200
    size = 32

    print("=" * 60)
    print("RES-082: Pixel Entropy Rate Analysis")
    print("=" * 60)

    # Generate samples
    cppn_data = []
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size)
        comp = compute_compressibility(img)
        h1 = compute_pixel_entropy(img)
        h_cond = compute_conditional_entropy(img)
        h_block2 = compute_block_entropy(img, 2)
        h_block4 = compute_block_entropy(img, 4)
        cppn_data.append({'comp': comp, 'h1': h1, 'h_cond': h_cond,
                          'h_block2': h_block2, 'h_block4': h_block4})

    # Generate matched random samples with varying smoothness
    random_data = []
    for sigma in np.linspace(0.5, 4.0, n_samples):
        img = generate_smooth_random(size, sigma)
        comp = compute_compressibility(img)
        h1 = compute_pixel_entropy(img)
        h_cond = compute_conditional_entropy(img)
        h_block2 = compute_block_entropy(img, 2)
        h_block4 = compute_block_entropy(img, 4)
        random_data.append({'comp': comp, 'h1': h1, 'h_cond': h_cond,
                            'h_block2': h_block2, 'h_block4': h_block4})

    # Also pure uniform random for baseline
    uniform_data = []
    for _ in range(n_samples):
        img = (np.random.rand(size, size) > 0.5).astype(np.uint8)
        comp = compute_compressibility(img)
        h1 = compute_pixel_entropy(img)
        h_cond = compute_conditional_entropy(img)
        h_block2 = compute_block_entropy(img, 2)
        h_block4 = compute_block_entropy(img, 4)
        uniform_data.append({'comp': comp, 'h1': h1, 'h_cond': h_cond,
                              'h_block2': h_block2, 'h_block4': h_block4})

    # Summary statistics
    print("\n--- Overall Statistics ---")
    for name, data in [("CPPN", cppn_data), ("Smooth Random", random_data), ("Uniform Random", uniform_data)]:
        comps = [d['comp'] for d in data]
        h_conds = [d['h_cond'] for d in data]
        print(f"\n{name}:")
        print(f"  Compressibility: {np.mean(comps):.3f} +/- {np.std(comps):.3f}")
        print(f"  Conditional entropy: {np.mean(h_conds):.3f} +/- {np.std(h_conds):.3f}")

    # Match by compressibility bins and compare entropy
    print("\n--- Matched Compressibility Analysis ---")
    bins = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    results = []
    for lo, hi in bins:
        cppn_in_bin = [d for d in cppn_data if lo <= d['comp'] < hi]
        rand_in_bin = [d for d in random_data if lo <= d['comp'] < hi]

        if len(cppn_in_bin) < 10 or len(rand_in_bin) < 10:
            print(f"\nBin [{lo}, {hi}): Insufficient samples (CPPN: {len(cppn_in_bin)}, Random: {len(rand_in_bin)})")
            continue

        cppn_h = [d['h_cond'] for d in cppn_in_bin]
        rand_h = [d['h_cond'] for d in rand_in_bin]

        t_stat, p_val = stats.ttest_ind(cppn_h, rand_h)
        effect_size = (np.mean(cppn_h) - np.mean(rand_h)) / np.sqrt((np.var(cppn_h) + np.var(rand_h)) / 2)

        print(f"\nBin [{lo}, {hi}):")
        print(f"  CPPN (n={len(cppn_in_bin)}): H_cond = {np.mean(cppn_h):.4f} +/- {np.std(cppn_h):.4f}")
        print(f"  Random (n={len(rand_in_bin)}): H_cond = {np.mean(rand_h):.4f} +/- {np.std(rand_h):.4f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Cohen's d: {effect_size:.3f}")

        results.append({
            'bin': f"[{lo}, {hi})",
            'cppn_mean': np.mean(cppn_h),
            'rand_mean': np.mean(rand_h),
            'p_val': p_val,
            'effect_size': effect_size,
            'n_cppn': len(cppn_in_bin),
            'n_rand': len(rand_in_bin)
        })

    # Block entropy comparison
    print("\n--- Block Entropy (2x2) Comparison ---")
    cppn_h2 = [d['h_block2'] for d in cppn_data]
    rand_h2 = [d['h_block2'] for d in random_data]
    uniform_h2 = [d['h_block2'] for d in uniform_data]

    print(f"CPPN: {np.mean(cppn_h2):.4f} +/- {np.std(cppn_h2):.4f}")
    print(f"Smooth Random: {np.mean(rand_h2):.4f} +/- {np.std(rand_h2):.4f}")
    print(f"Uniform Random: {np.mean(uniform_h2):.4f} +/- {np.std(uniform_h2):.4f}")

    # Correlation analysis: entropy rate vs compressibility
    print("\n--- Entropy-Compressibility Correlation ---")
    for name, data in [("CPPN", cppn_data), ("Smooth Random", random_data)]:
        comps = [d['comp'] for d in data]
        h_conds = [d['h_cond'] for d in data]
        r, p = stats.pearsonr(comps, h_conds)
        print(f"{name}: r={r:.3f}, p={p:.4f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Check if any bin shows significant difference
    sig_results = [r for r in results if r['p_val'] < 0.01 and abs(r['effect_size']) > 0.5]

    if sig_results:
        # Check direction
        cppn_lower = sum(1 for r in sig_results if r['cppn_mean'] < r['rand_mean'])
        if cppn_lower == len(sig_results):
            print("VALIDATED: CPPN has significantly lower entropy rate than random at matched order")
            status = "VALIDATED"
        elif cppn_lower == 0:
            print("REFUTED: CPPN has HIGHER entropy rate than random at matched order")
            status = "REFUTED"
        else:
            print("INCONCLUSIVE: Mixed results across bins")
            status = "INCONCLUSIVE"
    else:
        print("REFUTED: No significant difference in entropy rate at matched compressibility")
        status = "REFUTED"

    # Key metrics for log
    avg_effect = np.mean([r['effect_size'] for r in results]) if results else 0
    min_p = min([r['p_val'] for r in results]) if results else 1.0

    print(f"\nKey metrics: avg_effect_size={avg_effect:.3f}, min_p={min_p:.4f}")
    print(f"Status: {status}")

    return status, avg_effect, min_p

if __name__ == "__main__":
    status, effect, p_val = main()
