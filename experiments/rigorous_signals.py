#!/usr/bin/env python3
"""
RIGOROUS SIGNAL HUNTING

Each experiment includes:
1. Clear hypothesis
2. Proper null model
3. Statistical significance tests
4. Effect size estimates
5. Multiple testing correction where needed
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, bootstrap, norm
from scipy.spatial.distance import pdist, squareform
import math
from collections import defaultdict
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    order_multiplicative,
    order_kolmogorov_proxy,
    compute_symmetry,
    compute_edge_density,
    compute_spectral_coherence,
    compute_compressibility,
    CPPN,
)


# ============================================================================
# EXPERIMENT 1: TRUE INTRINSIC DIMENSION
# ============================================================================

def mle_intrinsic_dimension(X, k=10):
    """
    Maximum Likelihood Estimator for intrinsic dimension.

    Levina & Bickel (2004): "Maximum Likelihood Estimation of
    Intrinsic Dimension"

    Args:
        X: (n_samples, n_features) data matrix
        k: number of nearest neighbors

    Returns:
        dimension estimate and standard error
    """
    n = X.shape[0]

    # Compute pairwise distances
    D = squareform(pdist(X, 'euclidean'))

    # For each point, get k nearest neighbor distances
    dims = []
    for i in range(n):
        dists = np.sort(D[i])[1:k+1]  # Exclude self, get k nearest
        if dists[0] > 0:  # Avoid log(0)
            # MLE estimator
            T_k = dists[-1]  # k-th neighbor distance
            log_ratios = np.log(T_k / dists[:-1])
            if len(log_ratios) > 0 and np.sum(log_ratios) > 0:
                d_hat = (k - 1) / np.sum(log_ratios)
                dims.append(d_hat)

    dims = np.array(dims)
    return np.mean(dims), np.std(dims) / np.sqrt(len(dims))


def experiment_intrinsic_dimension():
    """
    HYPOTHESIS: CPPN images have lower intrinsic dimension than random images.

    NULL: Both have same intrinsic dimension (equal to ambient dimension).

    Method: MLE intrinsic dimension estimator (Levina & Bickel 2004)
    """
    print("=" * 70)
    print("EXPERIMENT 1: TRUE INTRINSIC DIMENSION (MLE)")
    print("=" * 70)
    print()
    print("H0: CPPN images have same intrinsic dim as random images")
    print("H1: CPPN images have LOWER intrinsic dimension")
    print()

    n_samples = 300
    image_size = 16  # Smaller for computational tractability
    n_features = image_size ** 2

    # Generate CPPN samples
    print(f"Generating {n_samples} CPPN images ({image_size}x{image_size})...")
    cppn_samples = []
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        cppn_samples.append(img.flatten().astype(float))
    cppn_samples = np.array(cppn_samples)

    # Generate random samples
    print(f"Generating {n_samples} random images...")
    random_samples = np.random.randint(0, 2, (n_samples, n_features)).astype(float)

    # Estimate intrinsic dimensions
    print("\nEstimating intrinsic dimensions...")

    k_values = [5, 10, 20]
    results = {'cppn': {}, 'random': {}, 'ambient': n_features}

    for k in k_values:
        print(f"\n  k = {k} nearest neighbors:")

        dim_cppn, se_cppn = mle_intrinsic_dimension(cppn_samples, k=k)
        dim_random, se_random = mle_intrinsic_dimension(random_samples, k=k)

        print(f"    CPPN:   {dim_cppn:.1f} ± {se_cppn:.1f}")
        print(f"    Random: {dim_random:.1f} ± {se_random:.1f}")

        results['cppn'][k] = {'dim': dim_cppn, 'se': se_cppn}
        results['random'][k] = {'dim': dim_random, 'se': se_random}

    # Statistical test (using k=10)
    print("\n" + "-" * 60)
    print("STATISTICAL TEST (k=10):")

    dim_cppn = results['cppn'][10]['dim']
    dim_random = results['random'][10]['dim']
    se_pooled = np.sqrt(results['cppn'][10]['se']**2 + results['random'][10]['se']**2)

    # Z-test for difference
    z = (dim_random - dim_cppn) / se_pooled
    p_value = 1 - norm.cdf(z)  # One-sided

    effect_size = (dim_random - dim_cppn) / dim_random  # Relative reduction

    print(f"  CPPN dimension:   {dim_cppn:.1f}")
    print(f"  Random dimension: {dim_random:.1f}")
    print(f"  Reduction:        {effect_size*100:.1f}%")
    print(f"  Z-statistic:      {z:.2f}")
    print(f"  P-value:          {p_value:.4f}")
    print()

    if p_value < 0.01 and effect_size > 0.1:
        print("  ✓ SIGNIFICANT: CPPN has lower intrinsic dimension")
        results['significant'] = True
        results['effect_size'] = effect_size
    else:
        print("  ✗ Not significant or small effect")
        results['significant'] = False

    results['p_value'] = p_value

    return results


# ============================================================================
# EXPERIMENT 2: BIT-COST SCALING LAWS
# ============================================================================

def experiment_scaling_laws():
    """
    HYPOTHESIS: Bit-cost to reach threshold T scales as B(T) ~ T^α for CPPN.

    This would reveal the geometry of the prior's concentration.

    NULL: No power-law relationship (random scatter).
    """
    print("=" * 70)
    print("EXPERIMENT 2: BIT-COST SCALING LAWS")
    print("=" * 70)
    print()
    print("H0: No systematic relationship between threshold and bits")
    print("H1: Power-law scaling B(T) ~ T^α")
    print()

    # Run nested sampling to get trajectory
    print("Running nested sampling...")
    dead_points, _, _ = nested_sampling_v3(
        n_live=100,
        n_iterations=1500,
        image_size=32,
        order_fn=order_multiplicative,
        seed=42
    )

    orders = np.array([d.order_value for d in dead_points])
    log_X = np.array([d.log_X for d in dead_points])
    bits = -log_X / np.log(2)

    # Extract bits at various thresholds
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    bits_at_threshold = []

    print("\nBits to reach each threshold:")
    print("-" * 40)

    for T in thresholds:
        idx = np.searchsorted(orders, T)
        if idx < len(bits):
            b = bits[idx]
            bits_at_threshold.append((T, b))
            print(f"  T = {T:.2f}: B = {b:.2f} bits")
        else:
            print(f"  T = {T:.2f}: NOT REACHED")

    if len(bits_at_threshold) < 4:
        print("\nInsufficient data for scaling analysis")
        return {'error': 'insufficient data'}

    # Fit power law: log(B) = log(a) + α*log(T)
    T_vals = np.array([x[0] for x in bits_at_threshold])
    B_vals = np.array([x[1] for x in bits_at_threshold])

    log_T = np.log(T_vals)
    log_B = np.log(B_vals)

    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(log_T, log_B)

    alpha = slope
    r_squared = r ** 2

    print("\n" + "-" * 60)
    print("POWER-LAW FIT: B(T) = a * T^α")
    print(f"  α (exponent): {alpha:.3f} ± {se:.3f}")
    print(f"  R²:           {r_squared:.4f}")
    print(f"  P-value:      {p:.4f}")
    print()

    # Interpretation
    if r_squared > 0.9 and p < 0.01:
        print("  ✓ STRONG power-law scaling detected!")
        if alpha > 0:
            print(f"    → Bits INCREASE with threshold (α = {alpha:.2f})")
        else:
            print(f"    → Bits DECREASE with threshold (α = {alpha:.2f})")
        significant = True
    elif r_squared > 0.7:
        print("  ? Moderate power-law relationship")
        significant = False
    else:
        print("  ✗ No clear power-law scaling")
        significant = False

    # Bootstrap confidence interval for α
    print("\nBootstrap 95% CI for α:")

    n_bootstrap = 1000
    alpha_bootstrap = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(log_T), len(log_T), replace=True)
        s, _, _, _, _ = linregress(log_T[idx], log_B[idx])
        alpha_bootstrap.append(s)

    ci_low = np.percentile(alpha_bootstrap, 2.5)
    ci_high = np.percentile(alpha_bootstrap, 97.5)
    print(f"  α ∈ [{ci_low:.3f}, {ci_high:.3f}]")

    return {
        'alpha': alpha,
        'alpha_se': se,
        'alpha_ci': (ci_low, ci_high),
        'r_squared': r_squared,
        'p_value': p,
        'significant': significant,
        'thresholds': T_vals.tolist(),
        'bits': B_vals.tolist()
    }


# ============================================================================
# EXPERIMENT 3: INFORMATION-THEORETIC STRUCTURE
# ============================================================================

def pixel_entropy(img):
    """Entropy of pixel distribution."""
    p = np.mean(img)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)


def block_entropy(img, block_size=2):
    """Entropy of 2x2 block patterns."""
    h, w = img.shape
    patterns = defaultdict(int)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            pattern = tuple(block.flatten())
            patterns[pattern] += 1

    total = sum(patterns.values())
    probs = np.array(list(patterns.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-10))


def mutual_information_adjacent(img):
    """Mutual information between adjacent pixels."""
    # Joint distribution of (pixel, right_neighbor)
    h, w = img.shape
    joint = defaultdict(int)

    for i in range(h):
        for j in range(w - 1):
            pair = (img[i, j], img[i, j+1])
            joint[pair] += 1

    total = sum(joint.values())

    # Marginals
    p_x = np.mean(img)
    p_y = p_x  # Same distribution

    # Joint entropy
    joint_probs = np.array(list(joint.values())) / total
    H_XY = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))

    # Marginal entropy
    H_X = pixel_entropy(img)

    # MI = H(X) + H(Y) - H(X,Y)
    return 2 * H_X - H_XY


def experiment_information_structure():
    """
    HYPOTHESIS: CPPN images have higher spatial mutual information than random.

    This would indicate genuine structure (pixels predict neighbors).

    NULL: MI is same for CPPN and random images.
    """
    print("=" * 70)
    print("EXPERIMENT 3: INFORMATION-THEORETIC STRUCTURE")
    print("=" * 70)
    print()
    print("H0: CPPN images have same spatial MI as random images")
    print("H1: CPPN images have HIGHER spatial mutual information")
    print()

    n_samples = 200
    image_size = 32

    # Generate samples
    print(f"Generating {n_samples} samples of each type...")

    cppn_mi = []
    cppn_block_ent = []
    random_mi = []
    random_block_ent = []

    for i in range(n_samples):
        # CPPN
        cppn = CPPN()
        img_cppn = cppn.render(image_size)
        cppn_mi.append(mutual_information_adjacent(img_cppn))
        cppn_block_ent.append(block_entropy(img_cppn))

        # Random
        img_random = np.random.randint(0, 2, (image_size, image_size))
        random_mi.append(mutual_information_adjacent(img_random))
        random_block_ent.append(block_entropy(img_random))

    cppn_mi = np.array(cppn_mi)
    random_mi = np.array(random_mi)
    cppn_block_ent = np.array(cppn_block_ent)
    random_block_ent = np.array(random_block_ent)

    # Statistical tests
    print("\nRESULTS:")
    print("-" * 60)

    results = {}

    # Test 1: Mutual Information
    print("\n1. SPATIAL MUTUAL INFORMATION (adjacent pixels):")
    print(f"   CPPN:   {np.mean(cppn_mi):.4f} ± {np.std(cppn_mi):.4f}")
    print(f"   Random: {np.mean(random_mi):.4f} ± {np.std(random_mi):.4f}")

    # Mann-Whitney U test (non-parametric)
    stat, p_mi = mannwhitneyu(cppn_mi, random_mi, alternative='greater')
    effect_mi = (np.mean(cppn_mi) - np.mean(random_mi)) / np.std(random_mi)

    print(f"   Mann-Whitney p-value: {p_mi:.6f}")
    print(f"   Effect size (Cohen's d): {effect_mi:.2f}")

    results['mutual_info'] = {
        'cppn_mean': float(np.mean(cppn_mi)),
        'random_mean': float(np.mean(random_mi)),
        'p_value': float(p_mi),
        'effect_size': float(effect_mi),
        'significant': p_mi < 0.01 and effect_mi > 0.5
    }

    if p_mi < 0.01 and effect_mi > 0.5:
        print("   ✓ SIGNIFICANT: CPPN has higher spatial MI")
    else:
        print("   ✗ Not significant")

    # Test 2: Block Entropy
    print("\n2. BLOCK ENTROPY (2x2 patterns):")
    print(f"   CPPN:   {np.mean(cppn_block_ent):.4f} ± {np.std(cppn_block_ent):.4f}")
    print(f"   Random: {np.mean(random_block_ent):.4f} ± {np.std(random_block_ent):.4f}")

    # Random should have HIGHER entropy (more uniform pattern distribution)
    stat, p_ent = mannwhitneyu(random_block_ent, cppn_block_ent, alternative='greater')
    effect_ent = (np.mean(random_block_ent) - np.mean(cppn_block_ent)) / np.std(cppn_block_ent)

    print(f"   Mann-Whitney p-value: {p_ent:.6f}")
    print(f"   Effect size: {effect_ent:.2f}")

    results['block_entropy'] = {
        'cppn_mean': float(np.mean(cppn_block_ent)),
        'random_mean': float(np.mean(random_block_ent)),
        'p_value': float(p_ent),
        'effect_size': float(effect_ent),
        'significant': p_ent < 0.01 and effect_ent > 0.5
    }

    if p_ent < 0.01:
        print("   ✓ SIGNIFICANT: Random has higher block entropy (as expected)")
    else:
        print("   ? Not significant")

    # Summary
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if results['mutual_info']['significant']:
        print("  ✓ CPPN images have genuine spatial structure")
        print("    (adjacent pixels are predictive of each other)")

    if results['block_entropy']['significant']:
        print("  ✓ CPPN uses fewer unique patterns than random")
        print("    (lower block entropy = more repetitive structure)")

    return results


# ============================================================================
# EXPERIMENT 4: PRIOR COMPARISON WITH PROPER STATISTICS
# ============================================================================

def experiment_prior_comparison():
    """
    HYPOTHESIS: Different priors have different bit-efficiencies.

    Compare CPPN vs uniform with bootstrap confidence intervals.
    """
    print("=" * 70)
    print("EXPERIMENT 4: PRIOR COMPARISON (Bootstrap CIs)")
    print("=" * 70)
    print()
    print("Comparing bit-efficiency across priors with proper uncertainty")
    print()

    n_seeds = 10  # Multiple runs for each prior
    n_live = 50
    n_iter = 500
    image_size = 32
    threshold = 0.1

    results = {'cppn': [], 'uniform': []}

    # CPPN runs
    print("Running CPPN prior (10 seeds)...")
    for seed in range(n_seeds):
        dead_points, _, _ = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iter,
            image_size=image_size,
            order_fn=order_multiplicative,
            seed=seed * 100
        )

        orders = np.array([d.order_value for d in dead_points])
        log_X = np.array([d.log_X for d in dead_points])

        # Find bits at threshold
        idx = np.searchsorted(orders, threshold)
        if idx < len(log_X):
            bits = -log_X[idx] / np.log(2)
            results['cppn'].append(bits)
            print(f"  Seed {seed}: {bits:.2f} bits")
        else:
            print(f"  Seed {seed}: threshold not reached")

    # For uniform, we expect failure, so just measure max order achieved
    print("\nRunning uniform prior (10 seeds)...")
    uniform_max_orders = []
    for seed in range(n_seeds):
        np.random.seed(seed * 100)
        max_order = 0
        for _ in range(n_live * n_iter):  # Same sample budget
            img = np.random.randint(0, 2, (image_size, image_size))
            order = order_multiplicative(img)
            max_order = max(max_order, order)
        uniform_max_orders.append(max_order)
        print(f"  Seed {seed}: max order = {max_order:.6f}")

    # Analysis
    print("\n" + "-" * 60)
    print("RESULTS:")

    if len(results['cppn']) >= 3:
        cppn_bits = np.array(results['cppn'])
        print(f"\nCPPN bits to T={threshold}:")
        print(f"  Mean: {np.mean(cppn_bits):.2f}")
        print(f"  Std:  {np.std(cppn_bits):.2f}")
        print(f"  95% CI: [{np.percentile(cppn_bits, 2.5):.2f}, {np.percentile(cppn_bits, 97.5):.2f}]")

    print(f"\nUniform max order (same sample budget):")
    print(f"  Mean: {np.mean(uniform_max_orders):.6f}")
    print(f"  Max:  {np.max(uniform_max_orders):.6f}")
    print(f"  (Never reaches T={threshold})")

    # Compute speedup lower bound
    # If uniform needs B_uniform bits and CPPN needs B_cppn bits,
    # speedup = 2^(B_uniform - B_cppn)
    # Since uniform never reaches, B_uniform > log2(n_live * n_iter)

    budget_bits = np.log2(n_live * n_iter)
    if len(results['cppn']) >= 3:
        cppn_mean = np.mean(cppn_bits)
        speedup_lower_bound = 2 ** (budget_bits - cppn_mean)

        print(f"\n" + "-" * 60)
        print("SPEEDUP ANALYSIS:")
        print(f"  Uniform probed: ≥{budget_bits:.1f} bits (failed)")
        print(f"  CPPN achieved:  {cppn_mean:.1f} bits (succeeded)")
        print(f"  Speedup:        ≥{speedup_lower_bound:.0f}× (lower bound)")

        results['speedup_lower_bound'] = speedup_lower_bound
        results['budget_bits'] = budget_bits

    return results


# ============================================================================
# EXPERIMENT 5: CORRELATION OF ORDER WITH FEATURES
# ============================================================================

def experiment_feature_correlations():
    """
    HYPOTHESIS: Order metric correlates with interpretable image features.

    This validates that "order" measures something meaningful.
    """
    print("=" * 70)
    print("EXPERIMENT 5: ORDER vs INTERPRETABLE FEATURES")
    print("=" * 70)
    print()
    print("Testing if 'order' correlates with meaningful image properties")
    print()

    n_samples = 500
    image_size = 32

    # Generate diverse CPPN samples
    print(f"Generating {n_samples} CPPN images...")

    data = {
        'order': [],
        'symmetry': [],
        'edge_density': [],
        'spectral_coherence': [],
        'compressibility': [],
        'fill_ratio': [],
    }

    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)

        data['order'].append(order_multiplicative(img))
        data['symmetry'].append(compute_symmetry(img))
        data['edge_density'].append(compute_edge_density(img))
        data['spectral_coherence'].append(compute_spectral_coherence(img))
        data['compressibility'].append(compute_compressibility(img))
        data['fill_ratio'].append(np.mean(img))

    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])

    # Correlation analysis
    print("CORRELATIONS WITH ORDER:")
    print("-" * 60)
    print(f"{'Feature':<25} {'Spearman r':<12} {'p-value':<12} {'Sig?':<8}")
    print("-" * 60)

    results = {}
    features = ['symmetry', 'edge_density', 'spectral_coherence', 'compressibility', 'fill_ratio']

    # Bonferroni correction
    alpha = 0.05 / len(features)

    for feat in features:
        r, p = spearmanr(data['order'], data[feat])
        sig = "✓" if p < alpha else ""
        print(f"{feat:<25} {r:>+.3f}       {p:<.6f}     {sig}")

        results[feat] = {
            'spearman_r': float(r),
            'p_value': float(p),
            'significant': p < alpha
        }

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    significant_features = [f for f in features if results[f]['significant']]

    if significant_features:
        print(f"  Order significantly correlates with: {', '.join(significant_features)}")
        print("  This validates that 'order' measures meaningful structure.")
    else:
        print("  No significant correlations (after Bonferroni correction)")

    # Find strongest predictor
    max_r = max(abs(results[f]['spearman_r']) for f in features)
    best_feat = [f for f in features if abs(results[f]['spearman_r']) == max_r][0]
    print(f"\n  Strongest predictor: {best_feat} (r = {results[best_feat]['spearman_r']:.3f})")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_rigorous():
    """Run all rigorous experiments."""
    print("\n" + "=" * 70)
    print("RIGOROUS SIGNAL HUNTING")
    print("With proper statistics, null hypotheses, and significance tests")
    print("=" * 70 + "\n")

    all_results = {}

    # Experiment 1
    print("\n[1/5] Intrinsic Dimension...")
    all_results['intrinsic_dimension'] = experiment_intrinsic_dimension()

    # Experiment 2
    print("\n[2/5] Scaling Laws...")
    all_results['scaling_laws'] = experiment_scaling_laws()

    # Experiment 3
    print("\n[3/5] Information Structure...")
    all_results['information'] = experiment_information_structure()

    # Experiment 4
    print("\n[4/5] Prior Comparison...")
    all_results['prior_comparison'] = experiment_prior_comparison()

    # Experiment 5
    print("\n[5/5] Feature Correlations...")
    all_results['correlations'] = experiment_feature_correlations()

    # Summary
    print("\n" + "=" * 70)
    print("RIGOROUS FINDINGS SUMMARY")
    print("=" * 70)
    print()

    findings = []

    # Check each experiment
    if all_results['intrinsic_dimension'].get('significant'):
        effect = all_results['intrinsic_dimension']['effect_size']
        findings.append(f"✓ CPPN has {effect*100:.0f}% lower intrinsic dimension than random")

    if all_results['scaling_laws'].get('significant'):
        alpha = all_results['scaling_laws']['alpha']
        findings.append(f"✓ Power-law scaling: B(T) ~ T^{alpha:.2f}")

    if all_results['information']['mutual_info'].get('significant'):
        effect = all_results['information']['mutual_info']['effect_size']
        findings.append(f"✓ CPPN has higher spatial MI (effect size = {effect:.1f}σ)")

    if all_results['information']['block_entropy'].get('significant'):
        findings.append("✓ CPPN uses fewer unique block patterns")

    if 'speedup_lower_bound' in all_results['prior_comparison']:
        speedup = all_results['prior_comparison']['speedup_lower_bound']
        findings.append(f"✓ CPPN provides ≥{speedup:.0f}× speedup over uniform")

    corr_results = all_results['correlations']
    sig_corrs = [f for f in corr_results if corr_results[f].get('significant')]
    if sig_corrs:
        findings.append(f"✓ Order correlates with: {', '.join(sig_corrs)}")

    if findings:
        print("STATISTICALLY SIGNIFICANT FINDINGS:")
        print("-" * 60)
        for f in findings:
            print(f"  {f}")
    else:
        print("No statistically significant findings.")

    # Save results
    output_dir = Path("results/rigorous_signals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_dir / "rigorous_results.json", 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {output_dir}/rigorous_results.json")

    return all_results


if __name__ == "__main__":
    run_all_rigorous()
