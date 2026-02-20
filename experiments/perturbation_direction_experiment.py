"""
Experiment: Perturbation Direction Anisotropy (RES-065)

Tests whether there are privileged directions in weight space for increasing order.
Compares gradient-aligned perturbations vs random direction perturbations.

Related work:
- RES-015: Lipschitz constant scales with order (dynamic sensitivity)
- RES-049: Fisher information 157x higher at high order (curvature)

This experiment tests DIRECTION, not magnitude - are some directions privileged
for increasing order?

Methodology:
1. Sample N CPPNs from prior
2. For each CPPN, compute local order gradient via finite differences
3. Apply perturbation in gradient direction vs random directions
4. Compare absolute order change magnitude
5. Test: gradient-aligned perturbations should have larger |dO|

Null hypothesis: Sensitivity is isotropic - gradient and random directions
produce equal |dO| on average.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_order_gradient(cppn: CPPN, epsilon: float = 0.01) -> np.ndarray:
    """
    Compute gradient of order w.r.t. weights via finite differences.

    Returns normalized gradient vector (unit length).
    """
    weights = cppn.get_weights()
    n_weights = len(weights)
    gradient = np.zeros(n_weights)

    # Compute current order
    img = cppn.render(size=32)
    order_0 = order_multiplicative(img)

    for i in range(n_weights):
        # Perturb weight i
        cppn_plus = cppn.copy()
        w_plus = weights.copy()
        w_plus[i] += epsilon
        cppn_plus.set_weights(w_plus)
        order_plus = order_multiplicative(cppn_plus.render(size=32))

        cppn_minus = cppn.copy()
        w_minus = weights.copy()
        w_minus[i] -= epsilon
        cppn_minus.set_weights(w_minus)
        order_minus = order_multiplicative(cppn_minus.render(size=32))

        gradient[i] = (order_plus - order_minus) / (2 * epsilon)

    # Normalize
    norm = np.linalg.norm(gradient)
    if norm > 1e-10:
        gradient = gradient / norm
    else:
        # If gradient is zero, use random direction
        gradient = np.random.randn(n_weights)
        gradient = gradient / np.linalg.norm(gradient)

    return gradient


def perturbation_experiment(n_samples: int = 200,
                           perturbation_magnitude: float = 0.1,
                           n_random_dirs: int = 20):
    """
    Compare gradient-aligned vs random direction perturbations.

    For each CPPN:
    1. Compute gradient direction
    2. Perturb in +gradient direction, measure |dO|
    3. Perturb in n_random_dirs random directions, measure |dO|
    4. Compare gradient |dO| vs mean random |dO|

    Returns dict of results.
    """
    np.random.seed(42)

    gradient_changes = []
    random_changes = []
    gradient_vs_random_ratios = []
    gradient_increases_order = []  # Track sign

    for i in range(n_samples):
        if i % 20 == 0:
            print(f"  Sample {i}/{n_samples}...")

        # Sample CPPN from prior
        cppn = CPPN()
        weights = cppn.get_weights()
        n_weights = len(weights)

        # Current order
        img = cppn.render(size=32)
        order_0 = order_multiplicative(img)

        # Compute gradient
        gradient = compute_order_gradient(cppn)

        # Perturb in gradient direction
        cppn_grad = cppn.copy()
        w_grad = weights + perturbation_magnitude * gradient
        cppn_grad.set_weights(w_grad)
        order_grad = order_multiplicative(cppn_grad.render(size=32))
        dO_grad = order_grad - order_0
        gradient_changes.append(abs(dO_grad))
        gradient_increases_order.append(dO_grad > 0)

        # Perturb in random directions
        random_dOs = []
        for _ in range(n_random_dirs):
            rand_dir = np.random.randn(n_weights)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)

            cppn_rand = cppn.copy()
            w_rand = weights + perturbation_magnitude * rand_dir
            cppn_rand.set_weights(w_rand)
            order_rand = order_multiplicative(cppn_rand.render(size=32))
            random_dOs.append(abs(order_rand - order_0))

        mean_random_dO = np.mean(random_dOs)
        random_changes.append(mean_random_dO)

        # Ratio
        if mean_random_dO > 1e-10:
            gradient_vs_random_ratios.append(abs(dO_grad) / mean_random_dO)
        else:
            gradient_vs_random_ratios.append(1.0)

    # Statistical tests
    gradient_changes = np.array(gradient_changes)
    random_changes = np.array(random_changes)
    gradient_vs_random_ratios = np.array(gradient_vs_random_ratios)

    # Paired comparison: gradient vs mean random for each CPPN
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(gradient_changes, random_changes,
                                                alternative='greater')

    # Effect size: Cohen's d
    diff = gradient_changes - random_changes
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    # Mann-Whitney (unpaired)
    mann_whitney_stat, mann_whitney_p = stats.mannwhitneyu(
        gradient_changes, random_changes, alternative='greater'
    )

    # Mean ratio
    mean_ratio = np.mean(gradient_vs_random_ratios)
    median_ratio = np.median(gradient_vs_random_ratios)

    # T-test
    ttest_stat, ttest_p = stats.ttest_rel(gradient_changes, random_changes)
    # One-sided p-value
    ttest_p_onesided = ttest_p / 2 if ttest_stat > 0 else 1 - ttest_p / 2

    # Fraction where gradient > random
    frac_gradient_wins = np.mean(gradient_changes > random_changes)

    # Sign test: does gradient direction increase order?
    frac_increases = np.mean(gradient_increases_order)
    sign_test_result = stats.binomtest(sum(gradient_increases_order), n_samples, 0.5,
                                        alternative='greater')
    sign_test_p = sign_test_result.pvalue

    results = {
        'n_samples': n_samples,
        'perturbation_magnitude': perturbation_magnitude,
        'n_random_dirs': n_random_dirs,

        'gradient_dO_mean': float(np.mean(gradient_changes)),
        'gradient_dO_std': float(np.std(gradient_changes)),
        'random_dO_mean': float(np.mean(random_changes)),
        'random_dO_std': float(np.std(random_changes)),

        'mean_ratio': float(mean_ratio),
        'median_ratio': float(median_ratio),
        'ratio_std': float(np.std(gradient_vs_random_ratios)),

        'frac_gradient_wins': float(frac_gradient_wins),
        'frac_increases_order': float(frac_increases),

        'cohens_d': float(cohens_d),
        'wilcoxon_p': float(wilcoxon_p),
        'mann_whitney_p': float(mann_whitney_p),
        'ttest_p_onesided': float(ttest_p_onesided),
        'sign_test_p': float(sign_test_p),
    }

    return results


def main():
    """Run perturbation direction experiment."""
    print("=" * 60)
    print("PERTURBATION DIRECTION ANISOTROPY EXPERIMENT (RES-065)")
    print("=" * 60)
    print()
    print("Question: Are there privileged directions in weight space?")
    print("Test: gradient-aligned perturbations vs random perturbations")
    print()

    results = perturbation_experiment(
        n_samples=200,
        perturbation_magnitude=0.1,
        n_random_dirs=20
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Gradient direction |dO|: {results['gradient_dO_mean']:.4f} +/- {results['gradient_dO_std']:.4f}")
    print(f"Random direction |dO|:   {results['random_dO_mean']:.4f} +/- {results['random_dO_std']:.4f}")
    print()
    print(f"Mean ratio (gradient/random): {results['mean_ratio']:.2f}x")
    print(f"Median ratio: {results['median_ratio']:.2f}x")
    print()
    print(f"Fraction gradient > random: {results['frac_gradient_wins']*100:.1f}%")
    print(f"Fraction gradient INCREASES order: {results['frac_increases_order']*100:.1f}%")
    print()
    print("Statistical tests:")
    print(f"  Cohen's d: {results['cohens_d']:.3f}")
    print(f"  Wilcoxon signed-rank (gradient > random): p = {results['wilcoxon_p']:.2e}")
    print(f"  Mann-Whitney U (gradient > random): p = {results['mann_whitney_p']:.2e}")
    print(f"  Paired t-test (one-sided): p = {results['ttest_p_onesided']:.2e}")
    print(f"  Sign test (gradient increases order): p = {results['sign_test_p']:.2e}")
    print()

    # Interpretation
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if results['wilcoxon_p'] < 0.01 and results['cohens_d'] > 0.5:
        print("VALIDATED: Sensitivity is ANISOTROPIC")
        print(f"  Gradient-aligned perturbations produce {results['mean_ratio']:.1f}x larger |dO|")
        print(f"  Effect size d={results['cohens_d']:.2f} (large if >0.8)")
        status = "validated"
    elif results['wilcoxon_p'] < 0.01:
        print("PARTIALLY VALIDATED: Anisotropy exists but effect size is small")
        print(f"  Effect size d={results['cohens_d']:.2f}")
        status = "validated"
    else:
        print("REFUTED: Sensitivity appears ISOTROPIC")
        print(f"  No significant difference between gradient and random directions")
        print(f"  p = {results['wilcoxon_p']:.3f}")
        status = "refuted"

    # Sign interpretation
    if results['sign_test_p'] < 0.01:
        print()
        print(f"BONUS: Gradient direction predicts order INCREASE")
        print(f"  {results['frac_increases_order']*100:.0f}% of gradient perturbations increase order (p={results['sign_test_p']:.2e})")

    print()

    # Summary for log
    summary = (
        f"Gradient-aligned perturbations produce {results['mean_ratio']:.1f}x larger |dO| than random "
        f"(d={results['cohens_d']:.2f}, p={results['wilcoxon_p']:.1e}). "
        f"Gradient direction increases order in {results['frac_increases_order']*100:.0f}% of cases. "
        f"Sensitivity is {'anisotropic with privileged directions' if status == 'validated' else 'isotropic'}."
    )

    print("SUMMARY:", summary)
    print()
    print("STATUS:", status.upper())

    return results, status, summary


if __name__ == "__main__":
    results, status, summary = main()
