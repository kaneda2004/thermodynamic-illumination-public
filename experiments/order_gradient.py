"""
Order Gradient Landscape Experiment

Hypothesis: CPPN generators exhibit lower order-gradient magnitudes (||dO/dw||)
than linear generators in high-order regions (order > 0.2), indicating flatter
landscape peaks that facilitate nested sampling convergence.

Generated via 9-turn deliberation (Theorist/Empiricist/Skeptic).
Research ID: RES-009
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    PRIOR_SIGMA,
)


@dataclass
class LinearGenerator:
    """
    Linear generator with same parameter count as minimal CPPN.

    CPPN has: 4 input->output weights + 1 output bias = 5 params
    Linear: 4 input weights + 1 bias = 5 params

    Output: w * [x, y, r, bias]^T + b -> sigmoid -> threshold
    """
    weights: np.ndarray = None
    bias: float = 0.0

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.random.randn(4) * PRIOR_SIGMA
            self.bias = np.random.randn() * PRIOR_SIGMA

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        bias_field = np.ones_like(x)

        # Stack inputs: [x, y, r, bias]
        inputs = np.stack([x, y, r, bias_field], axis=-1)

        # Linear combination
        linear_out = np.sum(inputs * self.weights, axis=-1) + self.bias

        # Sigmoid and threshold
        sigmoid_out = 1 / (1 + np.exp(-np.clip(linear_out, -10, 10)))
        return (sigmoid_out > 0.5).astype(np.uint8)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.weights, [self.bias]])

    def set_weights(self, w: np.ndarray):
        self.weights = w[:4]
        self.bias = w[4]

    def copy(self) -> 'LinearGenerator':
        return LinearGenerator(
            weights=self.weights.copy(),
            bias=self.bias
        )


def compute_gradient_magnitude(generator, order_fn, epsilon: float = 0.01, n_samples: int = 20) -> float:
    """
    Estimate ||dO/dw|| via finite differences.

    Sample random perturbation directions, compute (O(w+eps*d) - O(w-eps*d)) / (2*eps)
    Average over n_samples directions for robust estimate.
    """
    base_weights = generator.get_weights()
    n_params = len(base_weights)

    base_img = generator.render(32)
    base_order = order_fn(base_img)

    gradient_samples = []

    for _ in range(n_samples):
        # Random direction (unit vector)
        direction = np.random.randn(n_params)
        direction = direction / np.linalg.norm(direction)

        # Forward perturbation
        gen_plus = generator.copy()
        gen_plus.set_weights(base_weights + epsilon * direction)
        order_plus = order_fn(gen_plus.render(32))

        # Backward perturbation
        gen_minus = generator.copy()
        gen_minus.set_weights(base_weights - epsilon * direction)
        order_minus = order_fn(gen_minus.render(32))

        # Directional derivative estimate
        deriv = (order_plus - order_minus) / (2 * epsilon)
        gradient_samples.append(abs(deriv))

    # Average gradient magnitude
    return np.mean(gradient_samples)


def run_experiment(
    n_samples: int = 300,
    image_size: int = 32,
    epsilons: list = None,
    order_bins: list = None,
    min_per_bin: int = 30,
    n_gradient_samples: int = 20,
    seed: int = 42
) -> dict:
    """
    Run the order gradient landscape experiment.

    Tests whether CPPN generators have lower order-gradient magnitudes
    than linear generators in high-order regions.
    """
    if epsilons is None:
        epsilons = [0.01, 0.001]
    if order_bins is None:
        order_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)]

    np.random.seed(seed)

    print("=" * 60)
    print("ORDER GRADIENT LANDSCAPE EXPERIMENT")
    print("=" * 60)
    print("\nHypothesis: CPPN has lower ||dO/dw|| than Linear in high-order regions")
    print("Null: Gradient magnitudes are equal between generators\n")

    primary_eps = epsilons[0]

    # Step 1: Generate CPPN samples
    print(f"1. Generating {n_samples} CPPN samples...")
    cppn_data = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        grad_mag = compute_gradient_magnitude(cppn, order_multiplicative, primary_eps, n_gradient_samples)
        cppn_data.append({
            'order': order,
            'gradient_magnitude': grad_mag,
            'n_params': len(cppn.get_weights())
        })
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{n_samples}")

    print(f"   CPPN: {len(cppn.get_weights())} parameters")

    # Step 2: Generate Linear samples
    print(f"\n2. Generating {n_samples} Linear generator samples...")
    linear_data = []
    for i in range(n_samples):
        linear = LinearGenerator()
        img = linear.render(image_size)
        order = order_multiplicative(img)
        grad_mag = compute_gradient_magnitude(linear, order_multiplicative, primary_eps, n_gradient_samples)
        linear_data.append({
            'order': order,
            'gradient_magnitude': grad_mag,
            'n_params': len(linear.get_weights())
        })
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{n_samples}")

    print(f"   Linear: {len(linear.get_weights())} parameters")

    # Step 3: Bin by order
    print("\n3. Binning samples by order level...")

    results_by_bin = {}

    for bin_low, bin_high in order_bins:
        bin_label = f"[{bin_low:.1f}, {bin_high:.1f})"

        cppn_in_bin = [d for d in cppn_data if bin_low <= d['order'] < bin_high]
        linear_in_bin = [d for d in linear_data if bin_low <= d['order'] < bin_high]

        n_cppn = len(cppn_in_bin)
        n_linear = len(linear_in_bin)

        print(f"\n   Bin {bin_label}: CPPN={n_cppn}, Linear={n_linear}")

        if n_cppn < min_per_bin or n_linear < min_per_bin:
            print(f"   SKIPPED (need >= {min_per_bin} per group)")
            results_by_bin[bin_label] = {
                'status': 'skipped',
                'n_cppn': n_cppn,
                'n_linear': n_linear,
                'reason': f'insufficient samples (need >= {min_per_bin})'
            }
            continue

        cppn_grads = np.array([d['gradient_magnitude'] for d in cppn_in_bin])
        linear_grads = np.array([d['gradient_magnitude'] for d in linear_in_bin])

        # Mann-Whitney U test (CPPN < Linear)
        stat, p_value = stats.mannwhitneyu(cppn_grads, linear_grads, alternative='less')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(cppn_grads) + np.var(linear_grads)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(linear_grads) - np.mean(cppn_grads)) / pooled_std
        else:
            cohens_d = 0.0

        print(f"   CPPN gradient: {np.mean(cppn_grads):.4f} +/- {np.std(cppn_grads):.4f}")
        print(f"   Linear gradient: {np.mean(linear_grads):.4f} +/- {np.std(linear_grads):.4f}")
        print(f"   Mann-Whitney p: {p_value:.6f}")
        print(f"   Cohen's d: {cohens_d:.3f}")

        results_by_bin[bin_label] = {
            'status': 'tested',
            'n_cppn': n_cppn,
            'n_linear': n_linear,
            'cppn_mean': float(np.mean(cppn_grads)),
            'cppn_std': float(np.std(cppn_grads)),
            'linear_mean': float(np.mean(linear_grads)),
            'linear_std': float(np.std(linear_grads)),
            'mann_whitney_stat': float(stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }

    # Step 4: Epsilon stability check
    print(f"\n4. Epsilon stability check (eps={epsilons[1]})...")

    # Sample a subset for stability check
    stability_samples = 50
    np.random.seed(seed + 100)

    stability_cppn_eps1 = []
    stability_cppn_eps2 = []

    for _ in range(stability_samples):
        cppn = CPPN()
        grad1 = compute_gradient_magnitude(cppn, order_multiplicative, epsilons[0], n_gradient_samples)
        grad2 = compute_gradient_magnitude(cppn, order_multiplicative, epsilons[1], n_gradient_samples)
        stability_cppn_eps1.append(grad1)
        stability_cppn_eps2.append(grad2)

    stability_corr, stability_p = stats.pearsonr(stability_cppn_eps1, stability_cppn_eps2)
    print(f"   Correlation between eps={epsilons[0]} and eps={epsilons[1]}: r={stability_corr:.3f}")

    # Step 5: Determine outcome
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Bonferroni threshold
    n_tests = sum(1 for b in results_by_bin.values() if b['status'] == 'tested')
    bonferroni_alpha = 0.05 / max(1, n_tests)

    print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")

    significant_bins = []
    effect_bins = []

    for bin_label, res in results_by_bin.items():
        if res['status'] != 'tested':
            continue

        sig = res['p_value'] < bonferroni_alpha
        eff = res['cohens_d'] > 0.5

        print(f"\n{bin_label}:")
        print(f"  p < {bonferroni_alpha:.4f}: {sig} (p={res['p_value']:.6f})")
        print(f"  Cohen's d > 0.5: {eff} (d={res['cohens_d']:.3f})")

        if sig:
            significant_bins.append(bin_label)
        if eff:
            effect_bins.append(bin_label)

    # Focus on high-order bins
    high_order_bins = ["[0.2, 0.3)", "[0.3, 1.0)"]
    high_order_sig = any(b in significant_bins for b in high_order_bins)
    high_order_eff = any(b in effect_bins for b in high_order_bins)

    stable = stability_corr > 0.7

    print(f"\nStability (r > 0.7): {stable} (r={stability_corr:.3f})")
    print(f"Significant in high-order region: {high_order_sig}")
    print(f"Large effect in high-order region: {high_order_eff}")

    # Determine status
    if high_order_sig and high_order_eff and stable:
        status = 'validated'
        summary = "CPPN shows lower gradient magnitude in high-order regions (flatter landscape)"
    elif not high_order_sig and not high_order_eff:
        status = 'refuted'
        summary = "No evidence of gradient magnitude difference between generators"
    else:
        status = 'inconclusive'
        summary = "Mixed evidence - some criteria met but not all"

    print(f"\nSTATUS: {status.upper()}")
    print(f"Summary: {summary}")

    # Compile results
    results = {
        'experiment': 'order_gradient_landscape',
        'hypothesis': 'CPPN has lower ||dO/dw|| than Linear in high-order regions',
        'null_hypothesis': 'Gradient magnitudes equal between generators at each order level',
        'status': status,
        'summary': summary,
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'primary_epsilon': primary_eps,
            'n_gradient_samples': n_gradient_samples,
            'order_bins': [[b[0], b[1]] for b in order_bins],
            'bonferroni_alpha': float(bonferroni_alpha)
        },
        'results_by_bin': results_by_bin,
        'stability_check': {
            'epsilon_values': epsilons,
            'correlation': float(stability_corr),
            'p_value': float(stability_p),
            'stable': bool(stable)
        },
        'validation_checks': {
            'high_order_significant': bool(high_order_sig),
            'high_order_effect': bool(high_order_eff),
            'stable_gradient_estimate': bool(stable)
        },
        'order_distributions': {
            'cppn': {
                'mean': float(np.mean([d['order'] for d in cppn_data])),
                'std': float(np.std([d['order'] for d in cppn_data])),
                'min': float(min(d['order'] for d in cppn_data)),
                'max': float(max(d['order'] for d in cppn_data))
            },
            'linear': {
                'mean': float(np.mean([d['order'] for d in linear_data])),
                'std': float(np.std([d['order'] for d in linear_data])),
                'min': float(min(d['order'] for d in linear_data)),
                'max': float(max(d['order'] for d in linear_data))
            }
        }
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'order_gradient'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'gradient_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir / 'gradient_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
