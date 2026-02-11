#!/usr/bin/env python3
"""
RES-126: Coordinate Input Ablation Study

Hypothesis: Removing individual CPPN coordinate inputs (x, y, r, bias)
reveals differential contributions to image order, with radial input (r)
being most important due to its role in creating global structure.

Method:
1. Generate CPPNs with all inputs enabled (baseline)
2. Create ablated versions by zeroing each input channel
3. Measure order degradation for each ablation
4. Statistical comparison across many samples
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


@dataclass
class AblatedCPPN:
    """CPPN wrapper that zeros specific input channels during activation."""
    cppn: CPPN
    ablate_x: bool = False
    ablate_y: bool = False
    ablate_r: bool = False
    ablate_bias: bool = False

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Activate with ablated inputs."""
        # Compute standard inputs
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)

        # Apply ablations (zero out specific inputs)
        if self.ablate_x:
            x = np.zeros_like(x)
        if self.ablate_y:
            y = np.zeros_like(y)
        if self.ablate_r:
            r = np.zeros_like(r)
        if self.ablate_bias:
            bias = np.zeros_like(bias)

        # Run CPPN with modified inputs
        values = {0: x, 1: y, 2: r, 3: bias}
        for nid in self.cppn._get_eval_order():
            node = next(n for n in self.cppn.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.cppn.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            from core.thermo_sampler_v3 import ACTIVATIONS
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.cppn.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)


def run_ablation_experiment(n_samples: int = 500, size: int = 32, seed: int = 42):
    """
    Run coordinate input ablation study.

    For each CPPN:
    1. Measure baseline order
    2. Measure order with each input ablated
    3. Compute degradation = baseline - ablated
    """
    set_global_seed(seed)

    input_names = ['x', 'y', 'r', 'bias']

    # Storage for results
    baseline_orders = []
    ablated_orders = {name: [] for name in input_names}
    degradations = {name: [] for name in input_names}

    for i in range(n_samples):
        # Generate a CPPN
        cppn = CPPN()

        # Baseline order
        img_baseline = cppn.render(size)
        order_baseline = order_multiplicative(img_baseline)
        baseline_orders.append(order_baseline)

        # Ablate each input and measure order
        for name in input_names:
            ablated = AblatedCPPN(
                cppn=cppn,
                ablate_x=(name == 'x'),
                ablate_y=(name == 'y'),
                ablate_r=(name == 'r'),
                ablate_bias=(name == 'bias')
            )
            img_ablated = ablated.render(size)
            order_ablated = order_multiplicative(img_ablated)

            ablated_orders[name].append(order_ablated)
            degradations[name].append(order_baseline - order_ablated)

    # Convert to arrays
    baseline_orders = np.array(baseline_orders)
    for name in input_names:
        ablated_orders[name] = np.array(ablated_orders[name])
        degradations[name] = np.array(degradations[name])

    return baseline_orders, ablated_orders, degradations, input_names


def analyze_results(baseline_orders, ablated_orders, degradations, input_names):
    """Statistical analysis of ablation effects."""

    print("=" * 70)
    print("RES-126: Coordinate Input Ablation Study")
    print("=" * 70)

    print(f"\nBaseline order: mean={baseline_orders.mean():.4f}, std={baseline_orders.std():.4f}")
    print("\n" + "-" * 70)
    print("Ablation Effects (sorted by degradation)")
    print("-" * 70)

    # Compute statistics for each ablation
    results = []
    for name in input_names:
        deg = degradations[name]
        abl = ablated_orders[name]

        # Paired t-test: baseline vs ablated
        t_stat, p_val = stats.ttest_rel(baseline_orders, abl)

        # Effect size (Cohen's d for paired samples)
        d = deg.mean() / deg.std() if deg.std() > 0 else 0

        results.append({
            'name': name,
            'mean_degradation': deg.mean(),
            'std_degradation': deg.std(),
            'ablated_mean': abl.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'cohens_d': d,
            'pct_with_degradation': (deg > 0).mean() * 100
        })

    # Sort by mean degradation (most impactful first)
    results.sort(key=lambda x: x['mean_degradation'], reverse=True)

    for r in results:
        print(f"\nAblated input: {r['name'].upper()}")
        print(f"  Mean degradation: {r['mean_degradation']:.4f} +/- {r['std_degradation']:.4f}")
        print(f"  Ablated order mean: {r['ablated_mean']:.4f}")
        print(f"  % samples with degradation: {r['pct_with_degradation']:.1f}%")
        print(f"  t-statistic: {r['t_stat']:.2f}")
        print(f"  p-value: {r['p_val']:.2e}")
        print(f"  Cohen's d: {r['cohens_d']:.3f}")

    # Test hypothesis: is r most important?
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Is radial input (r) most important?")
    print("=" * 70)

    # Find r's rank
    r_result = next(r for r in results if r['name'] == 'r')
    r_rank = [r['name'] for r in results].index('r') + 1
    most_important = results[0]['name']

    print(f"\nMost important input: {most_important.upper()} (degradation={results[0]['mean_degradation']:.4f})")
    print(f"R input rank: {r_rank} of 4 (degradation={r_result['mean_degradation']:.4f})")

    # Statistical comparison: r vs most important (if different)
    if most_important != 'r':
        deg_r = degradations['r']
        deg_top = degradations[most_important]
        t_stat, p_val = stats.ttest_rel(deg_top, deg_r)
        print(f"\nComparison {most_important.upper()} vs R:")
        print(f"  t-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_val:.2e}")

    # Check if any single input dominates
    print("\n" + "-" * 70)
    print("Cross-Input Comparison (Bonferroni corrected, alpha=0.01/6=0.00167)")
    print("-" * 70)

    n_comparisons = 6  # 4 choose 2
    alpha_corrected = 0.01 / n_comparisons

    sig_pairs = []
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            t, p = stats.ttest_rel(degradations[r1['name']], degradations[r2['name']])
            sig = "***" if p < alpha_corrected else ""
            sig_pairs.append((r1['name'], r2['name'], abs(t), p, sig))
            if p < alpha_corrected:
                print(f"  {r1['name'].upper()} vs {r2['name'].upper()}: t={abs(t):.2f}, p={p:.2e} {sig}")

    return results, sig_pairs


def main():
    print("Running coordinate input ablation experiment...")
    print("n_samples=500, size=32x32\n")

    baseline_orders, ablated_orders, degradations, input_names = run_ablation_experiment(
        n_samples=500, size=32, seed=42
    )

    results, sig_pairs = analyze_results(baseline_orders, ablated_orders, degradations, input_names)

    # Summary for research log
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 70)

    # Determine outcome
    most_important = results[0]['name']
    effect_size = results[0]['cohens_d']
    p_value = results[0]['p_val']

    hypothesis_confirmed = (most_important == 'r')

    print(f"\nHypothesis: R is most important input")
    print(f"Result: {'VALIDATED' if hypothesis_confirmed else 'REFUTED'}")
    print(f"Most important: {most_important.upper()}")
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    print(f"P-value: {p_value:.2e}")

    # Output ranking
    print("\nInput importance ranking (by order degradation):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name'].upper()}: d={r['cohens_d']:.3f}, deg={r['mean_degradation']:.4f}")

    return results, hypothesis_confirmed


if __name__ == "__main__":
    results, hypothesis_confirmed = main()
