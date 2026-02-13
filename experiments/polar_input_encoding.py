#!/usr/bin/env python3
"""
RES-057: Test if polar coordinates (r, theta) as CPPN inputs improve order generation.

Compares three input encodings:
1. Cartesian-only: [x, y, bias]
2. Current default: [x, y, r, bias]
3. Full polar: [x, y, r, theta, bias]

Hypothesis: Adding theta should enable radially-symmetric and rotationally-structured
patterns that improve order generation compared to Cartesian-only inputs.
"""

import numpy as np
from dataclasses import dataclass, field
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    order_multiplicative, ACTIVATIONS, PRIOR_SIGMA,
    Node, Connection
)
from scipy import stats

@dataclass
class FlexCPPN:
    """CPPN with configurable input encoding."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=list)
    output_id: int = -1
    encoding: str = "cartesian"  # "cartesian", "radial", "polar"

    def __post_init__(self):
        if not self.nodes:
            if self.encoding == "cartesian":
                # [x, y, bias]
                self.input_ids = [0, 1, 2]
                self.nodes = [
                    Node(0, 'identity', 0.0),  # x
                    Node(1, 'identity', 0.0),  # y
                    Node(2, 'identity', 0.0),  # bias
                    Node(3, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
                ]
                self.output_id = 3
            elif self.encoding == "radial":
                # [x, y, r, bias] - current default
                self.input_ids = [0, 1, 2, 3]
                self.nodes = [
                    Node(0, 'identity', 0.0),  # x
                    Node(1, 'identity', 0.0),  # y
                    Node(2, 'identity', 0.0),  # r
                    Node(3, 'identity', 0.0),  # bias
                    Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
                ]
                self.output_id = 4
            elif self.encoding == "polar":
                # [x, y, r, theta, bias]
                self.input_ids = [0, 1, 2, 3, 4]
                self.nodes = [
                    Node(0, 'identity', 0.0),  # x
                    Node(1, 'identity', 0.0),  # y
                    Node(2, 'identity', 0.0),  # r
                    Node(3, 'identity', 0.0),  # theta
                    Node(4, 'identity', 0.0),  # bias
                    Node(5, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
                ]
                self.output_id = 5

            for inp in self.input_ids:
                self.connections.append(Connection(inp, self.output_id, np.random.randn() * PRIOR_SIGMA))

    def _get_eval_order(self) -> list:
        hidden_ids = sorted([n.id for n in self.nodes
                            if n.id not in self.input_ids and n.id != self.output_id])
        return hidden_ids + [self.output_id]

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) / np.pi  # Normalize to [-1, 1]
        bias = np.ones_like(x)

        if self.encoding == "cartesian":
            values = {0: x, 1: y, 2: bias}
        elif self.encoding == "radial":
            values = {0: x, 1: y, 2: r, 3: bias}
        elif self.encoding == "polar":
            values = {0: x, 1: y, 2: r, 3: theta, 4: bias}

        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)


def sample_orders(encoding: str, n_samples: int = 500, seed: int = 42) -> np.ndarray:
    """Generate n_samples CPPNs with given encoding and measure order."""
    np.random.seed(seed)
    orders = []
    for _ in range(n_samples):
        cppn = FlexCPPN(encoding=encoding)
        img = cppn.render(32)
        order = order_multiplicative(img)
        orders.append(order)
    return np.array(orders)


def run_experiment():
    """Compare input encodings for order generation."""
    print("=" * 60)
    print("RES-057: Polar Coordinate Input Encoding Experiment")
    print("=" * 60)

    n_samples = 1000
    n_bootstrap = 10000

    encodings = ["cartesian", "radial", "polar"]
    results = {}

    for enc in encodings:
        print(f"\nSampling {n_samples} CPPNs with {enc} encoding...")
        orders = sample_orders(enc, n_samples, seed=42)
        results[enc] = {
            'orders': orders,
            'mean': np.mean(orders),
            'std': np.std(orders),
            'median': np.median(orders),
            'nonzero': np.mean(orders > 0.01),
            'high_order': np.mean(orders > 0.1)
        }
        print(f"  Mean order: {results[enc]['mean']:.4f} +/- {results[enc]['std']:.4f}")
        print(f"  Median: {results[enc]['median']:.4f}")
        print(f"  Non-zero (>0.01): {results[enc]['nonzero']*100:.1f}%")
        print(f"  High order (>0.1): {results[enc]['high_order']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Statistical Comparisons")
    print("=" * 60)

    # Pairwise comparisons using Mann-Whitney U test (non-parametric)
    comparisons = [
        ("cartesian", "radial", "Radial vs Cartesian"),
        ("cartesian", "polar", "Polar vs Cartesian"),
        ("radial", "polar", "Polar vs Radial")
    ]

    all_results = []
    for enc1, enc2, label in comparisons:
        orders1 = results[enc1]['orders']
        orders2 = results[enc2]['orders']

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(orders2, orders1, alternative='greater')

        # Effect size (Cliff's delta for non-parametric)
        n1, n2 = len(orders1), len(orders2)
        # Count how many times enc2 > enc1
        dom = np.sum(orders2[:, None] > orders1[None, :]) - np.sum(orders2[:, None] < orders1[None, :])
        cliffs_delta = dom / (n1 * n2)

        # Bootstrap CI for mean difference
        boot_diffs = []
        for _ in range(n_bootstrap):
            idx1 = np.random.choice(n1, n1, replace=True)
            idx2 = np.random.choice(n2, n2, replace=True)
            boot_diffs.append(np.mean(orders2[idx2]) - np.mean(orders1[idx1]))
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

        mean_diff = results[enc2]['mean'] - results[enc1]['mean']

        all_results.append({
            'comparison': label,
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'cliffs_delta': cliffs_delta,
            'p_value': p_value
        })

        print(f"\n{label}:")
        print(f"  Mean difference: {mean_diff:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
        print(f"  Cliff's delta: {cliffs_delta:.3f}")
        print(f"  Mann-Whitney p-value: {p_value:.2e}")

    # Bonferroni correction
    alpha = 0.01
    bonferroni_alpha = alpha / len(comparisons)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Check polar vs cartesian (main hypothesis)
    polar_vs_cartesian = all_results[1]
    radial_vs_cartesian = all_results[0]
    polar_vs_radial = all_results[2]

    print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")

    # Determine outcome
    if polar_vs_cartesian['p_value'] < bonferroni_alpha and polar_vs_cartesian['cliffs_delta'] > 0.147:
        # Cliff's delta > 0.147 is considered "small" effect
        if polar_vs_radial['p_value'] < bonferroni_alpha and polar_vs_radial['cliffs_delta'] > 0.147:
            status = "VALIDATED"
            summary = f"Polar encoding (x,y,r,theta,bias) produces higher order than both Cartesian-only and radial. Polar vs Cartesian: d={polar_vs_cartesian['cliffs_delta']:.3f}, p={polar_vs_cartesian['p_value']:.2e}"
        else:
            status = "PARTIAL"
            summary = f"Polar better than Cartesian (d={polar_vs_cartesian['cliffs_delta']:.3f}) but not significantly better than radial"
    elif radial_vs_cartesian['p_value'] < bonferroni_alpha and radial_vs_cartesian['cliffs_delta'] > 0.147:
        status = "PARTIAL"
        summary = f"Radial encoding helps (d={radial_vs_cartesian['cliffs_delta']:.3f}) but adding theta doesn't improve further"
    else:
        if polar_vs_cartesian['p_value'] >= bonferroni_alpha:
            status = "REFUTED"
            summary = f"No significant improvement from polar coordinates (p={polar_vs_cartesian['p_value']:.2e})"
        else:
            status = "INCONCLUSIVE"
            summary = f"Effect too small (d={polar_vs_cartesian['cliffs_delta']:.3f}) to be practically meaningful"

    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Return results for logging
    return {
        'status': status,
        'summary': summary,
        'metrics': {
            'polar_mean': float(results['polar']['mean']),
            'radial_mean': float(results['radial']['mean']),
            'cartesian_mean': float(results['cartesian']['mean']),
            'polar_vs_cartesian_effect': float(polar_vs_cartesian['cliffs_delta']),
            'polar_vs_cartesian_p': float(polar_vs_cartesian['p_value']),
            'polar_vs_radial_effect': float(polar_vs_radial['cliffs_delta']),
            'polar_vs_radial_p': float(polar_vs_radial['p_value'])
        }
    }


if __name__ == "__main__":
    result = run_experiment()
