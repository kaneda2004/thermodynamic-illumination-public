#!/usr/bin/env python3
"""
RES-083: Test whether skip connections (input-to-output) improve CPPN order generation.

Hypothesis: Direct input-to-output connections provide better gradient paths,
enabling higher order images more efficiently.

Methodology:
- Compare "with_skip" (input->output direct) vs "no_skip" (input->hidden->output only)
- Both have same total parameter count
- Measure: mean order, max order achieved, samples to reach order threshold
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from dataclasses import dataclass, field
from core.thermo_sampler_v3 import (
    Node, Connection, CPPN, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA
)

# Use the validated multiplicative order metric
compute_order = order_multiplicative

@dataclass
class CPPNWithSkip:
    """CPPN with direct input-to-output skip connections."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4
    hidden_id: int = 5

    def __post_init__(self):
        if not self.nodes:
            # 4 inputs + 1 hidden + 1 output = 6 nodes
            self.nodes = [
                Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
                Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
                Node(5, 'tanh', np.random.randn() * PRIOR_SIGMA),  # Hidden
                Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # Output
            ]
            # Skip connections: input -> output (4 connections)
            for inp in self.input_ids:
                self.connections.append(Connection(inp, self.output_id, np.random.randn() * PRIOR_SIGMA))
            # Input -> hidden (4 connections)
            for inp in self.input_ids:
                self.connections.append(Connection(inp, self.hidden_id, np.random.randn() * PRIOR_SIGMA))
            # Hidden -> output (1 connection)
            self.connections.append(Connection(self.hidden_id, self.output_id, np.random.randn() * PRIOR_SIGMA))

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}

        # Hidden node first
        hidden_node = next(n for n in self.nodes if n.id == self.hidden_id)
        total = np.zeros_like(x) + hidden_node.bias
        for conn in self.connections:
            if conn.to_id == self.hidden_id and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[self.hidden_id] = ACTIVATIONS[hidden_node.activation](total)

        # Output node
        output_node = next(n for n in self.nodes if n.id == self.output_id)
        total = np.zeros_like(x) + output_node.bias
        for conn in self.connections:
            if conn.to_id == self.output_id and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[self.output_id] = ACTIVATIONS[output_node.activation](total)

        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)


@dataclass
class CPPNNoSkip:
    """CPPN without skip connections - all paths go through hidden layer."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4
    hidden_ids: list = field(default_factory=lambda: [5, 6])

    def __post_init__(self):
        if not self.nodes:
            # 4 inputs + 2 hidden + 1 output = 7 nodes
            # 2 hidden nodes to have comparable expressiveness
            self.nodes = [
                Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
                Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
                Node(5, 'tanh', np.random.randn() * PRIOR_SIGMA),  # Hidden 1
                Node(6, 'sin', np.random.randn() * PRIOR_SIGMA),   # Hidden 2
                Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # Output
            ]
            # Input -> hidden1 (4 connections)
            for inp in self.input_ids:
                self.connections.append(Connection(inp, 5, np.random.randn() * PRIOR_SIGMA))
            # Input -> hidden2 (4 connections)
            for inp in self.input_ids:
                self.connections.append(Connection(inp, 6, np.random.randn() * PRIOR_SIGMA))
            # Hidden -> output (2 connections)
            self.connections.append(Connection(5, self.output_id, np.random.randn() * PRIOR_SIGMA))
            self.connections.append(Connection(6, self.output_id, np.random.randn() * PRIOR_SIGMA))

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}

        # Hidden nodes
        for hid in self.hidden_ids:
            hidden_node = next(n for n in self.nodes if n.id == hid)
            total = np.zeros_like(x) + hidden_node.bias
            for conn in self.connections:
                if conn.to_id == hid and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[hid] = ACTIVATIONS[hidden_node.activation](total)

        # Output node
        output_node = next(n for n in self.nodes if n.id == self.output_id)
        total = np.zeros_like(x) + output_node.bias
        for conn in self.connections:
            if conn.to_id == self.output_id and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[self.output_id] = ACTIVATIONS[output_node.activation](total)

        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)


def run_experiment(n_samples=1000, n_trials=5):
    """Compare order distributions between skip and no-skip architectures."""
    from scipy import stats

    results = {
        'with_skip': {'orders': [], 'high_order_count': []},
        'no_skip': {'orders': [], 'high_order_count': []}
    }

    threshold = 0.7  # "High order" threshold

    for trial in range(n_trials):
        np.random.seed(42 + trial)

        # Sample with skip connections
        skip_orders = []
        for _ in range(n_samples):
            cppn = CPPNWithSkip()
            img = cppn.render(32)
            order = compute_order(img)
            skip_orders.append(order)
        results['with_skip']['orders'].extend(skip_orders)
        results['with_skip']['high_order_count'].append(sum(1 for o in skip_orders if o > threshold))

        # Sample without skip connections
        noskip_orders = []
        for _ in range(n_samples):
            cppn = CPPNNoSkip()
            img = cppn.render(32)
            order = compute_order(img)
            noskip_orders.append(order)
        results['no_skip']['orders'].extend(noskip_orders)
        results['no_skip']['high_order_count'].append(sum(1 for o in noskip_orders if o > threshold))

    # Statistical analysis
    skip_orders = np.array(results['with_skip']['orders'])
    noskip_orders = np.array(results['no_skip']['orders'])

    # Mann-Whitney U test (non-parametric)
    stat, pvalue = stats.mannwhitneyu(skip_orders, noskip_orders, alternative='two-sided')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(skip_orders) + np.var(noskip_orders)) / 2)
    cohens_d = (np.mean(skip_orders) - np.mean(noskip_orders)) / (pooled_std + 1e-10)

    # High order rates
    skip_high = np.mean(results['with_skip']['high_order_count']) / n_samples
    noskip_high = np.mean(results['no_skip']['high_order_count']) / n_samples

    print("=" * 60)
    print("RES-083: Skip Connection Impact on Order Generation")
    print("=" * 60)
    print(f"\nWith Skip Connections:")
    print(f"  Mean order: {np.mean(skip_orders):.4f} +/- {np.std(skip_orders):.4f}")
    print(f"  Max order:  {np.max(skip_orders):.4f}")
    print(f"  High order rate (>{threshold}): {skip_high:.2%}")
    print(f"\nWithout Skip Connections:")
    print(f"  Mean order: {np.mean(noskip_orders):.4f} +/- {np.std(noskip_orders):.4f}")
    print(f"  Max order:  {np.max(noskip_orders):.4f}")
    print(f"  High order rate (>{threshold}): {noskip_high:.2%}")
    print(f"\nStatistical Tests:")
    print(f"  Mann-Whitney U p-value: {pvalue:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Significant (p<0.01): {pvalue < 0.01}")
    print(f"  Large effect (|d|>0.5): {abs(cohens_d) > 0.5}")

    # Determine verdict
    if pvalue < 0.01 and abs(cohens_d) > 0.5:
        if cohens_d > 0:
            verdict = "VALIDATED - Skip connections improve order"
        else:
            verdict = "REFUTED - Skip connections reduce order"
    elif pvalue < 0.01:
        verdict = "INCONCLUSIVE - Significant but small effect"
    else:
        verdict = "REFUTED - No significant difference"

    print(f"\nVERDICT: {verdict}")

    return {
        'skip_mean': float(np.mean(skip_orders)),
        'skip_std': float(np.std(skip_orders)),
        'noskip_mean': float(np.mean(noskip_orders)),
        'noskip_std': float(np.std(noskip_orders)),
        'pvalue': float(pvalue),
        'cohens_d': float(cohens_d),
        'skip_high_rate': float(skip_high),
        'noskip_high_rate': float(noskip_high),
        'verdict': verdict
    }


if __name__ == '__main__':
    results = run_experiment(n_samples=1000, n_trials=5)
