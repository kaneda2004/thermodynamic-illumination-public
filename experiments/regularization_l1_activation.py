#!/usr/bin/env python3
"""
RES-114: L1 Regularization Effects on CPPN Order

Hypothesis: L1 weight regularization during CPPN inference (adding L1 penalty
to activations) increases output image order by promoting sparse internal
representations.

Approach:
- Generate CPPNs with varying L1 regularization strengths on weights
- Compare order distributions across regularization levels
- Statistical analysis with Spearman correlation and Mann-Whitney U test

Author: Research System
"""

import sys
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, ACTIVATIONS, Node, Connection, PRIOR_SIGMA
)


@dataclass
class RegularizedCPPN:
    """CPPN with L1 weight regularization applied during weight initialization."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4
    l1_strength: float = 0.0  # L1 regularization strength

    def __post_init__(self):
        if not self.nodes:
            self.nodes = [
                Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
                Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
                Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),
            ]
            for inp in self.input_ids:
                self.connections.append(Connection(inp, self.output_id, np.random.randn() * PRIOR_SIGMA))

    def _apply_l1_shrinkage(self, weights: np.ndarray) -> np.ndarray:
        """Apply soft-thresholding (L1 proximal operator) to weights."""
        # Soft-thresholding: sign(w) * max(|w| - lambda, 0)
        return np.sign(weights) * np.maximum(np.abs(weights) - self.l1_strength, 0)

    def _get_eval_order(self) -> list[int]:
        hidden_ids = sorted([n.id for n in self.nodes
                            if n.id not in self.input_ids and n.id != self.output_id])
        return hidden_ids + [self.output_id]

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}

        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            # Apply L1 shrinkage to bias
            shrunk_bias = self._apply_l1_shrinkage(np.array([node.bias]))[0] if self.l1_strength > 0 else node.bias
            total = np.zeros_like(x) + shrunk_bias

            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    # Apply L1 shrinkage to weight
                    shrunk_weight = self._apply_l1_shrinkage(np.array([conn.weight]))[0] if self.l1_strength > 0 else conn.weight
                    total += values[conn.from_id] * shrunk_weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)


def generate_regularized_cppn(l1_strength: float, hidden_nodes: int = 2) -> RegularizedCPPN:
    """Generate a CPPN with given L1 regularization strength."""
    cppn = RegularizedCPPN(l1_strength=l1_strength)

    # Add hidden nodes for more interesting patterns
    activations = ['sin', 'tanh', 'relu', 'gauss', 'sigmoid']
    next_id = 5
    hidden_ids = []

    for _ in range(hidden_nodes):
        act = np.random.choice(activations)
        cppn.nodes.append(Node(next_id, act, np.random.randn() * PRIOR_SIGMA))
        hidden_ids.append(next_id)
        next_id += 1

    # Connect inputs to hidden
    for inp_id in cppn.input_ids:
        for hid_id in hidden_ids:
            if np.random.rand() < 0.5:
                cppn.connections.append(Connection(inp_id, hid_id, np.random.randn() * PRIOR_SIGMA))

    # Connect hidden to output
    for hid_id in hidden_ids:
        cppn.connections.append(Connection(hid_id, cppn.output_id, np.random.randn() * PRIOR_SIGMA))

    # Some hidden-to-hidden connections
    for i, h1 in enumerate(hidden_ids):
        for h2 in hidden_ids[i+1:]:
            if np.random.rand() < 0.3:
                cppn.connections.append(Connection(h1, h2, np.random.randn() * PRIOR_SIGMA))

    return cppn


def run_experiment(n_samples: int = 500, seed: int = 42):
    """Run the L1 regularization experiment."""
    np.random.seed(seed)

    # L1 strengths to test (0 = no regularization, increasing sparsity pressure)
    l1_strengths = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    results = {strength: [] for strength in l1_strengths}
    weight_sparsities = {strength: [] for strength in l1_strengths}

    print(f"Generating {n_samples} CPPNs per L1 strength level...")

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}")

        # Generate base random weights once
        np.random.seed(seed + i)
        base_weights = {
            'hidden_acts': [np.random.choice(['sin', 'tanh', 'relu', 'gaussian', 'sigmoid']) for _ in range(2)],
            'hidden_biases': [np.random.randn() * PRIOR_SIGMA for _ in range(2)],
            'connections': np.random.randn(20) * PRIOR_SIGMA,  # More than needed
            'conn_mask': np.random.rand(20) < 0.5,
            'output_bias': np.random.randn() * PRIOR_SIGMA,
        }

        for strength in l1_strengths:
            # Create CPPN with this L1 strength but same underlying weights
            np.random.seed(seed + i)  # Reset to same state
            cppn = generate_regularized_cppn(strength, hidden_nodes=2)

            # Render and compute order
            img = cppn.render(32)
            order = order_multiplicative(img)
            results[strength].append(order)

            # Compute effective weight sparsity after L1 shrinkage
            weights = np.array([c.weight for c in cppn.connections if c.enabled])
            biases = np.array([n.bias for n in cppn.nodes if n.id not in cppn.input_ids])
            all_params = np.concatenate([weights, biases])

            if strength > 0:
                shrunk = np.sign(all_params) * np.maximum(np.abs(all_params) - strength, 0)
                sparsity = np.mean(np.abs(shrunk) < 0.01)  # Effectively zero
            else:
                sparsity = np.mean(np.abs(all_params) < 0.01)
            weight_sparsities[strength].append(sparsity)

    # Statistical analysis
    print("\n" + "="*60)
    print("RESULTS: L1 Regularization Effects on CPPN Order")
    print("="*60)

    for strength in l1_strengths:
        orders = results[strength]
        sparsities = weight_sparsities[strength]
        print(f"\nL1={strength:.1f}: order={np.mean(orders):.4f} +/- {np.std(orders):.4f}, "
              f"sparsity={np.mean(sparsities):.3f}")

    # Correlation: L1 strength vs order (across all samples)
    all_strengths = []
    all_orders = []
    for strength in l1_strengths:
        all_strengths.extend([strength] * len(results[strength]))
        all_orders.extend(results[strength])

    rho, p_val = stats.spearmanr(all_strengths, all_orders)
    print(f"\nSpearman correlation (L1 strength vs order):")
    print(f"  rho = {rho:.4f}, p = {p_val:.2e}")

    # Compare no-regularization (L1=0) vs high-regularization (L1=2.0)
    u_stat, p_mann = stats.mannwhitneyu(results[0.0], results[2.0], alternative='two-sided')

    # Effect size (Cohen's d)
    mean_diff = np.mean(results[2.0]) - np.mean(results[0.0])
    pooled_std = np.sqrt((np.var(results[0.0]) + np.var(results[2.0])) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    print(f"\nMann-Whitney U test (L1=0 vs L1=2.0):")
    print(f"  U = {u_stat:.1f}, p = {p_mann:.2e}")
    print(f"  Cohen's d = {cohens_d:.4f}")

    # Determine result
    print("\n" + "="*60)
    if p_val < 0.01 and abs(rho) > 0.1:
        if rho > 0:
            print("HYPOTHESIS: VALIDATED - L1 regularization INCREASES order")
        else:
            print("HYPOTHESIS: REFUTED - L1 regularization DECREASES order")
    else:
        print("HYPOTHESIS: REFUTED - No significant effect of L1 regularization")
    print("="*60)

    return {
        'l1_strengths': l1_strengths,
        'results': {k: np.array(v) for k, v in results.items()},
        'sparsities': {k: np.array(v) for k, v in weight_sparsities.items()},
        'rho': rho,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'mean_orders': {k: np.mean(v) for k, v in results.items()},
    }


if __name__ == '__main__':
    results = run_experiment(n_samples=500, seed=42)

    print("\n\nFINAL METRICS:")
    print(f"  effect_size (Cohen's d): {results['cohens_d']:.4f}")
    print(f"  p_value: {results['p_value']:.2e}")
    print(f"  correlation (rho): {results['rho']:.4f}")
