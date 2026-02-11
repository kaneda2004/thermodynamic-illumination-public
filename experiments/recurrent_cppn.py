#!/usr/bin/env python3
"""
RES-070: Test whether recurrent connections improve order generation in CPPNs.

Hypothesis: Recurrent connections (feedback loops) enable iterative refinement
of spatial structure, improving order metrics.

Methodology:
- Compare standard feedforward CPPN vs RecurrentCPPN with feedback loops
- RecurrentCPPN runs multiple iterations, allowing hidden states to influence output
- Measure order metrics: compressibility, symmetry, spectral coherence
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    compute_compressibility, compute_symmetry, compute_spectral_coherence
)


@dataclass
class RecurrentCPPN:
    """CPPN with recurrent connections allowing iterative refinement."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4
    hidden_id: int = 5  # Recurrent hidden node
    iterations: int = 3  # Number of recurrent passes

    def __post_init__(self):
        if not self.nodes:
            self.nodes = [
                Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
                Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
                Node(self.hidden_id, 'tanh', np.random.randn() * PRIOR_SIGMA),
                Node(self.output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA),
            ]
            # Feedforward: inputs -> hidden -> output
            for inp in self.input_ids:
                self.connections.append(Connection(inp, self.hidden_id, np.random.randn() * PRIOR_SIGMA))
            self.connections.append(Connection(self.hidden_id, self.output_id, np.random.randn() * PRIOR_SIGMA))

            # RECURRENT: hidden -> hidden (feedback loop)
            self.connections.append(Connection(self.hidden_id, self.hidden_id, np.random.randn() * PRIOR_SIGMA))
            # RECURRENT: output -> hidden (output feedback)
            self.connections.append(Connection(self.output_id, self.hidden_id, np.random.randn() * PRIOR_SIGMA))

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run recurrent iterations."""
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = {0: x, 1: y, 2: r, 3: bias}

        # Initialize hidden and output states
        hidden_val = np.zeros_like(x)
        output_val = np.zeros_like(x)

        for _ in range(self.iterations):
            # Compute hidden with recurrent inputs
            hidden_node = next(n for n in self.nodes if n.id == self.hidden_id)
            total_hidden = np.zeros_like(x) + hidden_node.bias

            for conn in self.connections:
                if conn.to_id == self.hidden_id and conn.enabled:
                    if conn.from_id in inputs:
                        total_hidden += inputs[conn.from_id] * conn.weight
                    elif conn.from_id == self.hidden_id:
                        total_hidden += hidden_val * conn.weight  # Recurrent
                    elif conn.from_id == self.output_id:
                        total_hidden += output_val * conn.weight  # Output feedback

            hidden_val = ACTIVATIONS[hidden_node.activation](total_hidden)

            # Compute output
            output_node = next(n for n in self.nodes if n.id == self.output_id)
            total_output = np.zeros_like(x) + output_node.bias
            for conn in self.connections:
                if conn.to_id == self.output_id and conn.enabled:
                    if conn.from_id == self.hidden_id:
                        total_output += hidden_val * conn.weight

            output_val = ACTIVATIONS[output_node.activation](total_output)

        return output_val

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def get_weights(self) -> np.ndarray:
        weights = [c.weight for c in self.connections if c.enabled]
        biases = [n.bias for n in self.nodes if n.id not in self.input_ids]
        return np.array(weights + biases)

    def set_weights(self, w: np.ndarray):
        idx = 0
        for c in self.connections:
            if c.enabled:
                c.weight = w[idx]
                idx += 1
        for n in self.nodes:
            if n.id not in self.input_ids:
                n.bias = w[idx]
                idx += 1


def compute_order(img: np.ndarray) -> float:
    """Combined order metric."""
    return (compute_compressibility(img) +
            compute_symmetry(img) +
            compute_spectral_coherence(img)) / 3


def run_experiment(n_samples=200, n_iterations_list=[1, 2, 3, 5, 8]):
    """Compare feedforward vs recurrent CPPN order generation."""
    np.random.seed(42)

    results = {
        'feedforward': [],
        'recurrent': {k: [] for k in n_iterations_list}
    }

    print("Generating samples...")

    # Feedforward CPPN baseline
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        results['feedforward'].append(compute_order(img))

    # Recurrent CPPN with varying iterations
    for n_iter in n_iterations_list:
        for _ in range(n_samples):
            rcppn = RecurrentCPPN(iterations=n_iter)
            img = rcppn.render(32)
            results['recurrent'][n_iter].append(compute_order(img))

    # Statistical analysis
    print("\n" + "="*60)
    print("RESULTS: Recurrent Connections in CPPN")
    print("="*60)

    ff_mean = np.mean(results['feedforward'])
    ff_std = np.std(results['feedforward'])
    print(f"\nFeedforward CPPN: mean={ff_mean:.4f}, std={ff_std:.4f}")

    best_iter = None
    best_effect = -np.inf
    best_p = 1.0

    for n_iter in n_iterations_list:
        rec_mean = np.mean(results['recurrent'][n_iter])
        rec_std = np.std(results['recurrent'][n_iter])

        # Mann-Whitney U test (non-parametric)
        stat, p_value = stats.mannwhitneyu(
            results['recurrent'][n_iter],
            results['feedforward'],
            alternative='greater'
        )

        # Cohen's d effect size
        pooled_std = np.sqrt((ff_std**2 + rec_std**2) / 2)
        effect_size = (rec_mean - ff_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\nRecurrent (iter={n_iter}): mean={rec_mean:.4f}, std={rec_std:.4f}")
        print(f"  vs Feedforward: effect_size={effect_size:.3f}, p={p_value:.4f}")

        if effect_size > best_effect:
            best_effect = effect_size
            best_iter = n_iter
            best_p = p_value

    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Bonferroni correction for multiple comparisons
    alpha = 0.01 / len(n_iterations_list)

    if best_p < alpha and best_effect > 0.5:
        status = "VALIDATED"
        summary = f"Recurrent ({best_iter} iter) improves order by d={best_effect:.2f}"
    elif best_p < 0.05 and best_effect > 0.2:
        status = "INCONCLUSIVE"
        summary = f"Weak effect d={best_effect:.2f}, p={best_p:.4f} (needs replication)"
    else:
        status = "REFUTED"
        summary = f"No significant improvement (best d={best_effect:.2f}, p={best_p:.4f})"

    print(f"Status: {status}")
    print(f"Summary: {summary}")

    return {
        'status': status,
        'best_iterations': best_iter,
        'effect_size': best_effect,
        'p_value': best_p,
        'feedforward_mean': ff_mean,
        'recurrent_best_mean': np.mean(results['recurrent'][best_iter]) if best_iter else None,
        'summary': summary
    }


if __name__ == "__main__":
    results = run_experiment()
    print(f"\n{results}")
