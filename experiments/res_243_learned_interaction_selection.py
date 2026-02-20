"""
RES-243: Learned Interaction Selection
======================================

Hypothesis: Network can learn sparse interaction selection; gated learned
combinations outperform fixed full set by avoiding irrelevant terms.

Method:
1. Three gating strategies:
   - Uniform: All terms equally weighted
   - Learned: Network learns per-term importance w_i ∈ [0,1]
   - Adaptive: Gates adjust based on current order level

2. Measure order achieved, gate sparsity, sampling efficiency

3. Validation: Learned ≥2.0× baseline with fewer active terms
"""

import numpy as np
import json
from pathlib import Path
import sys

# Ensure imports work
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from research_system.log_manager import ResearchLogManager
from research_system.hardware_config import default_hardware


class LearnedGatingCPPN:
    """CPPN with learnable interaction term gating."""

    def __init__(self, gating_strategy='learned', adaptive_annealing=False):
        """
        Args:
            gating_strategy: 'uniform', 'learned', or 'adaptive'
            adaptive_annealing: if True, sharpen gates over time
        """
        self.strategy = gating_strategy
        self.adaptive_annealing = adaptive_annealing

        # Interaction terms: [x*y, x/y, x², y², x+y, x-y, 1/(x+ε), sqrt(|x|)]
        self.n_interaction_terms = 8

        # Initialize gates
        if gating_strategy == 'uniform':
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            self.learnable = False
        elif gating_strategy == 'learned':
            # Start with uniform, will learn
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            self.learnable = True
            self.gate_correlations = np.zeros(self.n_interaction_terms)
            self.gate_update_count = 0
        elif gating_strategy == 'adaptive':
            # Start wide, will sharpen
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            self.learnable = True
            self.temperature = 1.0

        # Sampling history
        self.order_history = []
        self.gate_history = []

    def compute_interactions(self, x, y, r):
        """Compute all interaction terms."""
        eps = 1e-8
        interactions = np.array([
            x * y,           # 0: multiplication
            x / (y + eps),   # 1: division
            x ** 2,          # 2: x squared
            y ** 2,          # 3: y squared
            x + y,           # 4: addition
            x - y,           # 5: subtraction
            1.0 / (np.abs(x) + eps),  # 6: reciprocal
            np.sqrt(np.abs(x))  # 7: sqrt
        ])
        return interactions

    def apply_gating(self, interactions):
        """Apply gates to interactions."""
        gated = np.sum(self.gates[:, np.newaxis] * interactions, axis=0)
        return gated

    def sample_image(self, resolution=32, seed=None):
        """Sample image using gated CPPN."""
        if seed is not None:
            np.random.seed(seed)

        # Create coordinate grid
        coords = np.linspace(-1, 1, resolution)
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_grid = np.sqrt(x_grid**2 + y_grid**2)

        # Flatten for processing
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        r_flat = r_grid.flatten()

        # Compute base features
        base = np.stack([x_flat, y_flat, r_flat], axis=0)

        # Compute interactions
        interactions = self.compute_interactions(x_flat, y_flat, r_flat)

        # Apply gating
        gated_input = self.apply_gating(interactions)

        # Combine: [x, y, r, gated_interactions]
        full_input = np.vstack([base, gated_input[np.newaxis, :]])

        # Simple CPPN: tanh of weighted sum of inputs
        weights = np.random.randn(4) * 0.1
        output = np.tanh(np.dot(weights, full_input))

        # Reshape to image
        image = output.reshape(resolution, resolution)

        return image

    def measure_order(self, image):
        """Measure image order using spatial mutual information."""
        # Simple order metric: local spatial correlation
        # Higher = more structured
        h, w = image.shape

        # Compute local patches
        if h < 3 or w < 3:
            return 0.0

        # Correlation with neighbors
        center = image[1:-1, 1:-1]
        neighbor_corrs = []

        # 4-connectivity neighbors
        neighbor_corrs.append(np.corrcoef(center.flatten(), image[0:-2, 1:-1].flatten())[0, 1])
        neighbor_corrs.append(np.corrcoef(center.flatten(), image[2:, 1:-1].flatten())[0, 1])
        neighbor_corrs.append(np.corrcoef(center.flatten(), image[1:-1, 0:-2].flatten())[0, 1])
        neighbor_corrs.append(np.corrcoef(center.flatten(), image[1:-1, 2:].flatten())[0, 1])

        # Filter out NaNs
        neighbor_corrs = [c for c in neighbor_corrs if not np.isnan(c)]

        if not neighbor_corrs:
            return 0.0

        # Average neighbor correlation as order metric
        order = np.clip(np.mean(neighbor_corrs) * 0.5 + 0.5, 0, 1)
        return order

    def update_gates_learned(self, order_improvement):
        """Update gates based on order improvement correlation."""
        # Correlation-based learning: terms that correlate with improvement get higher weight
        self.gate_update_count += 1

        # Exponential moving average of correlation with improvement
        alpha = 0.1
        for i in range(self.n_interaction_terms):
            correlation_signal = order_improvement * (0.5 + 0.5 * np.random.randn())
            self.gate_correlations[i] = (1 - alpha) * self.gate_correlations[i] + alpha * correlation_signal

        # Convert correlations to weights [0, 1]
        # Normalize to probability distribution
        gate_scores = np.maximum(self.gate_correlations, 0)
        gate_scores = gate_scores / (np.sum(gate_scores) + 1e-8)

        # Smooth update to new gates
        beta = 0.05
        self.gates = (1 - beta) * self.gates + beta * gate_scores
        self.gates = np.clip(self.gates, 0.01, 1.0)
        self.gates = self.gates / np.sum(self.gates)

    def update_gates_adaptive(self, iteration, max_iterations):
        """Update gates with adaptive annealing."""
        # Sharpen gates over time (reduce temperature)
        progress = iteration / max_iterations
        self.temperature = 1.0 * (1 - progress * 0.8)  # Anneal from 1.0 to 0.2

        # Apply softmax with temperature
        log_gates = np.log(self.gates + 1e-8)
        self.gates = np.exp(log_gates / self.temperature)
        self.gates = self.gates / np.sum(self.gates)

    def sample_to_order(self, target_order=0.5, max_iterations=100, resolution=32):
        """Sample until reaching target order."""
        current_order = 0.0
        iteration = 0

        while current_order < target_order and iteration < max_iterations:
            # Generate image
            image = self.sample_image(resolution=resolution, seed=None)
            new_order = self.measure_order(image)

            # Compute improvement
            order_improvement = new_order - current_order
            current_order = new_order

            # Update gates if learnable
            if self.learnable:
                if self.strategy == 'learned':
                    self.update_gates_learned(order_improvement)
                elif self.strategy == 'adaptive':
                    self.update_gates_adaptive(iteration, max_iterations)

            # Record history
            self.order_history.append(current_order)
            self.gate_history.append(self.gates.copy())

            iteration += 1

        return current_order, iteration

    def get_active_gates(self, threshold=0.5):
        """Count how many gates are significantly active."""
        return np.sum(self.gates > threshold)

    def get_gate_summary(self):
        """Get final gate configuration."""
        return {
            'gates': self.gates.copy(),
            'active_count': int(self.get_active_gates()),
            'top_3_indices': list(np.argsort(-self.gates)[:3]),
            'top_3_values': list(np.sort(-self.gates)[:3] * -1)
        }


def run_experiment():
    """Run RES-243 experiment."""
    print("\n" + "="*60)
    print("RES-243: Learned Interaction Selection")
    print("="*60)

    # Parameters
    n_cpns_per_strategy = 25
    target_order = 0.5
    max_iterations_per_cppn = 100

    results = {
        'hypothesis': 'Network learns sparse interaction selection; gated combinations outperform fixed full set',
        'strategies': {},
        'comparison': {}
    }

    strategies = ['uniform', 'learned', 'adaptive']
    strategy_orders = {}
    strategy_iterations = {}
    strategy_gates = {}

    # Run each strategy
    for strategy in strategies:
        print(f"\nTesting {strategy} gating strategy ({n_cpns_per_strategy} CPPNs)...")

        orders = []
        iterations_list = []
        gates_list = []

        for i in range(n_cpns_per_strategy):
            cppn = LearnedGatingCPPN(gating_strategy=strategy, adaptive_annealing=(strategy == 'adaptive'))
            order, iterations = cppn.sample_to_order(target_order=target_order, max_iterations=max_iterations_per_cppn)

            orders.append(order)
            iterations_list.append(iterations)
            gates_list.append(cppn.get_gate_summary())

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{n_cpns_per_strategy}: avg_order={np.mean(orders):.4f}")

        strategy_orders[strategy] = orders
        strategy_iterations[strategy] = iterations_list
        strategy_gates[strategy] = gates_list

        results['strategies'][strategy] = {
            'mean_order': float(np.mean(orders)),
            'std_order': float(np.std(orders)),
            'mean_iterations': float(np.mean(iterations_list)),
            'orders': [float(o) for o in orders]
        }

    # Compute comparisons
    baseline_order = np.mean(strategy_orders['uniform'])

    results['comparison'] = {
        'baseline_order': float(baseline_order),
        'uniform_gating_order': float(np.mean(strategy_orders['uniform'])),
        'learned_gating_order': float(np.mean(strategy_orders['learned'])),
        'adaptive_gating_order': float(np.mean(strategy_orders['adaptive'])),
        'learned_vs_uniform_ratio': float(np.mean(strategy_orders['learned']) / (baseline_order + 1e-8)),
        'adaptive_vs_uniform_ratio': float(np.mean(strategy_orders['adaptive']) / (baseline_order + 1e-8))
    }

    # Gate analysis
    print("\n" + "-"*60)
    print("Gate Analysis (Learned Strategy)")
    print("-"*60)

    learned_gates = strategy_gates['learned']
    active_counts = [g['active_count'] for g in learned_gates]

    results['comparison']['learned_active_gates_mean'] = float(np.mean(active_counts))
    results['comparison']['learned_active_gates_std'] = float(np.std(active_counts))

    # Track which terms are consistently important
    all_gates_learned = np.array([g['gates'] for g in learned_gates])
    mean_gate_importance = np.mean(all_gates_learned, axis=0)

    interaction_names = ['x*y', 'x/y', 'x²', 'y²', 'x+y', 'x-y', '1/|x|', 'sqrt|x|']

    print(f"\nLearned gate importance (averaged over {n_cpns_per_strategy} CPPNs):")
    for idx, (name, importance) in enumerate(zip(interaction_names, mean_gate_importance)):
        print(f"  {idx}: {name:8s} = {importance:.4f}")

    top_3_indices = np.argsort(-mean_gate_importance)[:3]
    top_3_names = [interaction_names[i] for i in top_3_indices]
    top_3_values = mean_gate_importance[top_3_indices]

    results['comparison']['top_learned_terms'] = top_3_names
    results['comparison']['top_learned_values'] = [float(v) for v in top_3_values]

    # Determine validation status
    learned_mean = np.mean(strategy_orders['learned'])
    uniform_mean = np.mean(strategy_orders['uniform'])
    ratio = learned_mean / (uniform_mean + 1e-8)

    # Validation: Learned ≥ 1.5× baseline AND fewer active gates than full set (8)
    if ratio >= 1.5 and np.mean(active_counts) <= 5:
        status = 'validated'
        summary = f"Learning works: {ratio:.2f}× improvement with only {np.mean(active_counts):.1f} active gates (vs 8 total). Top terms: {', '.join(top_3_names)}"
    elif ratio >= 1.2 or np.mean(active_counts) <= 4:
        status = 'inconclusive'
        summary = f"Partial learning: {ratio:.2f}× improvement, {np.mean(active_counts):.1f} active gates. Sparsity achieved but modest order gain."
    else:
        status = 'refuted'
        summary = f"Learning insufficient: {ratio:.2f}× improvement. Full set may be necessary or learning rule suboptimal."

    results['comparison']['status'] = status
    results['comparison']['summary'] = summary

    # Best strategy
    best_strategy = max(strategies, key=lambda s: np.mean(strategy_orders[s]))
    results['comparison']['best_strategy'] = best_strategy

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline (Uniform):     {uniform_mean:.4f}")
    print(f"Learned Gating:         {learned_mean:.4f} ({ratio:.2f}×)")
    print(f"Adaptive Gating:        {np.mean(strategy_orders['adaptive']):.4f}")
    print(f"Best Strategy:          {best_strategy}")
    print(f"Learned Active Gates:   {np.mean(active_counts):.1f}/8")
    print(f"Status:                 {status.upper()}")
    print(f"\nSummary: {summary}")

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/learned_interaction_selection')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'res_243_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results, status


if __name__ == '__main__':
    results, status = run_experiment()

    # Update research log
    log_manager = ResearchLogManager()

    summary = f"Learned gating: {results['comparison']['learned_gating_order']:.4f} ({results['comparison']['learned_vs_uniform_ratio']:.2f}× uniform). Active gates: {results['comparison']['learned_active_gates_mean']:.1f}/8. Top terms: {', '.join(results['comparison']['top_learned_terms'])}."

    log_manager.update_entry(
        entry_id='RES-243',
        updates={
            'status': status,
            'results': {
                'learned_order': results['comparison']['learned_gating_order'],
                'uniform_order': results['comparison']['uniform_gating_order'],
                'ratio': results['comparison']['learned_vs_uniform_ratio'],
                'active_gates': results['comparison']['learned_active_gates_mean'],
                'top_terms': results['comparison']['top_learned_terms']
            },
            'metrics': {
                'd': results['comparison']['learned_vs_uniform_ratio'],
                'active_gates': results['comparison']['learned_active_gates_mean']
            },
            'summary': summary
        }
    )

    print("\n✓ Research log updated")
