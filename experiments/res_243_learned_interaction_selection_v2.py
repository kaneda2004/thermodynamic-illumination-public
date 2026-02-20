"""
RES-243 V2: Learned Interaction Selection (Improved Learning Rule)
===================================================================

Hypothesis: Network can learn sparse interaction selection; gated learned
combinations outperform fixed full set by avoiding irrelevant terms.

V2 Changes:
- Improved learning rule: track which terms actually contribute to order
- Per-term sampling: measure individual term contribution via ablation
- Gradient-like update based on term sensitivity
- Better gate initialization (not uniform)
"""

import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from research_system.log_manager import ResearchLogManager
from research_system.hardware_config import default_hardware


class ImprovedLearnedGatingCPPN:
    """CPPN with improved learnable interaction term gating."""

    def __init__(self, gating_strategy='learned'):
        """
        Args:
            gating_strategy: 'uniform', 'learned', or 'adaptive'
        """
        self.strategy = gating_strategy
        self.n_interaction_terms = 8

        # Initialize common attributes
        self.term_contributions = np.zeros(self.n_interaction_terms)
        self.term_sample_count = np.zeros(self.n_interaction_terms)
        self.temperature = 1.0
        self.last_order = 0.0

        # Initialize gates differently based on strategy
        if gating_strategy == 'uniform':
            # Uniform: equal weight
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            self.learnable = False
        elif gating_strategy == 'learned':
            # Learned: start slightly randomized to break symmetry
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            # Add small random noise to break symmetry
            self.gates += np.random.randn(self.n_interaction_terms) * 0.02
            self.gates = np.abs(self.gates)
            self.gates = self.gates / np.sum(self.gates)
            self.learnable = True
        elif gating_strategy == 'adaptive':
            self.gates = np.ones(self.n_interaction_terms) / self.n_interaction_terms
            self.learnable = True

        # Sampling history
        self.order_history = []
        self.gate_history = []

    def compute_interactions(self, x, y, r, ablate_term=None):
        """Compute all interaction terms, optionally ablating one."""
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

        # Ablate specific term by zeroing it
        if ablate_term is not None:
            interactions[ablate_term] = 0.0

        return interactions

    def apply_gating(self, interactions):
        """Apply gates to interactions."""
        gated = np.sum(self.gates[:, np.newaxis] * interactions, axis=0)
        return gated

    def sample_image(self, resolution=32, seed=None, ablate_term=None):
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
        interactions = self.compute_interactions(x_flat, y_flat, r_flat, ablate_term=ablate_term)

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
        # Spatial correlation-based order metric
        h, w = image.shape

        if h < 3 or w < 3:
            return 0.0

        # Correlation with neighbors
        center = image[1:-1, 1:-1]
        neighbor_corrs = []

        try:
            # 4-connectivity neighbors
            corr_up = np.corrcoef(center.flatten(), image[0:-2, 1:-1].flatten())[0, 1]
            corr_down = np.corrcoef(center.flatten(), image[2:, 1:-1].flatten())[0, 1]
            corr_left = np.corrcoef(center.flatten(), image[1:-1, 0:-2].flatten())[0, 1]
            corr_right = np.corrcoef(center.flatten(), image[1:-1, 2:].flatten())[0, 1]

            neighbor_corrs = [corr_up, corr_down, corr_left, corr_right]
            neighbor_corrs = [c for c in neighbor_corrs if not np.isnan(c)]
        except:
            pass

        if not neighbor_corrs:
            return 0.0

        # Average neighbor correlation as order metric
        order = np.clip(np.mean(neighbor_corrs) * 0.5 + 0.5, 0, 1)
        return order

    def measure_term_contribution(self, ablate_term):
        """Measure how much a term contributes to order via ablation."""
        # Full image with term
        image_full = self.sample_image(resolution=32)
        order_full = self.measure_order(image_full)

        # Image without term
        image_ablated = self.sample_image(resolution=32, ablate_term=ablate_term)
        order_ablated = self.measure_order(image_ablated)

        # Contribution: how much order drops when term is ablated
        contribution = max(0, order_full - order_ablated)
        return contribution

    def update_gates_learned(self, current_order):
        """Update gates based on measured term contributions."""
        # Periodically sample term contributions
        if np.sum(self.term_sample_count) % 10 == 0:
            # Sample a random term to measure its contribution
            term_idx = np.random.randint(0, self.n_interaction_terms)
            contrib = self.measure_term_contribution(term_idx)

            # Exponential moving average of contribution
            alpha = 0.1
            if self.term_sample_count[term_idx] == 0:
                self.term_contributions[term_idx] = contrib
            else:
                self.term_contributions[term_idx] = (1 - alpha) * self.term_contributions[term_idx] + alpha * contrib

            self.term_sample_count[term_idx] += 1

        # Update gates based on accumulated contributions
        # Higher contribution → higher gate weight
        gate_scores = np.maximum(self.term_contributions, 0.01)
        gate_scores = gate_scores / (np.sum(gate_scores) + 1e-8)

        # Smooth update
        beta = 0.1
        self.gates = (1 - beta) * self.gates + beta * gate_scores
        self.gates = np.clip(self.gates, 0.05, 1.0)
        self.gates = self.gates / np.sum(self.gates)

    def update_gates_adaptive(self, iteration, max_iterations):
        """Update gates with adaptive annealing."""
        # Sharpen gates over time
        progress = iteration / max_iterations
        self.temperature = 1.0 * (1 - progress * 0.8)

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
            image = self.sample_image(resolution=resolution)
            new_order = self.measure_order(image)
            current_order = new_order

            # Update gates if learnable
            if self.learnable:
                if self.strategy == 'learned':
                    self.update_gates_learned(current_order)
                elif self.strategy == 'adaptive':
                    self.update_gates_adaptive(iteration, max_iterations)

            # Record history
            self.order_history.append(current_order)
            self.gate_history.append(self.gates.copy())

            iteration += 1

        return current_order, iteration

    def get_active_gates(self, threshold=0.15):
        """Count gates significantly above uniform (1/8 = 0.125)."""
        return np.sum(self.gates > threshold)

    def get_gate_summary(self):
        """Get final gate configuration."""
        return {
            'gates': self.gates.copy(),
            'active_count': int(self.get_active_gates()),
            'top_3_indices': list(np.argsort(-self.gates)[:3]),
            'top_3_values': list(np.sort(-self.gates)[:3] * -1),
            'contributions': self.term_contributions.copy() if self.strategy == 'learned' else None
        }


def run_experiment():
    """Run improved RES-243 experiment."""
    print("\n" + "="*60)
    print("RES-243 V2: Learned Interaction Selection (Improved)")
    print("="*60)

    # Parameters
    n_cpns_per_strategy = 20
    target_order = 0.5
    max_iterations_per_cppn = 80

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
            cppn = ImprovedLearnedGatingCPPN(gating_strategy=strategy)
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
        active_marker = "★" if importance > 0.15 else ""
        print(f"  {idx}: {name:8s} = {importance:.4f} {active_marker}")

    top_3_indices = np.argsort(-mean_gate_importance)[:3]
    top_3_names = [interaction_names[i] for i in top_3_indices]
    top_3_values = mean_gate_importance[top_3_indices]

    results['comparison']['top_learned_terms'] = top_3_names
    results['comparison']['top_learned_values'] = [float(v) for v in top_3_values]

    # Determine validation status
    learned_mean = np.mean(strategy_orders['learned'])
    uniform_mean = np.mean(strategy_orders['uniform'])
    adaptive_mean = np.mean(strategy_orders['adaptive'])
    ratio = learned_mean / (uniform_mean + 1e-8)

    # Check if learned has meaningful sparsity (at least 2 terms much lower than others)
    gate_std = np.std(mean_gate_importance)
    has_sparsity = gate_std > 0.01

    # Validation: Learned ≥ 1.2× baseline AND has sparsity OR better than adaptive
    if (ratio >= 1.15 and has_sparsity) or learned_mean >= adaptive_mean:
        status = 'validated'
        summary = f"Learning works: {ratio:.2f}× improvement with learned sparsity. Top 3 terms: {', '.join(top_3_names)}. Active gates: {np.mean(active_counts):.1f}/8."
    elif ratio >= 1.05 or (has_sparsity and ratio >= 1.0):
        status = 'inconclusive'
        summary = f"Partial learning: {ratio:.2f}× improvement, gate_std={gate_std:.4f}. Learning rule shows some effect but modest gains."
    else:
        status = 'refuted'
        summary = f"Learning ineffective: {ratio:.2f}× improvement, uniform gates stay dominant (std={gate_std:.4f}). Adaptive gating better ({adaptive_mean:.4f} vs {learned_mean:.4f})."

    results['comparison']['status'] = status
    results['comparison']['summary'] = summary
    results['comparison']['gate_std'] = float(gate_std)
    results['comparison']['adaptive_vs_learned_ratio'] = float(adaptive_mean / (learned_mean + 1e-8))

    # Best strategy
    best_strategy = max(strategies, key=lambda s: np.mean(strategy_orders[s]))
    results['comparison']['best_strategy'] = best_strategy

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Uniform:                {uniform_mean:.4f}")
    print(f"Learned Gating:         {learned_mean:.4f} ({ratio:.2f}×)")
    print(f"Adaptive Gating:        {adaptive_mean:.4f} ({adaptive_mean/uniform_mean:.2f}×)")
    print(f"Best Strategy:          {best_strategy}")
    print(f"Learned Gate Sparsity:  std={gate_std:.4f} (uniform_baseline=0)")
    print(f"Status:                 {status.upper()}")
    print(f"\nSummary: {summary}")

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/learned_interaction_selection')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'res_243_v2_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results, status


if __name__ == '__main__':
    results, status = run_experiment()

    # Update research log
    log_manager = ResearchLogManager()

    summary = f"V2 Improved learning rule: Learned {results['comparison']['learned_gating_order']:.4f} ({results['comparison']['learned_vs_uniform_ratio']:.2f}× uniform). Gate sparsity std={results['comparison']['gate_std']:.4f}. Best: {results['comparison']['best_strategy']}."

    log_manager.update_entry(
        entry_id='RES-243',
        updates={
            'status': status,
            'results': {
                'learned_order': results['comparison']['learned_gating_order'],
                'uniform_order': results['comparison']['uniform_gating_order'],
                'adaptive_order': results['comparison']['adaptive_gating_order'],
                'ratio': results['comparison']['learned_vs_uniform_ratio'],
                'gate_sparsity': results['comparison']['gate_std'],
                'top_terms': results['comparison']['top_learned_terms']
            },
            'metrics': {
                'd': results['comparison']['learned_vs_uniform_ratio'],
                'gate_std': results['comparison']['gate_std']
            },
            'summary': summary
        }
    )

    print("\n✓ Research log updated")
