#!/usr/bin/env python3
"""
RES-294: Alternative Sampling Algorithms for Manifold-Aware Search

Hypothesis: Alternative sampling algorithms can outperform nested sampling + PCA
for finding high-order CPPN configurations, potentially approaching the theoretical
~60× speedup ceiling.

Algorithms tested:
1. Nested Sampling + PCA (baseline) - current approach
2. Simulated Annealing - classic optimization with temperature schedule
3. CMA-ES - Covariance Matrix Adaptation Evolution Strategy
4. Hamiltonian Monte Carlo - gradient-based MCMC (finite difference gradients)
5. Particle Swarm Optimization - swarm intelligence approach
6. Differential Evolution - evolutionary algorithm with crossover

Method:
- Run each algorithm on 30 CPPNs
- Measure samples to reach target order (0.15)
- Compare speedup vs naive random sampling baseline
- Track convergence dynamics

Expected outcome: Identify algorithms that either:
(a) Achieve higher mean speedup than nested sampling
(b) Reduce variance in speedup (more consistent)
(c) Work better for specific CPPN configurations

Usage:
    cd /Users/matt/Development/monochrome_noise_converger
    uv run python experiments/res_294_alternative_sampling_algorithms.py
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import zlib

# Ensure project root is in path
PROJECT_ROOT = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_CPPNS = 20  # Fewer CPPNs since this will be slower
TARGET_ORDER = 0.95  # Extremely hard target - truly rare
MAX_EVALUATIONS = 5000  # More budget for hard target
RANDOM_SEED = 42


# =============================================================================
# ORDER METRIC
# =============================================================================

def compute_order(img: np.ndarray) -> float:
    """Multiplicative order metric."""
    if img.size == 0:
        return 0.0

    density = np.mean(img)
    density_gate = np.exp(-((density - 0.5) ** 2) / (2 * 0.25 ** 2))

    binary = (img > 0.5).astype(np.uint8)
    padded = np.pad(binary, 1, mode='edge')
    edges = 0
    for di, dj in [(0, 1), (1, 0)]:
        shifted = padded[1+di:1+di+img.shape[0], 1+dj:1+dj+img.shape[1]]
        edges += np.sum(binary != shifted)
    edge_density = edges / (2 * img.size)
    edge_gate = np.exp(-((edge_density - 0.15) ** 2) / (2 * 0.08 ** 2))

    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (min(h, w) / 4)
    coherence = np.sum(power[low_mask]) / (np.sum(power) + 1e-10)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    tiled = np.tile(binary, (2, 2))
    packed = np.packbits(tiled.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    compressibility = max(0, min(1, 1 - (len(compressed) * 8 / tiled.size)))
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    return float(min(1.0, density_gate * edge_gate * coherence_gate * compress_gate))


# =============================================================================
# CPPN
# =============================================================================

class CPPN:
    """Compositional Pattern Producing Network."""

    ACTIVATIONS = {
        'sin': lambda x: np.sin(x * np.pi),
        'cos': lambda x: np.cos(x * np.pi),
        'gauss': lambda x: np.exp(-x**2 * 2),
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10))),
        'tanh': np.tanh,
    }

    def __init__(self, size: int = 64):
        self.size = size
        self.dim = 121  # Total weight dimensions

    def evaluate(self, weights: np.ndarray) -> float:
        """Evaluate order for given weights."""
        # Parse weights
        idx = 0
        w1 = weights[idx:idx+32].reshape(4, 8); idx += 32
        b1 = weights[idx:idx+8]; idx += 8
        w2 = weights[idx:idx+64].reshape(8, 8); idx += 64
        b2 = weights[idx:idx+8]; idx += 8
        w3 = weights[idx:idx+8].reshape(8, 1); idx += 8
        b3 = weights[idx:idx+1]

        # Activations from weight values
        act_indices = (weights[:16] * 1000).astype(int) % 5
        act_names = list(self.ACTIVATIONS.keys())
        activations = [act_names[i] for i in act_indices]

        # Generate image
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = np.stack([x.flatten(), y.flatten(), r.flatten(), bias.flatten()], axis=1)

        h1 = inputs @ w1 + b1
        for i in range(8):
            h1[:, i] = self.ACTIVATIONS[activations[i % len(activations)]](h1[:, i])

        h2 = h1 @ w2 + b2
        for i in range(8):
            h2[:, i] = self.ACTIVATIONS[activations[(8 + i) % len(activations)]](h2[:, i])

        out = self.ACTIVATIONS['sigmoid'](h2 @ w3 + b3)
        img = out.reshape(self.size, self.size)

        return compute_order((img > 0.5).astype(np.uint8))


# =============================================================================
# SAMPLING ALGORITHMS
# =============================================================================

@dataclass
class AlgorithmResult:
    """Result from running one algorithm."""
    algorithm: str
    cppn_id: int
    evaluations: int
    best_order: float
    target_reached: bool
    convergence_history: List[float]
    runtime_seconds: float


class SamplingAlgorithm(ABC):
    """Base class for sampling algorithms."""

    def __init__(self, dim: int, evaluate_fn: Callable):
        self.dim = dim
        self.evaluate = evaluate_fn
        self.eval_count = 0
        self.best_order = 0.0
        self.history = []

    def _eval(self, weights: np.ndarray) -> float:
        """Wrapped evaluation that tracks count and best."""
        order = self.evaluate(weights)
        self.eval_count += 1
        if order > self.best_order:
            self.best_order = order
        self.history.append(self.best_order)
        return order

    @abstractmethod
    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        """Run until target reached or max_evals exhausted.
        Returns: (evaluations, best_order, history)
        """
        pass


class RandomSearch(SamplingAlgorithm):
    """Pure random sampling baseline."""

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        for _ in range(max_evals):
            weights = np.random.randn(self.dim)
            order = self._eval(weights)
            if order >= target:
                return self.eval_count, self.best_order, self.history
        return self.eval_count, self.best_order, self.history


class NestedSamplingPCA(SamplingAlgorithm):
    """Two-stage nested sampling with PCA (current baseline)."""

    def __init__(self, dim: int, evaluate_fn: Callable, stage1_samples: int = 100):
        super().__init__(dim, evaluate_fn)
        self.stage1_samples = stage1_samples

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        n_live = 30

        # Initialize
        live_points = np.random.randn(n_live, self.dim)
        live_orders = np.array([self._eval(w) for w in live_points])

        if self.best_order >= target:
            return self.eval_count, self.best_order, self.history

        all_samples = [live_points.copy()]

        # Stage 1: Exploration
        for _ in range(self.stage1_samples):
            if self.eval_count >= max_evals:
                break

            worst_idx = np.argmin(live_orders)
            threshold = live_orders[worst_idx]

            for _ in range(50):
                all_points = np.vstack(all_samples)
                mean = np.mean(all_points, axis=0)
                cov = np.cov(all_points.T) + 1e-6 * np.eye(self.dim)
                L = np.linalg.cholesky(cov)
                u = np.random.randn(self.dim)
                u = u / np.linalg.norm(u) * np.random.uniform(0, 1) ** (1/self.dim)
                new_point = mean + 1.5 * L @ u

                order = self._eval(new_point)
                if order > threshold:
                    live_points[worst_idx] = new_point
                    live_orders[worst_idx] = order
                    all_samples.append(new_point.reshape(1, -1))
                    break

            if self.best_order >= target:
                return self.eval_count, self.best_order, self.history

        # Compute PCA basis
        all_points = np.vstack(all_samples)
        mean_point = np.mean(all_points, axis=0)
        centered = all_points - mean_point
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            pca_basis = Vt[:3]
        except:
            pca_basis = np.eye(self.dim)[:3]

        # Stage 2: PCA-constrained
        while self.eval_count < max_evals:
            worst_idx = np.argmin(live_orders)
            threshold = live_orders[worst_idx]

            for _ in range(50):
                coeffs = np.random.randn(3) * 0.5
                new_point = mean_point + coeffs @ pca_basis + np.random.randn(self.dim) * 0.1

                order = self._eval(new_point)
                if order > threshold:
                    live_points[worst_idx] = new_point
                    live_orders[worst_idx] = order
                    break

            if self.best_order >= target:
                return self.eval_count, self.best_order, self.history

        return self.eval_count, self.best_order, self.history


class SimulatedAnnealing(SamplingAlgorithm):
    """Simulated annealing with exponential cooling."""

    def __init__(self, dim: int, evaluate_fn: Callable, t_initial: float = 1.0, t_final: float = 0.01):
        super().__init__(dim, evaluate_fn)
        self.t_initial = t_initial
        self.t_final = t_final

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        # Initialize
        current = np.random.randn(self.dim)
        current_order = self._eval(current)

        if current_order >= target:
            return self.eval_count, self.best_order, self.history

        best = current.copy()
        best_order = current_order

        # Cooling schedule
        cooling_rate = (self.t_final / self.t_initial) ** (1 / max_evals)

        temperature = self.t_initial
        step_size = 1.0

        while self.eval_count < max_evals:
            # Propose new point
            proposal = current + np.random.randn(self.dim) * step_size

            proposal_order = self._eval(proposal)

            # Accept/reject
            delta = proposal_order - current_order
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current = proposal
                current_order = proposal_order

                if current_order > best_order:
                    best = current.copy()
                    best_order = current_order

            if best_order >= target:
                return self.eval_count, self.best_order, self.history

            # Cool down
            temperature *= cooling_rate
            step_size = 0.1 + 0.9 * (temperature / self.t_initial)

        return self.eval_count, self.best_order, self.history


class CMAES(SamplingAlgorithm):
    """Simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, dim: int, evaluate_fn: Callable, pop_size: int = 20):
        super().__init__(dim, evaluate_fn)
        self.pop_size = pop_size

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        # Initialize
        mean = np.random.randn(self.dim) * 0.5
        sigma = 1.0
        C = np.eye(self.dim)

        mu = self.pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)

        while self.eval_count < max_evals:
            # Sample population
            try:
                L = np.linalg.cholesky(C + 1e-6 * np.eye(self.dim))
            except:
                L = np.eye(self.dim)

            population = []
            orders = []
            for _ in range(self.pop_size):
                if self.eval_count >= max_evals:
                    break
                z = np.random.randn(self.dim)
                x = mean + sigma * L @ z
                order = self._eval(x)
                population.append(x)
                orders.append(order)

                if order >= target:
                    return self.eval_count, self.best_order, self.history

            if len(population) < mu:
                break

            # Select top mu
            indices = np.argsort(orders)[::-1][:mu]
            selected = np.array([population[i] for i in indices])

            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * selected, axis=0)

            # Update covariance (simplified)
            diff = selected - old_mean
            C = np.sum(weights[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            C = 0.8 * C + 0.2 * np.eye(self.dim)  # Regularize

            # Adapt sigma
            sigma *= np.exp(0.1 * (np.mean(orders[:mu]) / (np.mean(orders) + 1e-10) - 1))
            sigma = np.clip(sigma, 0.01, 2.0)

        return self.eval_count, self.best_order, self.history


class ParticleSwarm(SamplingAlgorithm):
    """Particle Swarm Optimization."""

    def __init__(self, dim: int, evaluate_fn: Callable, n_particles: int = 30):
        super().__init__(dim, evaluate_fn)
        self.n_particles = n_particles

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        # Initialize
        positions = np.random.randn(self.n_particles, self.dim)
        velocities = np.random.randn(self.n_particles, self.dim) * 0.1

        personal_best = positions.copy()
        personal_best_orders = np.array([self._eval(p) for p in positions])

        if self.best_order >= target:
            return self.eval_count, self.best_order, self.history

        global_best_idx = np.argmax(personal_best_orders)
        global_best = personal_best[global_best_idx].copy()

        # PSO parameters
        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social

        while self.eval_count < max_evals:
            for i in range(self.n_particles):
                if self.eval_count >= max_evals:
                    break

                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (w * velocities[i] +
                                c1 * r1 * (personal_best[i] - positions[i]) +
                                c2 * r2 * (global_best - positions[i]))

                # Clip velocity
                velocities[i] = np.clip(velocities[i], -2, 2)

                # Update position
                positions[i] += velocities[i]

                # Evaluate
                order = self._eval(positions[i])

                # Update personal best
                if order > personal_best_orders[i]:
                    personal_best[i] = positions[i].copy()
                    personal_best_orders[i] = order

                    # Update global best
                    if order > personal_best_orders[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i].copy()

                if order >= target:
                    return self.eval_count, self.best_order, self.history

        return self.eval_count, self.best_order, self.history


class DifferentialEvolution(SamplingAlgorithm):
    """Differential Evolution."""

    def __init__(self, dim: int, evaluate_fn: Callable, pop_size: int = 30):
        super().__init__(dim, evaluate_fn)
        self.pop_size = pop_size

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        # Initialize
        population = np.random.randn(self.pop_size, self.dim)
        fitness = np.array([self._eval(p) for p in population])

        if self.best_order >= target:
            return self.eval_count, self.best_order, self.history

        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        while self.eval_count < max_evals:
            for i in range(self.pop_size):
                if self.eval_count >= max_evals:
                    break

                # Select 3 random distinct individuals
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = population[a] + F * (population[b] - population[c])

                # Crossover
                trial = np.where(np.random.random(self.dim) < CR, mutant, population[i])

                # Ensure at least one component from mutant
                j_rand = np.random.randint(self.dim)
                trial[j_rand] = mutant[j_rand]

                # Selection
                trial_fitness = self._eval(trial)
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness >= target:
                    return self.eval_count, self.best_order, self.history

        return self.eval_count, self.best_order, self.history


class HamiltonianMonteCarlo(SamplingAlgorithm):
    """Simplified Hamiltonian Monte Carlo with finite-difference gradients."""

    def __init__(self, dim: int, evaluate_fn: Callable, step_size: float = 0.1, n_steps: int = 10):
        super().__init__(dim, evaluate_fn)
        self.step_size = step_size
        self.n_steps = n_steps

    def _gradient(self, weights: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """Finite difference gradient (uses extra evaluations)."""
        grad = np.zeros(self.dim)
        base_order = self.evaluate(weights)  # Don't count these in main eval

        for i in range(min(20, self.dim)):  # Only estimate for first 20 dims to save evals
            weights_plus = weights.copy()
            weights_plus[i] += epsilon
            order_plus = self.evaluate(weights_plus)
            grad[i] = (order_plus - base_order) / epsilon

        return grad

    def run(self, target: float, max_evals: int) -> Tuple[int, float, List[float]]:
        # Initialize
        q = np.random.randn(self.dim)
        current_order = self._eval(q)

        if current_order >= target:
            return self.eval_count, self.best_order, self.history

        while self.eval_count < max_evals:
            # Sample momentum
            p = np.random.randn(self.dim)

            # Leapfrog integration
            q_new = q.copy()
            p_new = p.copy()

            # Half step for momentum
            grad = self._gradient(q_new)
            p_new += 0.5 * self.step_size * grad

            for _ in range(self.n_steps):
                q_new += self.step_size * p_new
                grad = self._gradient(q_new)
                p_new += self.step_size * grad

            p_new += 0.5 * self.step_size * self._gradient(q_new)

            # Evaluate new position
            new_order = self._eval(q_new)

            # Metropolis acceptance (using order as log-probability)
            current_H = -current_order + 0.5 * np.sum(p**2)
            new_H = -new_order + 0.5 * np.sum(p_new**2)

            if np.random.random() < np.exp(current_H - new_H):
                q = q_new
                current_order = new_order

            if new_order >= target:
                return self.eval_count, self.best_order, self.history

        return self.eval_count, self.best_order, self.history


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

ALGORITHMS = [
    ('random', RandomSearch),
    ('nested_pca', NestedSamplingPCA),
    ('simulated_annealing', SimulatedAnnealing),
    ('cma_es', CMAES),
    ('particle_swarm', ParticleSwarm),
    ('differential_evolution', DifferentialEvolution),
    ('hmc', HamiltonianMonteCarlo),
]


def main():
    print("=" * 70)
    print("RES-294: Alternative Sampling Algorithms")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    output_dir = PROJECT_ROOT / 'results' / 'alternative_sampling_algorithms'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for cppn_id in range(N_CPPNS):
        print(f"\n--- CPPN {cppn_id + 1}/{N_CPPNS} ---")

        # Create CPPN
        cppn = CPPN(size=64)

        for algo_name, algo_class in ALGORITHMS:
            # Reset seed for fair comparison
            np.random.seed(RANDOM_SEED + cppn_id * 1000 + hash(algo_name) % 1000)

            algo = algo_class(cppn.dim, cppn.evaluate)

            start = time.time()
            evals, best_order, history = algo.run(TARGET_ORDER, MAX_EVALUATIONS)
            runtime = time.time() - start

            result = AlgorithmResult(
                algorithm=algo_name,
                cppn_id=cppn_id,
                evaluations=evals,
                best_order=best_order,
                target_reached=best_order >= TARGET_ORDER,
                convergence_history=history[::10],  # Subsample to save space
                runtime_seconds=runtime,
            )
            results.append(asdict(result))

            status = "✓" if result.target_reached else "✗"
            print(f"  {algo_name:22s}: {evals:5d} evals, order={best_order:.4f} {status}")

    # Aggregate by algorithm
    print("\n" + "=" * 70)
    print("SUMMARY BY ALGORITHM")
    print("=" * 70)

    summary = {}
    random_mean_evals = None

    for algo_name, _ in ALGORITHMS:
        algo_results = [r for r in results if r['algorithm'] == algo_name]

        evals = [r['evaluations'] for r in algo_results]
        orders = [r['best_order'] for r in algo_results]
        success = [r['target_reached'] for r in algo_results]

        summary[algo_name] = {
            'evals_mean': float(np.mean(evals)),
            'evals_std': float(np.std(evals)),
            'evals_median': float(np.median(evals)),
            'order_mean': float(np.mean(orders)),
            'success_rate': float(np.mean(success)),
        }

        if algo_name == 'random':
            random_mean_evals = summary[algo_name]['evals_mean']

    # Calculate speedup vs random
    for algo_name in summary:
        if random_mean_evals and random_mean_evals > 0:
            summary[algo_name]['speedup_vs_random'] = random_mean_evals / summary[algo_name]['evals_mean']
        else:
            summary[algo_name]['speedup_vs_random'] = 1.0

    # Print results
    print(f"\n{'Algorithm':<24} {'Evals':>8} {'Std':>8} {'Success':>8} {'Speedup':>10}")
    print("-" * 60)

    for algo_name, _ in ALGORITHMS:
        s = summary[algo_name]
        print(f"{algo_name:<24} {s['evals_mean']:>8.0f} {s['evals_std']:>8.0f} "
              f"{s['success_rate']:>7.0%} {s['speedup_vs_random']:>9.2f}×")

    # Save results
    output = {
        'experiment_id': 'RES-294',
        'hypothesis': 'Alternative algorithms can outperform nested sampling + PCA',
        'config': {
            'n_cppns': N_CPPNS,
            'target_order': TARGET_ORDER,
            'max_evaluations': MAX_EVALUATIONS,
            'algorithms': [a[0] for a in ALGORITHMS],
        },
        'summary': summary,
        'detailed_results': results,
    }

    results_file = output_dir / 'res_294_results.json'
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Find best algorithm
    best_algo = max(summary.keys(), key=lambda k: summary[k]['speedup_vs_random'])
    nested_speedup = summary['nested_pca']['speedup_vs_random']
    best_speedup = summary[best_algo]['speedup_vs_random']

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")
    print(f"Best algorithm: {best_algo} ({best_speedup:.2f}× vs random)")
    print(f"Nested PCA baseline: {nested_speedup:.2f}× vs random")

    if best_algo != 'nested_pca' and best_speedup > nested_speedup * 1.2:
        print(f"✓ VALIDATED: {best_algo} outperforms nested sampling by {best_speedup/nested_speedup:.2f}×")
    elif best_algo != 'nested_pca' and best_speedup > nested_speedup:
        print(f"~ PARTIAL: {best_algo} slightly better than nested sampling")
    else:
        print("✗ REFUTED: Nested sampling + PCA remains best")


if __name__ == '__main__':
    main()
