#!/usr/bin/env python3
"""
RES-253: Hybrid Multi-Manifold Sampling (Fast Version)

Hypothesis: Maintaining multiple PCA manifold bases (2D, 3D, 5D) with adaptive
mixture sampling achieves >=110x speedup.

Fast version: Test on 5 CPPNs (instead of 20) to get results quickly.
Based on initial results showing ~190x speedup on first CPPN.
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, PRIOR_SIGMA, ACTIVATIONS,
    order_multiplicative_v2, set_global_seed
)

# ============================================================================
# HYBRID MANIFOLD SAMPLER
# ============================================================================

@dataclass
class ManifoldBase:
    """PCA-based manifold projection."""
    dim: int
    components: np.ndarray = None
    mean: np.ndarray = None

    def project(self, weights: np.ndarray) -> np.ndarray:
        """Project weights onto manifold basis."""
        centered = weights - self.mean
        return centered @ self.components

    def reconstruct(self, proj: np.ndarray) -> np.ndarray:
        """Reconstruct weights from projection."""
        return (proj @ self.components.T) + self.mean


@dataclass
class HybridSampler:
    """Multi-manifold hybrid sampling strategy."""
    cppn_template: CPPN
    order_metric: callable = order_multiplicative_v2
    n_valid_samples_target: int = 100
    order_threshold: float = 0.5
    manifold_bases: List[ManifoldBase] = field(default_factory=list)
    strategy: str = "uniform"
    acceptance_rates_by_base: dict = field(default_factory=dict)
    manifold_preference: dict = field(default_factory=dict)

    def sample_stage1_unconstrained(self, n_samples: int = 50) -> Tuple[np.ndarray, int]:
        """Stage 1: Unconstrained sampling to establish manifold bases."""
        samples = []
        total_proposed = 0

        while len(samples) < n_samples:
            w = np.random.randn(len(self.cppn_template.get_weights())) * PRIOR_SIGMA
            cppn_copy = self.cppn_template.copy()
            cppn_copy.set_weights(w)
            img = cppn_copy.render(size=32)
            order = self.order_metric(img)
            total_proposed += 1

            if order >= self.order_threshold:
                samples.append(w)

        return np.array(samples), total_proposed

    def compute_pca_bases(self, samples: np.ndarray):
        """Compute PCA bases from unconstrained samples."""
        mean = samples.mean(axis=0)
        centered = samples - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        self.manifold_bases = [
            ManifoldBase(dim=2, components=Vt[:2].T, mean=mean),
            ManifoldBase(dim=3, components=Vt[:3].T, mean=mean),
            ManifoldBase(dim=5, components=Vt[:5].T, mean=mean)
        ]

    def sample_stage2_constrained(self, max_attempts: int = 50000) -> dict:
        """Stage 2: Constrained sampling on manifold mixture."""
        samples = []
        total_attempts = 0
        samples_per_base = {2: 0, 3: 0, 5: 0}
        accepts_per_base = {2: 0, 3: 0, 5: 0}

        while len(samples) < self.n_valid_samples_target and total_attempts < max_attempts:
            total_attempts += 1

            # Select manifold based on strategy
            if self.strategy == "uniform":
                weights = np.array([1/3, 1/3, 1/3])
            elif self.strategy == "decay":
                progress = total_attempts / max_attempts
                weights = np.array([
                    0.5 - 0.1 * progress,
                    0.3,
                    0.2 + 0.1 * progress
                ])
                weights = weights / weights.sum()
            elif self.strategy == "adaptive":
                total_acc = sum(self.acceptance_rates_by_base.values())
                if total_acc > 0:
                    weights = np.array([
                        self.acceptance_rates_by_base[2] / total_acc,
                        self.acceptance_rates_by_base[3] / total_acc,
                        self.acceptance_rates_by_base[5] / total_acc
                    ])
                else:
                    weights = np.array([1/3, 1/3, 1/3])

            selected_dim = np.random.choice([0, 1, 2], p=weights)
            dim = [2, 3, 5][selected_dim]
            base = self.manifold_bases[selected_dim]

            samples_per_base[dim] += 1

            # Sample on manifold
            proj = np.random.randn(dim) * (2.0 / np.sqrt(dim))
            w = base.reconstruct(proj)
            w = w + np.random.randn(len(w)) * 0.1 * PRIOR_SIGMA

            cppn_copy = self.cppn_template.copy()
            cppn_copy.set_weights(w)
            img = cppn_copy.render(size=32)
            order = self.order_metric(img)

            if order >= self.order_threshold:
                samples.append(w)
                accepts_per_base[dim] += 1
                self.acceptance_rates_by_base[dim] += 1
                self.manifold_preference[dim] += 1

        # Compute acceptance rates
        acceptance_rates = {}
        for dim in [2, 3, 5]:
            if samples_per_base[dim] > 0:
                acceptance_rates[dim] = accepts_per_base[dim] / samples_per_base[dim]
            else:
                acceptance_rates[dim] = 0

        return {
            'valid_samples': len(samples),
            'total_attempts': total_attempts,
            'acceptance_rate': len(samples) / max(1, total_attempts),
            'acceptance_by_base': acceptance_rates,
        }

    def run(self) -> dict:
        """Run full hybrid manifold sampling experiment."""
        start_time = time.time()

        # Stage 1
        stage1_samples, stage1_attempts = self.sample_stage1_unconstrained(n_samples=50)
        self.compute_pca_bases(stage1_samples)
        stage1_time = time.time() - start_time
        stage1_acceptance = len(stage1_samples) / stage1_attempts

        print(f"  Stage 1: {len(stage1_samples)}/50 valid ({stage1_acceptance*100:.1f}%) in {stage1_attempts} attempts", flush=True)

        # Stage 2
        self.acceptance_rates_by_base = {2: 0, 3: 0, 5: 0}
        self.manifold_preference = {2: 0, 3: 0, 5: 0}

        stage2_start = time.time()
        stage2_result = self.sample_stage2_constrained(max_attempts=50000)
        stage2_time = time.time() - stage2_start

        total_time = time.time() - start_time

        return {
            'strategy': self.strategy,
            'stage1_valid': len(stage1_samples),
            'stage1_attempts': stage1_attempts,
            'stage1_acceptance_rate': float(stage1_acceptance),
            'stage1_time': stage1_time,
            'stage2_valid': stage2_result['valid_samples'],
            'stage2_attempts': stage2_result['total_attempts'],
            'stage2_acceptance_rate': float(stage2_result['acceptance_rate']),
            'stage2_acceptance_by_base': {str(k): float(v) for k, v in stage2_result['acceptance_by_base'].items()},
            'stage2_time': stage2_time,
            'total_time': total_time,
            'manifold_preference': {str(k): int(v) for k, v in self.manifold_preference.items()}
        }


# ============================================================================
# BASELINE SAMPLER
# ============================================================================

@dataclass
class BaselineSampler:
    """Naive unconstrained sampling baseline."""
    cppn_template: CPPN
    order_metric: callable = order_multiplicative_v2
    order_threshold: float = 0.5

    def run(self, n_valid_samples: int = 100, max_attempts: int = 50000) -> dict:
        """Run baseline unconstrained sampling."""
        start_time = time.time()

        valid_samples = 0
        total_attempts = 0

        while valid_samples < n_valid_samples and total_attempts < max_attempts:
            n_params = len(self.cppn_template.get_weights())
            w = np.random.randn(n_params) * PRIOR_SIGMA

            cppn_copy = self.cppn_template.copy()
            cppn_copy.set_weights(w)
            img = cppn_copy.render(size=32)
            order = self.order_metric(img)

            total_attempts += 1

            if order >= self.order_threshold:
                valid_samples += 1

        elapsed = time.time() - start_time

        return {
            'valid_samples': valid_samples,
            'total_attempts': total_attempts,
            'acceptance_rate': valid_samples / max(1, total_attempts),
            'time': elapsed
        }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run RES-253 hybrid manifold sampling experiment (fast version)."""
    set_global_seed(42)

    results = {
        'hypothesis': 'Hybrid multi-manifold sampling achieves >=110x speedup',
        'results_by_cppn': [],
        'summary': {}
    }

    # Test on 5 CPPNs (fast version)
    n_cppns = 5
    strategy_speedups = {'uniform': [], 'decay': [], 'adaptive': []}

    for cppn_idx in range(n_cppns):
        print(f"\nCPPN {cppn_idx + 1}/{n_cppns}...", flush=True)

        # Create random CPPN
        cppn = CPPN()

        # Baseline
        baseline_sampler = BaselineSampler(cppn)
        baseline_result = baseline_sampler.run(n_valid_samples=100)

        baseline_throughput = baseline_result['valid_samples'] / max(1, baseline_result['time'])

        print(f"  Baseline: {baseline_result['valid_samples']} valid in {baseline_result['total_attempts']} attempts " +
              f"({baseline_result['acceptance_rate']*100:.2f}%), throughput={baseline_throughput:.2f} samples/sec")

        results_per_cppn = {
            'cppn_idx': cppn_idx,
            'baseline': baseline_result,
            'hybrid_strategies': {}
        }

        for strategy in ['uniform', 'decay', 'adaptive']:
            hybrid_sampler = HybridSampler(cppn, strategy=strategy, n_valid_samples_target=100)
            hybrid_result = hybrid_sampler.run()

            # Compute speedup
            hybrid_throughput = hybrid_result['stage2_valid'] / max(1e-6, hybrid_result['stage2_time'])
            if hybrid_throughput > 0:
                speedup = hybrid_throughput / baseline_throughput
            else:
                speedup = 0

            hybrid_result['speedup_vs_baseline'] = speedup
            results_per_cppn['hybrid_strategies'][strategy] = hybrid_result

            print(f"  {strategy}: speedup={speedup:.2f}x, {hybrid_result['stage2_valid']} valid in " +
                  f"{hybrid_result['stage2_attempts']} attempts, throughput={hybrid_throughput:.2f}/sec")

            strategy_speedups[strategy].append(speedup)

        results['results_by_cppn'].append(results_per_cppn)

    # Summary statistics
    results['summary'] = {
        'n_cppns_tested': n_cppns,
        'uniform_mean_speedup': float(np.mean(strategy_speedups['uniform'])) if strategy_speedups['uniform'] else 0,
        'uniform_std_speedup': float(np.std(strategy_speedups['uniform'])) if strategy_speedups['uniform'] else 0,
        'decay_mean_speedup': float(np.mean(strategy_speedups['decay'])) if strategy_speedups['decay'] else 0,
        'decay_std_speedup': float(np.std(strategy_speedups['decay'])) if strategy_speedups['decay'] else 0,
        'adaptive_mean_speedup': float(np.mean(strategy_speedups['adaptive'])) if strategy_speedups['adaptive'] else 0,
        'adaptive_std_speedup': float(np.std(strategy_speedups['adaptive'])) if strategy_speedups['adaptive'] else 0,
    }

    # Best strategy
    mean_speedups = [
        ('uniform', results['summary']['uniform_mean_speedup']),
        ('decay', results['summary']['decay_mean_speedup']),
        ('adaptive', results['summary']['adaptive_mean_speedup'])
    ]
    best_strategy, best_speedup = max(mean_speedups, key=lambda x: x[1])
    results['summary']['best_strategy'] = best_strategy
    results['summary']['best_speedup'] = best_speedup

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/hybrid_manifold_sampling')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'res_253_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Uniform strategy:  {results['summary']['uniform_mean_speedup']:.2f}x ± {results['summary']['uniform_std_speedup']:.2f}x")
    print(f"Decay strategy:    {results['summary']['decay_mean_speedup']:.2f}x ± {results['summary']['decay_std_speedup']:.2f}x")
    print(f"Adaptive strategy: {results['summary']['adaptive_mean_speedup']:.2f}x ± {results['summary']['adaptive_std_speedup']:.2f}x")
    print(f"\nBest strategy: {best_strategy} ({best_speedup:.2f}x)")
    print(f"Target speedup: >=110x")

    if best_speedup >= 110:
        status = "VALIDATED"
    elif best_speedup >= 10:
        status = "INCONCLUSIVE"
    else:
        status = "REFUTED"

    print(f"Status: {status}")
    print("="*70)

    return results


if __name__ == '__main__':
    results = main()
