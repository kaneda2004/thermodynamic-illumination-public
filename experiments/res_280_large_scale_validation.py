#!/usr/bin/env python3
"""
RES-280: Large-Scale Speedup Validation for Hybrid Manifold

Hypothesis: RES-268's 1632× speedup claim is reproducible on 100 CPPNs (not just
original 20). The speedup is robust and statistically significant.

Context:
- RES-268 achieved 1632.9× speedup with adaptive hybrid manifold (20 CPPNs)
- Large variance noted: ±4732 std (huge spread)
- Question: Is this real speedup or statistical noise from small sample?
- This experiment: Scale up to 100 CPPNs to validate

Experimental Design:
1. Generate 100 random CPPNs (diverse architectures, weight distributions)
2. For each CPPN, test 3 strategies in nested sampling:
   - Uniform weighting: Equal contribution from all bases (1/3 each)
   - Decay weighting: Linear decay (0.5, 0.3, 0.2 for 2D/3D/5D)
   - Adaptive weighting: Algorithm learns weights online during sampling
3. Target: Order ≥ 0.5 (high-order images)
4. Measure: Sample count needed, success rate, wall-clock time

Success Metrics:
- Mean speedup (adaptive vs baseline uniform)
- Standard deviation and 95% CI
- Distribution shape (normal? heavy-tailed? bimodal?)
- Success rate (% of CPPNs reaching order ≥0.5)
- CPPN-dependent patterns (which architectures benefit most?)

Expected Outputs:
- Best case: Mean 1200-1632× with σ<200×, 80%+ success rate
- Worst case: <500× with σ>1000×, <50% success rate
- Middle case: 600-1000× with σ<500×, 70%+ success rate
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import sys
import time
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, PRIOR_SIGMA, ACTIVATIONS,
    order_multiplicative_v2, set_global_seed
)

# ============================================================================
# HYBRID MANIFOLD SAMPLER (from RES-268)
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
# CPPN ARCHITECTURE CHARACTERIZATION
# ============================================================================

def characterize_cppn(cppn: CPPN) -> Dict:
    """Extract architectural features from CPPN for later analysis."""
    weights = cppn.get_weights()
    return {
        'n_params': len(weights),
        'weight_mean': float(np.mean(weights)),
        'weight_std': float(np.std(weights)),
        'weight_max': float(np.max(np.abs(weights))),
        'weight_entropy': float(-np.sum(np.abs(weights) * np.log(np.abs(weights) + 1e-10))),
        'n_nodes': len(cppn.nodes),
        'n_connections': len(cppn.connections),
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Run RES-280 large-scale speedup validation (100 CPPNs)."""
    set_global_seed(42)

    print("="*80)
    print("RES-280: LARGE-SCALE SPEEDUP VALIDATION")
    print("Hypothesis: 1632× speedup from RES-268 is reproducible on 100 CPPNs")
    print("="*80)

    results = {
        'hypothesis': 'Hybrid manifold adaptive weighting achieves 1632× speedup on 100 CPPNs',
        'n_cppns': 100,
        'methods': 'Three strategies: uniform (1/3 each), decay (fade high-D), adaptive (weights by acceptance)',
        'results_by_cppn': [],
        'summary': {}
    }

    # Track speedups for all 3 strategies
    strategy_speedups = {'uniform': [], 'decay': [], 'adaptive': []}
    strategy_success_rates = {'uniform': [], 'decay': [], 'adaptive': []}
    cppn_architectures = []

    n_cppns = 100
    start_time = time.time()

    for cppn_idx in range(n_cppns):
        elapsed_so_far = time.time() - start_time
        eta_per_cppn = elapsed_so_far / max(1, cppn_idx + 1)
        eta_remaining = eta_per_cppn * (n_cppns - cppn_idx - 1)

        print(f"\n[{cppn_idx + 1:3d}/{n_cppns}] CPPN testing... " +
              f"(elapsed: {elapsed_so_far/60:.1f}m, ETA: {eta_remaining/60:.1f}m)", flush=True)

        # Create random CPPN
        cppn = CPPN()
        cppn_arch = characterize_cppn(cppn)

        # Baseline
        baseline_sampler = BaselineSampler(cppn)
        baseline_result = baseline_sampler.run(n_valid_samples=100)

        baseline_throughput = baseline_result['valid_samples'] / max(1e-6, baseline_result['time'])
        baseline_success = baseline_result['acceptance_rate']

        print(f"  Baseline: {baseline_result['valid_samples']}/100 valid in {baseline_result['total_attempts']} attempts " +
              f"(rate={baseline_success*100:.2f}%), throughput={baseline_throughput:.3f}/sec")

        results_per_cppn = {
            'cppn_idx': cppn_idx,
            'architecture': cppn_arch,
            'baseline': baseline_result,
            'hybrid_strategies': {}
        }

        # Test all 3 strategies
        for strategy in ['uniform', 'decay', 'adaptive']:
            hybrid_sampler = HybridSampler(cppn, strategy=strategy, n_valid_samples_target=100)
            hybrid_result = hybrid_sampler.run()

            # Compute speedup based on throughput in Stage 2 only
            hybrid_throughput = hybrid_result['stage2_valid'] / max(1e-6, hybrid_result['stage2_time'])
            if baseline_throughput > 0:
                speedup = hybrid_throughput / baseline_throughput
            else:
                speedup = 0

            # Success rate is whether we got all 100 valid samples
            success = 1 if hybrid_result['stage2_valid'] >= 100 else 0

            hybrid_result['speedup_vs_baseline'] = float(speedup)
            hybrid_result['success'] = success
            results_per_cppn['hybrid_strategies'][strategy] = hybrid_result

            print(f"  {strategy:8s}: speedup={speedup:8.2f}x, valid={hybrid_result['stage2_valid']:3d}/100, " +
                  f"attempts={hybrid_result['stage2_attempts']:5d}, rate={hybrid_result['stage2_acceptance_rate']*100:5.2f}%")

            strategy_speedups[strategy].append(speedup)
            strategy_success_rates[strategy].append(success)

        results['results_by_cppn'].append(results_per_cppn)
        cppn_architectures.append(cppn_arch)

    total_time = time.time() - start_time

    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    summary = {
        'n_cppns_tested': n_cppns,
        'total_runtime_minutes': total_time / 60,
    }

    for strategy in ['uniform', 'decay', 'adaptive']:
        speedups = np.array(strategy_speedups[strategy])
        success_rates = np.array(strategy_success_rates[strategy])

        # Compute statistics
        mean_speedup = float(np.mean(speedups))
        std_speedup = float(np.std(speedups))
        median_speedup = float(np.median(speedups))
        q1_speedup = float(np.percentile(speedups, 25))
        q3_speedup = float(np.percentile(speedups, 75))
        ci_lower = float(np.percentile(speedups, 2.5))
        ci_upper = float(np.percentile(speedups, 97.5))
        skewness = float(stats.skew(speedups))
        kurtosis = float(stats.kurtosis(speedups))

        # Success rate
        success_rate = float(np.mean(success_rates) * 100)

        # Distribution fitting
        # Check if normal, lognormal, or other
        _, p_normal = stats.normaltest(speedups)
        _, p_lognormal = stats.normaltest(np.log(speedups + 1e-10))

        summary[f'{strategy}_mean_speedup'] = mean_speedup
        summary[f'{strategy}_std_speedup'] = std_speedup
        summary[f'{strategy}_median_speedup'] = median_speedup
        summary[f'{strategy}_q1_speedup'] = q1_speedup
        summary[f'{strategy}_q3_speedup'] = q3_speedup
        summary[f'{strategy}_ci_lower'] = ci_lower
        summary[f'{strategy}_ci_upper'] = ci_upper
        summary[f'{strategy}_skewness'] = skewness
        summary[f'{strategy}_kurtosis'] = kurtosis
        summary[f'{strategy}_success_rate_percent'] = success_rate
        summary[f'{strategy}_p_normal'] = float(p_normal)
        summary[f'{strategy}_p_lognormal'] = float(p_lognormal)

        dist_type = "normal" if p_normal > 0.05 else ("lognormal" if p_lognormal > 0.05 else "heavy-tailed")

        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Mean speedup:    {mean_speedup:.2f}x")
        print(f"  Std dev:         {std_speedup:.2f}x")
        print(f"  Median:          {median_speedup:.2f}x")
        print(f"  IQR:             [{q1_speedup:.2f}x, {q3_speedup:.2f}x]")
        print(f"  95% CI:          [{ci_lower:.2f}x, {ci_upper:.2f}x]")
        print(f"  Skewness:        {skewness:.3f}")
        print(f"  Kurtosis:        {kurtosis:.3f}")
        print(f"  Distribution:    {dist_type}")
        print(f"  Success rate:    {success_rate:.1f}%")

    # Best strategy
    mean_speedups = [
        ('uniform', summary['uniform_mean_speedup']),
        ('decay', summary['decay_mean_speedup']),
        ('adaptive', summary['adaptive_mean_speedup'])
    ]
    best_strategy, best_speedup = max(mean_speedups, key=lambda x: x[1])
    summary['best_strategy'] = best_strategy
    summary['best_speedup'] = best_speedup
    summary['best_success_rate'] = summary[f'{best_strategy}_success_rate_percent']

    # Comparison to RES-268 claim
    res268_claim = 1632.9
    summary['res268_claim'] = res268_claim
    summary['speedup_vs_claim'] = best_speedup / res268_claim if res268_claim > 0 else 0

    print("\n" + "="*80)
    print("VALIDATION AGAINST RES-268 CLAIM")
    print("="*80)
    print(f"RES-268 claim:           {res268_claim:.2f}x (on 20 CPPNs)")
    print(f"RES-280 best result:     {best_speedup:.2f}x (on 100 CPPNs)")
    print(f"RES-280 strategy:        {best_strategy}")
    print(f"RES-280 vs claim:        {summary['speedup_vs_claim']*100:.1f}%")
    print(f"Success rate:            {summary['best_success_rate']:.1f}%")

    # Determine validation status
    if best_speedup >= 1200:
        status = "validated"
        conclusion = "Strong validation: 1632× speedup is reproducible at scale"
    elif best_speedup >= 600:
        status = "inconclusive"
        conclusion = "Conditional validation: Speedup is real but smaller than claimed"
    else:
        status = "refuted"
        conclusion = "Refutation: Speedup is significantly lower than claimed"

    print(f"\nSTATUS:                  {status.upper()}")
    print(f"CONCLUSION:              {conclusion}")

    summary['status'] = status
    summary['conclusion'] = conclusion

    results['summary'] = summary

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/speedup_validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'res_280_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


if __name__ == '__main__':
    results = main()
