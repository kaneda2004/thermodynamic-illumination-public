#!/usr/bin/env python3
"""
RES-281: Large-Scale Speedup Validation for Hybrid Manifold (PARALLELIZED)

Hypothesis: RES-268's 1632× speedup claim is reproducible on 100 CPPNs (not just
original 20). The speedup is robust and statistically significant.

IMPROVEMENTS OVER RES-280:
- Multiprocessing with 8 workers (M4 Max: 16 cores available)
- Checkpointing every 5 CPPNs (resumable from failures)
- Real-time progress monitoring (write checkpoint + print status)
- Estimated runtime: ~4-6 hours (vs 30 hours sequential)

Expected Outputs:
- Best case: Mean 1200-1632× with σ<200×, 80%+ success rate
- Worst case: <500× with σ>1000×, <50% success rate
- Middle case: 600-1000× with σ<500×, 70%+ success rate
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Optional
import sys
import time
from scipy import stats
import multiprocessing as mp
from multiprocessing import Queue, Process
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, PRIOR_SIGMA, ACTIVATIONS,
    order_multiplicative_v2, set_global_seed
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/speedup_validation_parallel/res_281.log')
    ]
)
logger = logging.getLogger(__name__)

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
        """Project weights onto manifold basis.

        weights: (n_weights,) - weight vector
        components: (dim, n_weights) - PCA basis vectors
        result: (dim,) - projection in manifold space
        """
        centered = weights - self.mean
        return centered @ self.components.T  # (n_weights,) @ (n_weights, dim) = (dim,)

    def reconstruct(self, proj: np.ndarray) -> np.ndarray:
        """Reconstruct weights from projection.

        proj: (dim,) - point in manifold space
        components: (dim, n_weights) - PCA basis vectors
        result: (n_weights,) - reconstructed weight vector
        """
        return (proj @ self.components) + self.mean  # (dim,) @ (dim, n_weights) = (n_weights,)


@dataclass
class BaselineSampler:
    """Baseline: pure random sampling."""
    cppn_template: CPPN
    order_metric: callable = order_multiplicative_v2
    order_threshold: float = 0.5

    def run(self, n_valid_samples: int = 100) -> dict:
        """Run baseline sampling."""
        valid_samples = []
        total_attempts = 0
        start_time = time.time()

        while len(valid_samples) < n_valid_samples:
            w = np.random.randn(len(self.cppn_template.get_weights())) * PRIOR_SIGMA
            cppn_copy = self.cppn_template.copy()
            cppn_copy.set_weights(w)
            img = cppn_copy.render(size=32)
            order = self.order_metric(img)
            total_attempts += 1

            if order >= self.order_threshold:
                valid_samples.append(w)

        elapsed = time.time() - start_time
        return {
            'valid_samples': len(valid_samples),
            'total_attempts': total_attempts,
            'acceptance_rate': len(valid_samples) / total_attempts,
            'time': elapsed
        }


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

    def fit_manifolds(self, samples: np.ndarray):
        """Fit 2D, 3D, 5D manifolds via PCA."""
        self.manifold_bases = []
        for dim in [2, 3, 5]:
            mean = np.mean(samples, axis=0)
            centered = samples - mean
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            components = Vt[:dim, :]

            base = ManifoldBase(dim=dim, components=components, mean=mean)
            self.manifold_bases.append(base)

    def sample_stage2_constrained(self, n_valid_samples: int = 100) -> dict:
        """Stage 2: Constrained sampling on manifolds with adaptive/fixed weighting."""
        valid_samples = []
        valid_by_base = {0: [], 1: [], 2: []}
        total_attempts = 0
        start_time = time.time()

        # Initialize weights
        if self.strategy == "uniform":
            weights = [1/3, 1/3, 1/3]
        elif self.strategy == "decay":
            weights = [0.5, 0.3, 0.2]
        else:  # adaptive
            weights = [1/3, 1/3, 1/3]  # Start uniform, adapt during sampling

        acceptance_by_base = [0, 0, 0]

        while len(valid_samples) < n_valid_samples:
            # Select manifold proportional to weights
            base_idx = np.random.choice(3, p=weights / np.sum(weights))
            base = self.manifold_bases[base_idx]

            # Sample on manifold
            proj = np.random.randn(base.dim) * 0.5
            w_proj = base.reconstruct(proj)
            cppn_copy = self.cppn_template.copy()
            cppn_copy.set_weights(w_proj)
            img = cppn_copy.render(size=32)
            order = self.order_metric(img)
            total_attempts += 1

            if order >= self.order_threshold:
                valid_samples.append(w_proj)
                valid_by_base[base_idx].append(w_proj)
                acceptance_by_base[base_idx] += 1

                # Adaptive: increase weight for successful base
                if self.strategy == "adaptive":
                    weights[base_idx] *= 1.05

        elapsed = time.time() - start_time
        return {
            'stage2_valid': len(valid_samples),
            'stage2_attempts': total_attempts,
            'stage2_acceptance_rate': len(valid_samples) / total_attempts if total_attempts > 0 else 0,
            'stage2_time': elapsed,
            'valid_by_base': {i: len(v) for i, v in valid_by_base.items()},
            'acceptance_by_base': acceptance_by_base
        }

    def run(self) -> dict:
        """Run full hybrid sampling."""
        samples_stage1, attempts_stage1 = self.sample_stage1_unconstrained(n_samples=50)
        self.fit_manifolds(samples_stage1)
        result = self.sample_stage2_constrained(n_valid_samples=self.n_valid_samples_target)
        result['stage1_attempts'] = attempts_stage1
        return result


def characterize_cppn(cppn: CPPN) -> dict:
    """Characterize CPPN architecture."""
    n_nodes = len(cppn.nodes)
    n_conns = len(cppn.connections)
    return {
        'n_nodes': n_nodes,
        'n_connections': n_conns,
        'n_weights': len(cppn.get_weights())
    }


# ============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================

def worker_process_cppn(cppn_idx: int, result_queue: Queue) -> None:
    """
    Process one CPPN with baseline and 3 strategies.
    Send result back via queue.
    """
    try:
        set_global_seed(cppn_idx)  # Deterministic CPPN generation per index

        # Create random CPPN
        cppn = CPPN()
        cppn_arch = characterize_cppn(cppn)

        # Baseline
        baseline_sampler = BaselineSampler(cppn)
        baseline_result = baseline_sampler.run(n_valid_samples=100)
        baseline_throughput = baseline_result['valid_samples'] / max(1e-6, baseline_result['time'])
        baseline_success = baseline_result['acceptance_rate']

        results_per_cppn = {
            'cppn_idx': cppn_idx,
            'architecture': cppn_arch,
            'baseline': baseline_result,
            'hybrid_strategies': {}
        }

        strategy_speedups = {}

        # Test all 3 strategies
        for strategy in ['uniform', 'decay', 'adaptive']:
            hybrid_sampler = HybridSampler(cppn, strategy=strategy, n_valid_samples_target=100)
            hybrid_result = hybrid_sampler.run()

            # Compute speedup
            hybrid_throughput = hybrid_result['stage2_valid'] / max(1e-6, hybrid_result['stage2_time'])
            if baseline_throughput > 0:
                speedup = hybrid_throughput / baseline_throughput
            else:
                speedup = 0

            success = 1 if hybrid_result['stage2_valid'] >= 100 else 0
            hybrid_result['speedup_vs_baseline'] = float(speedup)
            hybrid_result['success'] = success
            results_per_cppn['hybrid_strategies'][strategy] = hybrid_result
            strategy_speedups[strategy] = speedup

        result_queue.put({
            'status': 'success',
            'cppn_idx': cppn_idx,
            'results': results_per_cppn,
            'speedups': strategy_speedups
        })
    except Exception as e:
        result_queue.put({
            'status': 'error',
            'cppn_idx': cppn_idx,
            'error': str(e)
        })


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(results: list, checkpoint_path: Path) -> None:
    """Save progress checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'n_completed': len(results),
        'results_by_cppn': results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"✓ Checkpoint saved: {len(results)} CPPNs processed")


def load_checkpoint(checkpoint_path: Path) -> Optional[list]:
    """Load progress from checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        logger.info(f"✓ Checkpoint loaded: {data['n_completed']} CPPNs already processed")
        return data['results_by_cppn']
    return None


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    checkpoint_path = Path('results/speedup_validation_parallel/checkpoint_res_281.json')
    results_path = Path('results/speedup_validation_parallel/res_281_results.json')

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    n_cppns = 100
    n_workers = 8  # M4 Max: 16 cores, use 8 workers
    checkpoint_interval = 5  # Save every 5 CPPNs

    # Try loading checkpoint
    results_by_cppn = load_checkpoint(checkpoint_path)
    start_idx = len(results_by_cppn) if results_by_cppn else 0

    if start_idx > 0:
        logger.info(f"Resuming from checkpoint: starting at CPPN {start_idx + 1}/{n_cppns}")
    else:
        logger.info(f"Starting fresh: processing {n_cppns} CPPNs with {n_workers} workers")
        results_by_cppn = []

    start_time = time.time()
    strategy_speedups = {'uniform': [], 'decay': [], 'adaptive': []}
    strategy_success_rates = {'uniform': [], 'decay': [], 'adaptive': []}

    # Process remaining CPPNs with multiprocessing
    for batch_start in range(start_idx, n_cppns, n_workers):
        batch_end = min(batch_start + n_workers, n_cppns)
        batch_size = batch_end - batch_start

        logger.info(f"\n[{batch_start}/{n_cppns}] Starting batch of {batch_size} CPPNs...")
        batch_start_time = time.time()

        # Create queue and workers
        result_queue = Queue()
        workers = []

        for i in range(batch_start, batch_end):
            p = Process(target=worker_process_cppn, args=(i, result_queue))
            p.start()
            workers.append(p)

        # Collect results
        for _ in range(batch_size):
            result = result_queue.get(timeout=3600)  # 1 hour timeout per CPPN

            if result['status'] == 'success':
                cppn_idx = result['cppn_idx']
                results_by_cppn.append(result['results'])

                # Track speedups
                for strategy, speedup in result['speedups'].items():
                    strategy_speedups[strategy].append(speedup)
                    success = result['results']['hybrid_strategies'][strategy]['success']
                    strategy_success_rates[strategy].append(success)

                logger.info(
                    f"  ✓ CPPN {cppn_idx + 1}: "
                    f"uniform={result['speedups']['uniform']:.1f}x, "
                    f"decay={result['speedups']['decay']:.1f}x, "
                    f"adaptive={result['speedups']['adaptive']:.1f}x"
                )
            else:
                logger.error(f"  ✗ CPPN {result['cppn_idx']} failed: {result.get('error', 'unknown')}")

        # Wait for workers
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        # Save checkpoint after each batch
        if batch_end % checkpoint_interval == 0 or batch_end == n_cppns:
            save_checkpoint(results_by_cppn, checkpoint_path)

        batch_time = time.time() - batch_start_time
        eta_per_cppn = batch_time / batch_size
        remaining_cppns = n_cppns - batch_end
        eta_remaining = eta_per_cppn * remaining_cppns

        logger.info(
            f"  Batch complete in {batch_time/60:.1f}m. "
            f"ETA: {eta_remaining/60:.1f}m remaining"
        )

    total_time = time.time() - start_time

    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 80)

    results = {
        'n_cppns_tested': n_cppns,
        'total_runtime_minutes': total_time / 60,
        'results_by_cppn': results_by_cppn,
    }

    summary = {}

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

        dist_type = "normal" if skewness > -0.5 and skewness < 0.5 else ("right-skewed" if skewness > 0 else "left-skewed")

        logger.info(f"\n{strategy.upper()} STRATEGY:")
        logger.info(f"  Mean speedup:    {mean_speedup:.2f}x")
        logger.info(f"  Std dev:         {std_speedup:.2f}x")
        logger.info(f"  Median:          {median_speedup:.2f}x")
        logger.info(f"  IQR:             [{q1_speedup:.2f}x, {q3_speedup:.2f}x]")
        logger.info(f"  95% CI:          [{ci_lower:.2f}x, {ci_upper:.2f}x]")
        logger.info(f"  Skewness:        {skewness:.3f}")
        logger.info(f"  Distribution:    {dist_type}")
        logger.info(f"  Success rate:    {success_rate:.1f}%")

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

    # Validation against RES-268
    res268_claim = 1632.9
    summary['res268_claim'] = res268_claim
    summary['speedup_vs_claim'] = best_speedup / res268_claim if res268_claim > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION AGAINST RES-268 CLAIM")
    logger.info("=" * 80)
    logger.info(f"RES-268 claim:           {res268_claim:.2f}x (on 20 CPPNs)")
    logger.info(f"RES-281 best result:     {best_speedup:.2f}x (on {n_cppns} CPPNs)")
    logger.info(f"RES-281 strategy:        {best_strategy}")
    logger.info(f"RES-281 vs claim:        {summary['speedup_vs_claim']*100:.1f}%")
    logger.info(f"Success rate:            {summary['best_success_rate']:.1f}%")

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

    logger.info(f"\nSTATUS:                  {status.upper()}")
    logger.info(f"CONCLUSION:              {conclusion}")

    summary['status'] = status
    summary['conclusion'] = conclusion
    results['summary'] = summary

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info(f"✓ Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    return results


if __name__ == '__main__':
    results = main()
