#!/usr/bin/env python3
"""
RES-254: Hierarchical Multi-Scale Composition with Nested Sampling Validation

HYPOTHESIS:
Multi-scale coordinate inputs [x, x/2, x/4, y, y/2, y/4] capture both coarse and fine
structure, achieving either:
  (A) Higher order (≥1.2× baseline), OR
  (B) Faster sampling efficiency (≥2× baseline samples to reach order ≥0.5)
compared to single-scale baseline [x, y, r].

BUILDS ON:
- RES-229: Multi-scale hierarchy REFUTED; hierarchy order gain 0.69×
- RES-235: Deep feature learning REFUTED; polynomial terms don't improve order
- RES-003: Spatial MI (366× higher than random); local vs global structure distinction
- RES-004: Baseline sampling efficiency metrics

METHOD:
1. Compare 5 CPPN variants:
   - V1: Baseline [x, y, r] (3 direct inputs + bias) — single scale
   - V2: Hierarchy [x, x/2, x/4, y, y/2, y/4] (6 direct inputs + bias) — multi-scale coarse-to-fine
   - V3: Interaction [x, y, x*y, x/y] (4 direct inputs + bias) — RES-229 showed 2.48× gain
   - V4: Adaptive [x, y, x*sqrt(x²+y²), tanh(x*y)] (4 direct inputs + bias) — non-linear composition
   - V5: Spectral [x, y, cos(2πx), sin(2πy)] (4 direct inputs + bias) — frequency basis

2. For each variant (25 CPPNs per variant):
   - Generate fresh CPPN with random weights
   - Render 32×32 image
   - Measure effective dimensionality of weight space
   - Measure initial order via order_multiplicative
   - Run nested sampling to order ≥0.5 (track samples needed)
   - Compute sampling efficiency metric

3. Validation Tests:
   - Test A: Does any variant achieve ≥1.2× baseline order?
   - Test B: Does any variant need ≤50% of baseline samples (2× speedup)?
   - Test C: Do multi-scale variants outperform single-scale?

EXPECTED: Interaction and adaptive variants show gains; pure multi-scale hierarchy
  does not (based on RES-229, RES-235 refutation pattern).
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict
import traceback

# Setup
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, nested_sampling_v3,
    set_global_seed, PRIOR_SIGMA, ACTIVATIONS
)

set_global_seed(42)

# ============================================================================
# CPPN VARIANTS WITH DIFFERENT COORDINATE INPUTS
# ============================================================================

class CPPN_Baseline(CPPN):
    """V1: Baseline single-scale (x, y, r)"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
            Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = [
            Connection(i, 4, np.random.randn() * PRIOR_SIGMA) for i in [0, 1, 2, 3]
        ]
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Hierarchy(CPPN):
    """V2: Multi-scale hierarchy [x, x/2, x/4, y, y/2, y/4]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),   # x
            Node(1, 'identity', 0.0),   # x/2
            Node(2, 'identity', 0.0),   # x/4
            Node(3, 'identity', 0.0),   # y
            Node(4, 'identity', 0.0),   # y/2
            Node(5, 'identity', 0.0),   # y/4
            Node(6, 'identity', 0.0),   # bias
            Node(7, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = [
            Connection(i, 7, np.random.randn() * PRIOR_SIGMA) for i in [0, 1, 2, 3, 4, 5, 6]
        ]
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3, 4, 5, 6], output_id=7)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bias = np.ones_like(x)
        values = {0: x, 1: x/2, 2: x/4, 3: y, 4: y/2, 5: y/4, 6: bias}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Interaction(CPPN):
    """V3: Interaction terms [x, y, x*y, x/y] (RES-229 showed 2.48× gain)"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),   # x
            Node(1, 'identity', 0.0),   # y
            Node(2, 'identity', 0.0),   # x*y
            Node(3, 'identity', 0.0),   # x/y
            Node(4, 'identity', 0.0),   # bias
            Node(5, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = [
            Connection(i, 5, np.random.randn() * PRIOR_SIGMA) for i in [0, 1, 2, 3, 4]
        ]
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3, 4], output_id=5)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bias = np.ones_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_div_y = np.where(np.abs(y) > 1e-8, x / (y + 1e-8), 0)
        values = {0: x, 1: y, 2: x*y, 3: x_div_y, 4: bias}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Adaptive(CPPN):
    """V4: Adaptive non-linear composition [x, y, x*sqrt(x²+y²), tanh(x*y)]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),   # x
            Node(1, 'identity', 0.0),   # y
            Node(2, 'identity', 0.0),   # x*r (adaptive radial modulation)
            Node(3, 'identity', 0.0),   # tanh(xy) (nonlinear interaction)
            Node(4, 'identity', 0.0),   # bias
            Node(5, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = [
            Connection(i, 5, np.random.randn() * PRIOR_SIGMA) for i in [0, 1, 2, 3, 4]
        ]
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3, 4], output_id=5)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bias = np.ones_like(x)
        r = np.sqrt(x**2 + y**2)
        values = {
            0: x,
            1: y,
            2: x * r,  # Radial-modulated x
            3: np.tanh(x * y),  # Nonlinear interaction
            4: bias
        }
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Spectral(CPPN):
    """V5: Spectral/frequency basis [x, y, cos(2πx), sin(2πy)]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),   # x
            Node(1, 'identity', 0.0),   # y
            Node(2, 'identity', 0.0),   # cos(2πx)
            Node(3, 'identity', 0.0),   # sin(2πy)
            Node(4, 'identity', 0.0),   # bias
            Node(5, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = [
            Connection(i, 5, np.random.randn() * PRIOR_SIGMA) for i in [0, 1, 2, 3, 4]
        ]
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3, 4], output_id=5)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bias = np.ones_like(x)
        values = {
            0: x,
            1: y,
            2: np.cos(2 * np.pi * x),
            3: np.sin(2 * np.pi * y),
            4: bias
        }
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def measure_effective_dimensionality(cppns: list) -> float:
    """Measure effective dimensionality from CPPN weight distribution."""
    weights_flat = []
    for cppn in cppns:
        w = np.array([c.weight for c in cppn.connections] + [n.bias for n in cppn.nodes[1:]])
        weights_flat.append(w)

    W = np.array(weights_flat)
    W_centered = W - W.mean(axis=0)
    U, S, _ = np.linalg.svd(W_centered, full_matrices=False)
    S = S[S > 1e-10]

    # Effective dimensionality via participation ratio
    eff_dim = (S.sum() ** 2) / (S ** 2).sum() if len(S) > 0 else 1.0
    return eff_dim


def measure_order_distribution(cppns: list, image_size: int = 32) -> dict:
    """Measure order distribution across CPPNs."""
    orders = []
    for cppn in cppns:
        img = cppn.render(size=image_size)
        order = order_multiplicative(img)
        orders.append(order)

    orders = np.array(orders)
    return {
        'mean': float(orders.mean()),
        'std': float(orders.std()),
        'p25': float(np.percentile(orders, 25)),
        'p50': float(np.percentile(orders, 50)),
        'p75': float(np.percentile(orders, 75)),
        'p90': float(np.percentile(orders, 90)),
        'min': float(orders.min()),
        'max': float(orders.max()),
    }


def measure_sampling_efficiency(cppns: list, target_order: float = 0.5, max_samples: int = 5000) -> dict:
    """Measure sampling efficiency to reach target order."""
    samples_needed = []
    success_count = 0

    for cppn in cppns:
        try:
            # Run nested sampling with short timeout
            result = nested_sampling_v3(
                cppn, max_iterations=max_samples // 10, target_order=target_order,
                verbose=False, seed=np.random.randint(0, 2**31)
            )
            if result['success']:
                samples_needed.append(result.get('samples_used', max_samples))
                success_count += 1
            else:
                samples_needed.append(max_samples)
        except Exception:
            samples_needed.append(max_samples)

    samples_array = np.array(samples_needed)
    success_rate = success_count / len(cppns)

    return {
        'median_samples': float(np.median(samples_array)) if len(samples_array) > 0 else max_samples,
        'mean_samples': float(np.mean(samples_array)) if len(samples_array) > 0 else max_samples,
        'success_rate': float(success_rate),
        'samples_p90': float(np.percentile(samples_array, 90)),
    }


def run_experiment() -> dict:
    """Run full hierarchical multi-scale experiment."""
    print("\n" + "="*70)
    print("RES-254: Hierarchical Multi-Scale Composition with Nested Sampling")
    print("="*70)

    image_size = 32
    n_cppns = 25

    # Define variants
    variants = {
        'baseline': CPPN_Baseline,
        'hierarchy': CPPN_Hierarchy,
        'interaction': CPPN_Interaction,
        'adaptive': CPPN_Adaptive,
        'spectral': CPPN_Spectral,
    }

    results = {}

    for variant_name, CPPN_class in variants.items():
        print(f"\n[{variant_name.upper()}] Generating {n_cppns} CPPNs...")
        set_global_seed(42)
        cppns = [CPPN_class() for _ in range(n_cppns)]

        print(f"  Measuring effective dimensionality...")
        eff_dim = measure_effective_dimensionality(cppns)

        print(f"  Measuring order distribution...")
        order_dist = measure_order_distribution(cppns, image_size)

        print(f"  Measuring sampling efficiency...")
        sampling_eff = measure_sampling_efficiency(cppns, target_order=0.5)

        results[variant_name] = {
            'eff_dim': eff_dim,
            'order_dist': order_dist,
            'sampling_efficiency': sampling_eff,
            'n_cppns': n_cppns,
        }

        print(f"  ✓ eff_dim={eff_dim:.3f}, mean_order={order_dist['mean']:.4f}±{order_dist['std']:.4f}")
        print(f"    sampling: median={sampling_eff['median_samples']:.0f} samples, success={sampling_eff['success_rate']:.2%}")

    # Compute comparisons
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    baseline_order = results['baseline']['order_dist']['mean']
    baseline_samples = results['baseline']['sampling_efficiency']['median_samples']
    comparison = {}

    for variant_name in ['hierarchy', 'interaction', 'adaptive', 'spectral']:
        variant_order = results[variant_name]['order_dist']['mean']
        variant_samples = results[variant_name]['sampling_efficiency']['median_samples']

        order_gain = variant_order / baseline_order if baseline_order > 0 else 0
        sampling_speedup = baseline_samples / variant_samples if variant_samples > 0 else 0

        comparison[f'{variant_name}_order_gain'] = order_gain
        comparison[f'{variant_name}_sampling_speedup'] = sampling_speedup

        print(f"{variant_name:12s}: order={order_gain:.3f}×, sampling_speedup={sampling_speedup:.3f}×")

    # Validation tests
    validation = {
        'test_hierarchy_12x': comparison['hierarchy_order_gain'] >= 1.2,
        'test_hierarchy_efficiency_2x': comparison['hierarchy_sampling_speedup'] >= 2.0,
        'test_interaction_12x': comparison['interaction_order_gain'] >= 1.2,
        'test_interaction_efficiency_2x': comparison['interaction_sampling_speedup'] >= 2.0,
        'test_adaptive_12x': comparison['adaptive_order_gain'] >= 1.2,
        'test_adaptive_efficiency_2x': comparison['adaptive_sampling_speedup'] >= 2.0,
        'test_spectral_12x': comparison['spectral_order_gain'] >= 1.2,
        'test_spectral_efficiency_2x': comparison['spectral_sampling_speedup'] >= 2.0,
    }

    # Find best variant
    best_variant = 'baseline'
    best_order_gain = 1.0
    best_sampling_speedup = 1.0

    for variant_name in ['hierarchy', 'interaction', 'adaptive', 'spectral']:
        order_gain = comparison[f'{variant_name}_order_gain']
        sampling_speedup = comparison[f'{variant_name}_sampling_speedup']
        if order_gain > best_order_gain:
            best_variant = variant_name
            best_order_gain = order_gain
            best_sampling_speedup = sampling_speedup

    validation['best_variant'] = best_variant
    validation['best_order_gain'] = best_order_gain
    validation['best_sampling_speedup'] = best_sampling_speedup

    # Determine conclusion
    # Validate if ANY variant achieves 1.2× order OR 2× sampling speedup
    if (validation['test_hierarchy_12x'] or validation['test_hierarchy_efficiency_2x'] or
        validation['test_interaction_12x'] or validation['test_interaction_efficiency_2x'] or
        validation['test_adaptive_12x'] or validation['test_adaptive_efficiency_2x'] or
        validation['test_spectral_12x'] or validation['test_spectral_efficiency_2x']):
        validation['conclusion'] = 'validate'
    else:
        validation['conclusion'] = 'refute'

    print(f"\nBest variant: {validation['best_variant']}")
    print(f"  Order gain: {validation['best_order_gain']:.3f}×")
    print(f"  Sampling speedup: {validation['best_sampling_speedup']:.3f}×")
    print(f"Conclusion: {validation['conclusion'].upper()}")

    return {
        'experiment_id': 'RES-253',
        'title': 'Hierarchical Multi-Scale Composition with Nested Sampling Validation',
        'hypothesis': 'Multi-scale [x,x/2,x/4,y,y/2,y/4] achieves ≥1.2× order or ≥2× sampling efficiency vs [x,y,r]',
        'method': 'Compare 5 CPPN variants: baseline, hierarchy, interaction, adaptive, spectral',
        'n_samples_per_variant': n_cppns,
        'results': results,
        'comparison': comparison,
        'validation': validation,
    }


if __name__ == '__main__':
    try:
        results = run_experiment()

        # Create output directory
        output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/hierarchical_composition_architecture')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'res_253_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # Print summary for log_manager
        best = results['validation']['best_variant']
        best_gain = results['validation']['best_order_gain']
        sampling_speedup = results['validation']['best_sampling_speedup']
        conclusion = results['validation']['conclusion']
        print(f"\nRES-253 | hierarchical_composition_architecture | {conclusion} | order={best_gain:.2f}× sampling={sampling_speedup:.2f}×")

    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)
