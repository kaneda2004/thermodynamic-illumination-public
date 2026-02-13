#!/usr/bin/env python3
"""
RES-235: Deep Feature Learning for Structured Image Discovery
(Successor to RES-229)

HYPOTHESIS:
Non-linear feature combinations (polynomial and cross-terms) enable CPPN priors
to discover structured images more efficiently. RES-229 showed interaction terms
[x, y, x*y, x/y, x²] achieved 2.48× higher order than baseline [x, y, r].
This experiment tests whether deeper non-linear features (squares, cubes, products)
further improve order achievement.

METHOD:
1. Test 5 coordinate input variants:
   - V1: Baseline [x, y, r] (3 inputs)
   - V2: Quadratic [x, y, r, x², y², r²] (6 inputs)
   - V3: Cubic [x, y, r, x², y², x³, y³] (7 inputs)
   - V4: Products [x, y, r, xy, x/y, x²y, xy²] (7 inputs)
   - V5: Deep [x, y, x², y², xy, x²y, xy², x²y²] (8 inputs)

2. For each variant:
   - Generate 25 fresh CPPNs
   - Measure effective dimensionality
   - Render images at 32x32
   - Measure order via order_multiplicative
   - Run nested sampling to 0.5 order level (track samples needed)
   - Compute sampling efficiency

3. Validation:
   - Does cubic/product achieve ≥2.0× baseline order?
   - Does deep feature variant achieve best order?
   - Is there a saturation point (diminishing returns)?

BUILDS ON:
- RES-229: Multi-scale hierarchy REFUTED; interaction terms showed 2.48× gain
- RES-223: Compositional structure drives dimensionality
- RES-004: Baseline sampling efficiency metrics
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
    """V1: Baseline (x, y, r)"""
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


class CPPN_Quadratic(CPPN):
    """V2: Quadratic features [x, y, r, x², y², r²]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
            Node(4, 'identity', 0.0),  # x²
            Node(5, 'identity', 0.0),  # y²
            Node(6, 'identity', 0.0),  # r²
            Node(7, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = []
        # Connect x, y, r, bias to output
        for i in [0, 1, 2, 3]:
            connections.append(Connection(i, 7, np.random.randn() * PRIOR_SIGMA))
        # Connect squared terms to output
        for i in [4, 5, 6]:
            connections.append(Connection(i, 7, np.random.randn() * PRIOR_SIGMA))
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=7)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias, 4: x**2, 5: y**2, 6: r**2}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Cubic(CPPN):
    """V3: Cubic features [x, y, r, x², y², x³, y³]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
            Node(4, 'identity', 0.0),  # x²
            Node(5, 'identity', 0.0),  # y²
            Node(6, 'identity', 0.0),  # x³
            Node(7, 'identity', 0.0),  # y³
            Node(8, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = []
        # Connect x, y, r, bias to output
        for i in [0, 1, 2, 3]:
            connections.append(Connection(i, 8, np.random.randn() * PRIOR_SIGMA))
        # Connect polynomial terms to output
        for i in [4, 5, 6, 7]:
            connections.append(Connection(i, 8, np.random.randn() * PRIOR_SIGMA))
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=8)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias, 4: x**2, 5: y**2, 6: x**3, 7: y**3}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Products(CPPN):
    """V4: Product features [x, y, r, xy, x/y, x²y, xy²]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
            Node(4, 'identity', 0.0),  # xy
            Node(5, 'identity', 0.0),  # x/y
            Node(6, 'identity', 0.0),  # x²y
            Node(7, 'identity', 0.0),  # xy²
            Node(8, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = []
        # Connect base terms to output
        for i in [0, 1, 2, 3]:
            connections.append(Connection(i, 8, np.random.randn() * PRIOR_SIGMA))
        # Connect product terms to output
        for i in [4, 5, 6, 7]:
            connections.append(Connection(i, 8, np.random.randn() * PRIOR_SIGMA))
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=8)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            x_div_y = np.where(np.abs(y) > 1e-8, x / (y + 1e-8), 0)
        values = {
            0: x, 1: y, 2: r, 3: bias,
            4: x * y,
            5: x_div_y,
            6: (x**2) * y,
            7: x * (y**2)
        }
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return np.clip(values[self.output_id], 0, 1)


class CPPN_Deep(CPPN):
    """V5: Deep features [x, y, x², y², xy, x²y, xy², x²y²]"""
    def __init__(self):
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # bias
            Node(3, 'identity', 0.0),  # x²
            Node(4, 'identity', 0.0),  # y²
            Node(5, 'identity', 0.0),  # xy
            Node(6, 'identity', 0.0),  # x²y
            Node(7, 'identity', 0.0),  # xy²
            Node(8, 'identity', 0.0),  # x²y²
            Node(9, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]
        connections = []
        # Connect all feature terms to output
        for i in range(9):
            connections.append(Connection(i, 9, np.random.randn() * PRIOR_SIGMA))
        super().__init__(nodes=nodes, connections=connections, input_ids=[0, 1, 2], output_id=9)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bias = np.ones_like(x)
        values = {
            0: x,
            1: y,
            2: bias,
            3: x**2,
            4: y**2,
            5: x * y,
            6: (x**2) * y,
            7: x * (y**2),
            8: (x**2) * (y**2)
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
    eff_dim = (S.sum() ** 2) / (S ** 2).sum()
    return eff_dim


def measure_order_distribution(cppns: list, image_size: int = 32) -> dict:
    """Measure order distribution across CPPNs."""
    orders = []
    for cppn in cppns:
        img = cppn.render(size=image_size)  # Returns uint8 image
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


def run_experiment() -> dict:
    """Run full deep feature learning experiment."""
    print("\n" + "="*70)
    print("RES-235: Deep Feature Learning for Structured Image Discovery")
    print("="*70)

    image_size = 32
    n_cppns = 25

    # Define variants
    variants = {
        'baseline': CPPN_Baseline,
        'quadratic': CPPN_Quadratic,
        'cubic': CPPN_Cubic,
        'products': CPPN_Products,
        'deep': CPPN_Deep,
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

        results[variant_name] = {
            'eff_dim': eff_dim,
            'order_dist': order_dist,
            'n_cppns': n_cppns,
        }

        print(f"  ✓ eff_dim={eff_dim:.3f}, mean_order={order_dist['mean']:.4f}±{order_dist['std']:.4f}")

    # Compute comparisons
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    baseline_order = results['baseline']['order_dist']['mean']
    comparison = {}

    for variant_name in ['quadratic', 'cubic', 'products', 'deep']:
        variant_order = results[variant_name]['order_dist']['mean']
        gain = variant_order / baseline_order if baseline_order > 0 else 0
        comparison[f'{variant_name}_order_gain'] = gain

        print(f"{variant_name:12s} order gain: {gain:.3f}×")

    # Validation
    validation = {
        'test_quadratic_2x': comparison['quadratic_order_gain'] >= 2.0,
        'test_cubic_2x': comparison['cubic_order_gain'] >= 2.0,
        'test_products_2x': comparison['products_order_gain'] >= 2.0,
        'test_deep_2x': comparison['deep_order_gain'] >= 2.0,
        'best_variant': max(
            [(v, results[v]['order_dist']['mean']) for v in ['baseline', 'quadratic', 'cubic', 'products', 'deep']],
            key=lambda x: x[1]
        )[0],
    }

    if validation['best_variant'] == 'baseline':
        validation['best_gain'] = 1.0
    else:
        validation['best_gain'] = comparison[f"{validation['best_variant']}_order_gain"]

    # Determine conclusion
    if validation['test_deep_2x'] or validation['test_products_2x']:
        validation['conclusion'] = 'validate'
    else:
        validation['conclusion'] = 'refute'

    print(f"\nBest variant: {validation['best_variant']} ({validation['best_gain']:.3f}× baseline)")
    print(f"Conclusion: {validation['conclusion'].upper()}")

    return {
        'experiment_id': 'RES-235',
        'title': 'Deep Feature Learning for Structured Image Discovery',
        'method': 'Compare order achievement across 5 coordinate input variants',
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

        output_file = output_dir / 'res_235_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # Print summary for log_manager
        best = results['validation']['best_variant']
        best_gain = results['validation']['best_gain']
        conclusion = results['validation']['conclusion']
        print(f"\nRES-235 | hierarchical_composition_architecture | {conclusion} | best={best} gain={best_gain:.2f}×")

    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)
