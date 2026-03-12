#!/usr/bin/env python3
"""
RES-229: Hierarchical Multi-Scale Composition
(Planned as RES-234, auto-claimed as RES-229)

HYPOTHESIS:
Multi-scale coordinate composition [x, x/2, x/4, y, y/2, y/4] achieves:
- Order score ≥1.2x baseline, OR
- Sampling efficiency ≥2x baseline

RATIONALE:
- RES-223 showed compositional (x,y,r) inputs drive dimensionality
- RES-003 found multi-scale spatial MI benefits
- Multi-resolution inputs may provide complementary features:
  * Fine scales (x, y) capture detail
  * Medium scales (x/2, y/2) capture mid-level structure
  * Coarse scales (x/4, y/4) capture global shape

METHOD:
1. Implement 3 hierarchical variants:
   - Variant A: Full hierarchy [x, x/2, x/4, y, y/2, y/4] (6 inputs)
   - Variant B: Interaction terms [x, y, x*y, x/y, x²] (different 6 inputs)
   - Variant C: Baseline [x, y, r] (3 inputs)

2. For each variant:
   a) Initialize 25 CPPNs
   b) Measure effective dimensionality (weight space)
   c) Render images and measure order scores
   d) Run nested sampling to order 0.5
   e) Track samples-to-threshold

3. Compare across variants:
   - eff_dim per input type
   - order quality distribution
   - sampling efficiency (samples_per_unit_order)

4. Validate hypothesis:
   - hierarchy_order ≥ 1.2 * baseline_order?, OR
   - hierarchy_efficiency ≥ 2.0 * baseline_efficiency?

BUILDS ON:
- RES-223: Compositional structure drives dimensionality
- RES-003: Multi-scale spatial MI properties
- RES-224: Sampling efficiency properties
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
    """Baseline: Standard (x, y, r) inputs"""
    def __init__(self):
        # Initialize 3 input nodes: x, y, r
        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
            Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]

        connections = []
        # Connect all inputs to output
        for inp_id in [0, 1, 2, 3]:
            connections.append(
                Connection(inp_id, 4, np.random.randn() * PRIOR_SIGMA)
            )

        super().__init__(nodes=nodes, connections=connections,
                        input_ids=[0, 1, 2, 3], output_id=4)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate network with standard (x, y, r, bias) inputs"""
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
        return values[self.output_id]


class CPPN_Hierarchy(CPPN):
    """
    Hierarchical: [x, x/2, x/4, y, y/2, y/4] (6 inputs)

    Multi-scale coordinate composition:
    - Fine scale (x, y): detail
    - Medium scale (x/2, y/2): mid-level structure
    - Coarse scale (x/4, y/4): global shape
    """
    def __init__(self):
        # Initialize 6 coordinate input nodes + 1 bias
        nodes = [
            Node(0, 'identity', 0.0),   # x (fine)
            Node(1, 'identity', 0.0),   # x/2 (medium)
            Node(2, 'identity', 0.0),   # x/4 (coarse)
            Node(3, 'identity', 0.0),   # y (fine)
            Node(4, 'identity', 0.0),   # y/2 (medium)
            Node(5, 'identity', 0.0),   # y/4 (coarse)
            Node(6, 'identity', 0.0),   # bias
            Node(7, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]

        connections = []
        # Connect all coordinate inputs to output
        for inp_id in [0, 1, 2, 3, 4, 5, 6]:
            connections.append(
                Connection(inp_id, 7, np.random.randn() * PRIOR_SIGMA)
            )

        super().__init__(nodes=nodes, connections=connections,
                        input_ids=[0, 1, 2, 3, 4, 5, 6], output_id=7)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate network with multi-scale coordinates"""
        bias = np.ones_like(x)
        values = {
            0: x,        # x (fine)
            1: x / 2,    # x/2 (medium)
            2: x / 4,    # x/4 (coarse)
            3: y,        # y (fine)
            4: y / 2,    # y/2 (medium)
            5: y / 4,    # y/4 (coarse)
            6: bias      # bias
        }

        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.output_id]


class CPPN_Interaction(CPPN):
    """
    Interaction terms: [x, y, x*y, x/y, x², y²] (6 inputs)

    Different multi-dimensional coverage:
    - Linear: x, y
    - Nonlinear: x*y (multiplicative)
    - Ratio: x/y (division, normalized)
    - Quadratic: x², y² (curvature)
    """
    def __init__(self):
        # Initialize 6 interaction input nodes + 1 bias
        nodes = [
            Node(0, 'identity', 0.0),   # x
            Node(1, 'identity', 0.0),   # y
            Node(2, 'identity', 0.0),   # x*y (computed)
            Node(3, 'identity', 0.0),   # x/y (computed)
            Node(4, 'identity', 0.0),   # x²
            Node(5, 'identity', 0.0),   # y²
            Node(6, 'identity', 0.0),   # bias
            Node(7, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        ]

        connections = []
        # Connect all interaction inputs to output
        for inp_id in [0, 1, 2, 3, 4, 5, 6]:
            connections.append(
                Connection(inp_id, 7, np.random.randn() * PRIOR_SIGMA)
            )

        super().__init__(nodes=nodes, connections=connections,
                        input_ids=[0, 1, 2, 3, 4, 5, 6], output_id=7)

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate network with interaction terms"""
        xy_product = x * y
        xy_ratio = np.divide(x, y, where=(np.abs(y) > 1e-6),
                           out=np.zeros_like(x))  # Safe division
        x_squared = x ** 2
        y_squared = y ** 2
        bias = np.ones_like(x)

        values = {
            0: x,            # x
            1: y,            # y
            2: xy_product,   # x*y
            3: xy_ratio,     # x/y (safe)
            4: x_squared,    # x²
            5: y_squared,    # y²
            6: bias          # bias
        }

        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.output_id]


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def compute_effective_dimension(weight_samples: np.ndarray) -> Dict[str, float]:
    """
    Compute effective dimensionality using Renyi entropy on eigenvalue distribution.

    Lower eff_dim = more constrained (lower intrinsic dimensionality)
    Higher eff_dim = more spread out (higher intrinsic dimensionality)
    """
    if weight_samples.shape[0] < 2:
        return {'eff_dim': np.nan, 'first_pc_var': np.nan}

    # Center data
    X = weight_samples - weight_samples.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Eigenvalue distribution
    explained_var = (S ** 2) / np.sum(S ** 2)

    # Renyi entropy: D_eff = exp(-sum(p_i * log(p_i)))
    p = explained_var[explained_var > 1e-10]
    if len(p) > 0:
        entropy = -np.sum(p * np.log(p + 1e-10))
        d_eff = np.exp(entropy)
    else:
        d_eff = 1.0

    return {
        'eff_dim': d_eff,
        'first_pc_var': explained_var[0],
        'n_components_90': int(np.argmax(np.cumsum(explained_var) >= 0.9) + 1)
    }


def sample_cppn_ensemble(cppn_class, n_samples: int = 25, size: int = 32):
    """
    Sample ensemble of CPPNs and compute statistics.

    Returns:
    - weight_samples: (n_samples, n_weights) array
    - cppn_instances: list of CPPN objects
    - rendered_images: list of rendered images
    - order_scores: list of order scores
    """
    weight_samples = []
    cppn_instances = []
    rendered_images = []
    order_scores = []

    for i in range(n_samples):
        cppn = cppn_class()
        cppn_instances.append(cppn)

        # Get weights
        w = cppn.get_weights()
        weight_samples.append(w)

        # Render image
        img = cppn.render(size=size)
        rendered_images.append(img)

        # Compute order
        order = order_multiplicative(img)
        order_scores.append(order)

    return (
        np.array(weight_samples),
        cppn_instances,
        rendered_images,
        np.array(order_scores)
    )


def run_nested_sampling_ensemble(cppn_instances: List, target_order: float = 0.5):
    """
    Run nested sampling for ensemble of CPPNs to track samples-to-order.

    Returns:
    - samples_to_order: list of sample counts needed to reach target_order
    - success_rate: fraction that reached target
    """
    samples_to_order = []
    success_count = 0

    for i, cppn in enumerate(cppn_instances[:10]):  # Sample 10 for efficiency
        try:
            # Run nested sampling
            result = nested_sampling_v3(cppn, max_nlive=50, order_threshold=target_order)

            if result and 'samples' in result:
                samples_to_order.append(result['samples'])
                if result['max_order'] >= target_order:
                    success_count += 1
        except Exception as e:
            pass  # Skip failures

    return samples_to_order, success_count / min(10, len(cppn_instances)) if cppn_instances else 0


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("RES-229: Hierarchical Multi-Scale Composition")
    print("=" * 80)

    results = {}

    # ========================================================================
    # VARIANT A: BASELINE (x, y, r)
    # ========================================================================

    print("\n[1/3] BASELINE: Standard (x, y, r) composition")
    print("-" * 80)
    try:
        weights_baseline, cpns_baseline, imgs_baseline, orders_baseline = \
            sample_cppn_ensemble(CPPN_Baseline, n_samples=25, size=32)

        metrics_baseline = compute_effective_dimension(weights_baseline)

        print(f"  ✓ Sampled {len(cpns_baseline)} CPPNs")
        print(f"  ✓ eff_dim = {metrics_baseline['eff_dim']:.4f}")
        print(f"  ✓ first_pc_var = {metrics_baseline['first_pc_var']:.4f}")
        print(f"  ✓ Mean order score = {np.mean(orders_baseline):.4f}")
        print(f"  ✓ Std order score = {np.std(orders_baseline):.4f}")
        print(f"  ✓ Order percentiles [25, 50, 75, 90] = {np.percentile(orders_baseline, [25, 50, 75, 90])}")

        # Run nested sampling on subset
        print("  Running nested sampling on 10 CPPNs...")
        samples_baseline, success_baseline = run_nested_sampling_ensemble(cpns_baseline)
        if samples_baseline:
            print(f"  ✓ Median samples-to-0.5 = {np.median(samples_baseline):.1f}")
            print(f"  ✓ Success rate (reached 0.5) = {success_baseline:.1%}")

        results['baseline'] = {
            'eff_dim': float(metrics_baseline['eff_dim']),
            'first_pc_var': float(metrics_baseline['first_pc_var']),
            'n_components_90': int(metrics_baseline['n_components_90']),
            'mean_order': float(np.mean(orders_baseline)),
            'std_order': float(np.std(orders_baseline)),
            'order_p25': float(np.percentile(orders_baseline, 25)),
            'order_p50': float(np.percentile(orders_baseline, 50)),
            'order_p75': float(np.percentile(orders_baseline, 75)),
            'order_p90': float(np.percentile(orders_baseline, 90)),
            'median_samples_to_05': float(np.median(samples_baseline)) if samples_baseline else None,
            'ns_success_rate': float(success_baseline)
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['baseline'] = {'error': str(e)}

    # ========================================================================
    # VARIANT B: HIERARCHICAL ([x, x/2, x/4, y, y/2, y/4])
    # ========================================================================

    print("\n[2/3] HIERARCHICAL: Multi-scale [x, x/2, x/4, y, y/2, y/4]")
    print("-" * 80)
    try:
        weights_hierarchy, cpns_hierarchy, imgs_hierarchy, orders_hierarchy = \
            sample_cppn_ensemble(CPPN_Hierarchy, n_samples=25, size=32)

        metrics_hierarchy = compute_effective_dimension(weights_hierarchy)

        print(f"  ✓ Sampled {len(cpns_hierarchy)} CPPNs")
        print(f"  ✓ eff_dim = {metrics_hierarchy['eff_dim']:.4f}")
        print(f"  ✓ first_pc_var = {metrics_hierarchy['first_pc_var']:.4f}")
        print(f"  ✓ Mean order score = {np.mean(orders_hierarchy):.4f}")
        print(f"  ✓ Std order score = {np.std(orders_hierarchy):.4f}")
        print(f"  ✓ Order percentiles [25, 50, 75, 90] = {np.percentile(orders_hierarchy, [25, 50, 75, 90])}")

        # Run nested sampling on subset
        print("  Running nested sampling on 10 CPPNs...")
        samples_hierarchy, success_hierarchy = run_nested_sampling_ensemble(cpns_hierarchy)
        if samples_hierarchy:
            print(f"  ✓ Median samples-to-0.5 = {np.median(samples_hierarchy):.1f}")
            print(f"  ✓ Success rate (reached 0.5) = {success_hierarchy:.1%}")

        results['hierarchy'] = {
            'eff_dim': float(metrics_hierarchy['eff_dim']),
            'first_pc_var': float(metrics_hierarchy['first_pc_var']),
            'n_components_90': int(metrics_hierarchy['n_components_90']),
            'mean_order': float(np.mean(orders_hierarchy)),
            'std_order': float(np.std(orders_hierarchy)),
            'order_p25': float(np.percentile(orders_hierarchy, 25)),
            'order_p50': float(np.percentile(orders_hierarchy, 50)),
            'order_p75': float(np.percentile(orders_hierarchy, 75)),
            'order_p90': float(np.percentile(orders_hierarchy, 90)),
            'median_samples_to_05': float(np.median(samples_hierarchy)) if samples_hierarchy else None,
            'ns_success_rate': float(success_hierarchy)
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['hierarchy'] = {'error': str(e)}

    # ========================================================================
    # VARIANT C: INTERACTION ([x, y, x*y, x/y, x², y²])
    # ========================================================================

    print("\n[3/3] INTERACTION: Interaction terms [x, y, x*y, x/y, x², y²]")
    print("-" * 80)
    try:
        weights_interaction, cpns_interaction, imgs_interaction, orders_interaction = \
            sample_cppn_ensemble(CPPN_Interaction, n_samples=25, size=32)

        metrics_interaction = compute_effective_dimension(weights_interaction)

        print(f"  ✓ Sampled {len(cpns_interaction)} CPPNs")
        print(f"  ✓ eff_dim = {metrics_interaction['eff_dim']:.4f}")
        print(f"  ✓ first_pc_var = {metrics_interaction['first_pc_var']:.4f}")
        print(f"  ✓ Mean order score = {np.mean(orders_interaction):.4f}")
        print(f"  ✓ Std order score = {np.std(orders_interaction):.4f}")
        print(f"  ✓ Order percentiles [25, 50, 75, 90] = {np.percentile(orders_interaction, [25, 50, 75, 90])}")

        # Run nested sampling on subset
        print("  Running nested sampling on 10 CPPNs...")
        samples_interaction, success_interaction = run_nested_sampling_ensemble(cpns_interaction)
        if samples_interaction:
            print(f"  ✓ Median samples-to-0.5 = {np.median(samples_interaction):.1f}")
            print(f"  ✓ Success rate (reached 0.5) = {success_interaction:.1%}")

        results['interaction'] = {
            'eff_dim': float(metrics_interaction['eff_dim']),
            'first_pc_var': float(metrics_interaction['first_pc_var']),
            'n_components_90': int(metrics_interaction['n_components_90']),
            'mean_order': float(np.mean(orders_interaction)),
            'std_order': float(np.std(orders_interaction)),
            'order_p25': float(np.percentile(orders_interaction, 25)),
            'order_p50': float(np.percentile(orders_interaction, 50)),
            'order_p75': float(np.percentile(orders_interaction, 75)),
            'order_p90': float(np.percentile(orders_interaction, 90)),
            'median_samples_to_05': float(np.median(samples_interaction)) if samples_interaction else None,
            'ns_success_rate': float(success_interaction)
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['interaction'] = {'error': str(e)}

    # ========================================================================
    # ANALYSIS & VALIDATION
    # ========================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS & HYPOTHESIS VALIDATION")
    print("=" * 80)

    baseline_order = results.get('baseline', {}).get('mean_order', np.nan)
    hierarchy_order = results.get('hierarchy', {}).get('mean_order', np.nan)
    interaction_order = results.get('interaction', {}).get('mean_order', np.nan)

    baseline_samples = results.get('baseline', {}).get('median_samples_to_05')
    hierarchy_samples = results.get('hierarchy', {}).get('median_samples_to_05')
    interaction_samples = results.get('interaction', {}).get('median_samples_to_05')

    print(f"\nOrder Score Comparison (mean):")
    print(f"  Baseline:     {baseline_order:.4f}")
    print(f"  Hierarchy:    {hierarchy_order:.4f}")
    print(f"  Interaction:  {interaction_order:.4f}")

    # Test 1: Hierarchy order gain
    if not np.isnan(baseline_order) and not np.isnan(hierarchy_order):
        order_gain_hierarchy = hierarchy_order / baseline_order if baseline_order > 0 else np.nan
        print(f"\n  Hierarchy order gain: {order_gain_hierarchy:.3f}x")
        test1_pass = order_gain_hierarchy >= 1.2
        print(f"  [TEST 1] Hierarchy ≥1.2x baseline: {test1_pass}")
    else:
        order_gain_hierarchy = np.nan
        test1_pass = False
        print(f"  [TEST 1] SKIPPED (missing data)")

    # Test 2: Hierarchy sampling efficiency
    if baseline_samples and hierarchy_samples:
        efficiency_gain = baseline_samples / hierarchy_samples if hierarchy_samples > 0 else np.nan
        print(f"\nSampling Efficiency (samples to 0.5):")
        print(f"  Baseline:     {baseline_samples:.1f}")
        print(f"  Hierarchy:    {hierarchy_samples:.1f}")
        print(f"  Efficiency gain (baseline/hierarchy): {efficiency_gain:.3f}x")
        test2_pass = efficiency_gain >= 2.0
        print(f"  [TEST 2] Hierarchy ≥2x efficiency: {test2_pass}")
    else:
        efficiency_gain = np.nan
        test2_pass = False
        print(f"  [TEST 2] SKIPPED (missing sampling data)")

    # Test 3: Interaction comparison
    if not np.isnan(baseline_order) and not np.isnan(interaction_order):
        order_gain_interaction = interaction_order / baseline_order if baseline_order > 0 else np.nan
        print(f"\n  Interaction order gain: {order_gain_interaction:.3f}x")
    else:
        order_gain_interaction = np.nan

    # Conclusion
    print("\n" + "-" * 80)
    if test1_pass or test2_pass:
        conclusion = "validate"
        print("✓ VALIDATED: Hierarchical composition improves CPPN efficiency")
        if test1_pass:
            print(f"  - Order: {order_gain_hierarchy:.3f}x baseline")
        if test2_pass:
            print(f"  - Efficiency: {efficiency_gain:.3f}x baseline")
    else:
        conclusion = "refute"
        print("✗ REFUTED: Hierarchical composition does NOT meet thresholds")
        print(f"  - Order gain: {order_gain_hierarchy:.3f}x (need 1.2x)")
        print(f"  - Efficiency gain: {efficiency_gain:.3f}x (need 2.0x)")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/hierarchical_composition_architecture')
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment_id": "RES-229",
        "title": "Hierarchical Multi-Scale Composition",
        "method": "Multi-scale hierarchical coordinate inputs vs baseline and interaction terms",
        "n_samples_per_variant": 25,
        "results": {
            "baseline": results.get('baseline', {}),
            "hierarchy": results.get('hierarchy', {}),
            "interaction": results.get('interaction', {})
        },
        "comparison": {
            "hierarchy_order_gain": float(order_gain_hierarchy) if not np.isnan(order_gain_hierarchy) else None,
            "hierarchy_efficiency_gain": float(efficiency_gain) if not np.isnan(efficiency_gain) else None,
            "interaction_order_gain": float(order_gain_interaction) if not np.isnan(order_gain_interaction) else None
        },
        "validation": {
            "test1_hierarchy_order_12x": test1_pass,
            "test2_hierarchy_efficiency_2x": test2_pass,
            "conclusion": conclusion
        }
    }

    results_path = results_dir / 'res_229_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"RES-229 | hierarchical_composition_architecture | {conclusion} | " +
          f"order_gain={order_gain_hierarchy:.2f}x efficiency_gain={efficiency_gain:.2f}x")


if __name__ == '__main__':
    main()
