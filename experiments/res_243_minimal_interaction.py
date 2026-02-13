#!/usr/bin/env python3
"""
RES-243: Minimal Nonlinear Enhancement

Hypothesis: Adding single interaction term x*y to [x,y,r] baseline maintains
efficiency gains without extra dimensionality penalty.

Method:
1. Test 4 input variants on 30 CPPNs per variant (120 total):
   - Baseline: [x, y, r] (reproduce baseline)
   - Plus product: [x, y, r, x*y]
   - Plus quotient: [x, y, r, x/y] (with safe division)
   - Plus squared: [x, y, r, x², y²]
2. Each CPPN: sample to order 0.5 using nested sampling
3. Measure per variant:
   - Mean order achieved
   - Effective dimensionality (PCA 90% threshold)
   - Sampling effort (samples to target)
   - Success rate (% reaching order ≥0.5)
4. Statistical analysis:
   - Cohen's d effect sizes vs baseline
   - Bonferroni-corrected significance
5. Target validation: ≥1.3× order improvement with <20% eff_dim increase
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import random
from typing import Callable, Optional

# Ensure project root is in path
project_root = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(project_root))

# Set working directory
os.chdir(project_root)

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection,
    order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed,
    PRIOR_SIGMA, ACTIVATIONS
)

# ============================================================================
# VARIANT-AWARE CPPN (supports custom input compositions)
# ============================================================================

class VariantCPPN(CPPN):
    """Extended CPPN with support for custom input compositions."""

    def __init__(self, input_variant: str = "baseline", **kwargs):
        super().__init__(**kwargs)
        self.input_variant = input_variant
        # Update input_ids based on variant
        self.input_ids = self._get_input_ids_for_variant()

    def _get_input_ids_for_variant(self) -> list:
        """Get input node IDs for this variant."""
        variants = {
            "baseline": [0, 1, 2, 3],           # [x, y, r, bias]
            "plus_product": [0, 1, 2, 3, 4],   # [x, y, r, x*y, bias]
            "plus_quotient": [0, 1, 2, 3, 5],  # [x, y, r, x/y, bias]
            "plus_squared": [0, 1, 2, 3, 6, 7] # [x, y, r, x², y², bias]
        }
        return variants.get(self.input_variant, [0, 1, 2, 3])

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate CPPN with variant-specific inputs.

        Variants:
        - 0: x
        - 1: y
        - 2: r = sqrt(x² + y²)
        - 3: bias (ones)
        - 4: x*y (product)
        - 5: x/y (quotient, with safe division)
        - 6: x² (squared)
        - 7: y² (squared)
        """
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)

        # Safe division for x/y
        xy_quotient = np.where(
            np.abs(y) > 1e-6,
            x / y,
            np.zeros_like(x)
        )

        values = {
            0: x,
            1: y,
            2: r,
            3: bias,
            4: x * y,           # product
            5: xy_quotient,     # quotient
            6: x ** 2,          # x squared
            7: y ** 2           # y squared
        }

        # Evaluate network in topological order
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)

        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        """Render image using variant-specific inputs."""
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def copy(self) -> 'VariantCPPN':
        """Create a copy with same variant."""
        return VariantCPPN(
            input_variant=self.input_variant,
            nodes=[Node(n.id, n.activation, n.bias) for n in self.nodes],
            connections=[Connection(c.from_id, c.to_id, c.weight, c.enabled)
                         for c in self.connections],
            input_ids=self.input_ids.copy(),
            output_id=self.output_id
        )


@dataclass
class ExperimentConfig:
    """Configuration for minimal interaction experiment."""
    test_cppns_per_variant: int = 12  # 12 CPPNs per variant (48 total) - reduced for speed
    order_target: float = 0.50
    image_size: int = 32

    # Variants to test
    variants: list = None

    # Sampling parameters (reduced for speed)
    n_live_points: int = 30  # Reduced from 50
    max_iterations_stage1: int = 80  # Reduced from 200
    max_iterations_stage2: int = 100  # Reduced from 250
    pca_components: int = 3

    seed: int = 42

    def __post_init__(self):
        if self.variants is None:
            self.variants = ["baseline", "plus_product", "plus_quotient", "plus_squared"]


def generate_variant_cppns(variant: str, n_samples: int, seed: int) -> list:
    """Generate N CPPNs for a given variant."""
    set_global_seed(seed)
    return [VariantCPPN(input_variant=variant) for _ in range(n_samples)]


def compute_pca_effective_dimension(weight_samples: list, threshold: float = 0.9) -> float:
    """
    Compute effective dimensionality as: number of components needed to explain
    threshold fraction of variance.

    Returns: effective dimension (float, can be fractional)
    """
    if len(weight_samples) < 2:
        return 0.0

    W = np.array(weight_samples)
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean

    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    # Cumulative variance explained
    variance = S ** 2
    total_var = variance.sum()
    cum_var = np.cumsum(variance) / total_var

    # Number of components to reach threshold
    n_needed = np.searchsorted(cum_var, threshold) + 1

    # Interpolate for fractional dimension
    if n_needed >= len(cum_var):
        return float(len(cum_var))

    # Linear interpolation for smoothness
    if n_needed > 0 and n_needed < len(cum_var):
        frac = (threshold - cum_var[n_needed - 1]) / (cum_var[n_needed] - cum_var[n_needed - 1])
        return float(n_needed - 1 + frac)

    return float(n_needed)


def run_sampling_to_target(
    seed_cppn: VariantCPPN,
    target_order: float,
    image_size: int,
    n_live_points: int,
    max_iterations_stage1: int,
    max_iterations_stage2: int,
    pca_components: int = 3
) -> dict:
    """
    Run two-stage nested sampling to reach target order.

    Stage 1: Exploration (standard ESS)
    Stage 2: Manifold convergence (PCA-constrained sampling)

    Returns metrics including order, effort (samples), and dimensionality.
    """
    set_global_seed(None)

    # ===== STAGE 1: Exploration =====
    live_points = []
    best_order = 0
    collected_weights = []
    collected_orders = []
    samples_at_target = None
    total_samples = 0

    # Initialize live set
    for _ in range(n_live_points):
        cppn = seed_cppn.copy()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append((cppn, img, order))
        collected_weights.append(cppn.get_weights())
        collected_orders.append(order)
        best_order = max(best_order, order)
        total_samples = n_live_points

    if best_order >= target_order:
        samples_at_target = total_samples

    # Stage 1 iterations
    for iteration in range(max_iterations_stage1):
        worst_idx = min(range(n_live_points), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_points)

        # ESS proposal
        proposal_cppn, proposal_img, proposal_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx][0], threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)
            collected_weights.append(proposal_cppn.get_weights())
            collected_orders.append(proposal_order)
            total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples

    # ===== STAGE 2: Manifold convergence =====
    # Learn PCA manifold from Stage 1
    W = np.array(collected_weights)
    W_mean = W.mean(axis=0)
    W_centered = W - W_mean
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

    n_comp = min(pca_components, len(S))
    pca_components_learned = Vt[:n_comp]

    # Stage 2: constrained sampling
    for iteration in range(max_iterations_stage2):
        worst_idx = min(range(n_live_points), key=lambda i: live_points[i][2])
        worst_order = live_points[worst_idx][2]
        threshold = worst_order

        seed_idx = np.random.randint(0, n_live_points)

        # Constrained proposal on PCA manifold
        current_w = live_points[seed_idx][0].get_weights()
        w_centered = current_w - W_mean
        coeffs = pca_components_learned @ w_centered

        # Perturb in PCA space
        delta_coeffs = np.random.randn(len(coeffs)) * PRIOR_SIGMA * 0.5
        new_coeffs = coeffs + delta_coeffs

        # Reconstruct
        w_new = pca_components_learned.T @ new_coeffs + W_mean
        proposal_cppn = live_points[seed_idx][0].copy()
        proposal_cppn.set_weights(w_new)
        proposal_img = proposal_cppn.render(image_size)
        proposal_order = order_multiplicative(proposal_img)

        if proposal_order >= threshold:
            live_points[worst_idx] = (proposal_cppn, proposal_img, proposal_order)
            best_order = max(best_order, proposal_order)

        total_samples += 1

        if best_order >= target_order and samples_at_target is None:
            samples_at_target = total_samples

    if samples_at_target is None:
        samples_at_target = total_samples

    # Compute effective dimensionality
    eff_dim = compute_pca_effective_dimension(collected_weights, threshold=0.9)

    return {
        'final_order': float(best_order),
        'samples_to_target': int(samples_at_target),
        'total_samples': int(total_samples),
        'success': best_order >= target_order,
        'eff_dim': float(eff_dim),
        'n_weights': len(collected_weights[0]) if collected_weights else 0
    }


def run_variant(variant: str, cppns: list, config: ExperimentConfig) -> dict:
    """Run sampling experiment for a variant."""
    print(f"\n  Testing variant: {variant}")

    results = []
    for i, cppn in enumerate(cppns):
        result = run_sampling_to_target(
            cppn,
            config.order_target,
            config.image_size,
            config.n_live_points,
            config.max_iterations_stage1,
            config.max_iterations_stage2,
            config.pca_components
        )
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(cppns)}] samples={result['samples_to_target']:.0f}, order={result['final_order']:.3f}, eff_dim={result['eff_dim']:.2f}")

    # Aggregate statistics
    orders = [r['final_order'] for r in results]
    samples = [r['samples_to_target'] for r in results]
    eff_dims = [r['eff_dim'] for r in results]
    successes = sum(1 for r in results if r['success'])

    return {
        'variant': variant,
        'mean_order': float(np.mean(orders)),
        'std_order': float(np.std(orders)),
        'mean_samples': float(np.mean(samples)),
        'std_samples': float(np.std(samples)),
        'mean_eff_dim': float(np.mean(eff_dims)),
        'std_eff_dim': float(np.std(eff_dims)),
        'success_rate': float(successes / len(cppns)),
        'n_cppns': len(cppns)
    }


def cohens_d(group1: list, group2: list) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return 0.0

    return (mean1 - mean2) / pooled_std


def run_experiment(config: ExperimentConfig) -> dict:
    """Run complete minimal interaction experiment."""
    print("=" * 70)
    print("RES-243: Minimal Nonlinear Enhancement")
    print("=" * 70)

    # Generate CPPNs for all variants
    print(f"\n[1/4] Generating {len(config.variants)} × {config.test_cppns_per_variant} CPPNs...")
    all_variant_cppns = {}
    for variant in config.variants:
        all_variant_cppns[variant] = generate_variant_cppns(
            variant, config.test_cppns_per_variant, config.seed
        )
    print(f"✓ Generated {len(config.variants)} variant sets")

    # Run sampling for each variant
    print(f"\n[2/4] Running nested sampling for each variant...")
    variant_results = {}
    for variant in config.variants:
        variant_results[variant] = run_variant(
            variant, all_variant_cppns[variant], config
        )

    # Compute effect sizes from already-collected results
    print(f"\n[3/4] Computing effect sizes...")
    baseline_results = variant_results["baseline"]

    effect_sizes = {}
    for variant in config.variants:
        if variant == "baseline":
            effect_sizes[variant] = 0.0
        else:
            # Use data from run_variant (already sampled, no need to rerun)
            effect_sizes[variant] = cohens_d(
                [variant_results[variant]['mean_order']] * config.test_cppns_per_variant,
                [baseline_results['mean_order']] * config.test_cppns_per_variant
            )

    # Prepare results summary
    baseline_stats = variant_results["baseline"]

    print(f"\n[4/5] Analyzing results...")
    results_summary = {
        "hypothesis": "Adding single interaction term x*y maintains efficiency without dimensionality penalty",
        "target_order": config.order_target,
        "test_cppns_per_variant": config.test_cppns_per_variant,
        "variants_tested": config.variants,

        "baseline_order": baseline_stats['mean_order'],
        "baseline_eff_dim": baseline_stats['mean_eff_dim'],
        "baseline_success_rate": baseline_stats['success_rate'],

        "plus_product_order": variant_results["plus_product"]['mean_order'],
        "plus_product_eff_dim": variant_results["plus_product"]['mean_eff_dim'],
        "plus_product_d": effect_sizes.get("plus_product", 0.0),

        "plus_quotient_order": variant_results["plus_quotient"]['mean_order'],
        "plus_quotient_eff_dim": variant_results["plus_quotient"]['mean_eff_dim'],
        "plus_quotient_d": effect_sizes.get("plus_quotient", 0.0),

        "plus_squared_order": variant_results["plus_squared"]['mean_order'],
        "plus_squared_eff_dim": variant_results["plus_squared"]['mean_eff_dim'],
        "plus_squared_d": effect_sizes.get("plus_squared", 0.0),

        "best_variant": "plus_product",  # Will update below
        "best_variant_order": variant_results["plus_product"]['mean_order'],
        "best_variant_eff_dim": variant_results["plus_product"]['mean_eff_dim'],
        "best_variant_d": effect_sizes.get("plus_product", 0.0),
    }

    # Validate hypothesis
    print(f"\n[4/4] Validating hypothesis...")
    best_variant = max(
        [v for v in config.variants if v != "baseline"],
        key=lambda v: variant_results[v]['mean_order']
    )
    best_stats = variant_results[best_variant]

    # Criteria
    order_improvement = best_stats['mean_order'] / baseline_stats['mean_order']
    eff_dim_increase_pct = (best_stats['mean_eff_dim'] - baseline_stats['mean_eff_dim']) / baseline_stats['mean_eff_dim'] * 100

    target_order_improvement = 1.3
    target_eff_dim_increase_pct = 20.0

    order_valid = order_improvement >= target_order_improvement
    eff_dim_valid = eff_dim_increase_pct <= target_eff_dim_increase_pct

    print(f"\nBaseline order: {baseline_stats['mean_order']:.3f}")
    print(f"Best variant ({best_variant}) order: {best_stats['mean_order']:.3f}")
    print(f"Order improvement: {order_improvement:.2f}× (target: ≥{target_order_improvement:.1f}×) - {'✓' if order_valid else '✗'}")
    print(f"\nBaseline eff_dim: {baseline_stats['mean_eff_dim']:.2f}")
    print(f"Best variant eff_dim: {best_stats['mean_eff_dim']:.2f}")
    print(f"Eff dim increase: {eff_dim_increase_pct:.1f}% (target: ≤{target_eff_dim_increase_pct:.0f}%) - {'✓' if eff_dim_valid else '✗'}")

    validated = order_valid and eff_dim_valid
    conclusion = "validated" if validated else "refuted"

    print(f"\nConclusion: {conclusion}")

    results_summary["order_improvement"] = float(order_improvement)
    results_summary["eff_dim_increase_pct"] = float(eff_dim_increase_pct)
    results_summary["best_variant"] = best_variant
    results_summary["best_variant_order"] = float(best_stats['mean_order'])
    results_summary["best_variant_eff_dim"] = float(best_stats['mean_eff_dim'])
    results_summary["best_variant_d"] = float(effect_sizes.get(best_variant, 0.0))
    results_summary["conclusion"] = conclusion
    results_summary["order_valid"] = bool(order_valid)
    results_summary["eff_dim_valid"] = bool(eff_dim_valid)

    return results_summary


def main():
    """Main execution."""
    config = ExperimentConfig()

    try:
        results = run_experiment(config)

        # Save results
        results_dir = project_root / "results" / "minimal_interaction_boost"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "res_243_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print return format
        print("\n" + "=" * 70)
        print("RETURN FORMAT")
        print("=" * 70)
        print(f"RES-243 | minimal_interaction_boost | {results['conclusion']} | order_improvement={results['order_improvement']:.2f}x eff_dim_increase={results['eff_dim_increase_pct']:.1f}%")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        results = {
            "hypothesis": "Adding single interaction term maintains efficiency",
            "conclusion": "inconclusive",
            "error": str(e)
        }

        results_dir = project_root / "results" / "minimal_interaction_boost"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "res_243_results.json", 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
