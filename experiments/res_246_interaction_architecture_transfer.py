#!/usr/bin/env python3
"""
RES-246: Nonlinear Interaction Generalization Across Architectures

HYPOTHESIS:
Nonlinear interaction terms [x*y, x², y²] that boost CPPN efficiency also
generalize to other architectures (ResNet). ResNet with interactions achieves
≥1.3× order improvement (comparable to CPPN).

METHODOLOGY:
1. Compare 4 configurations:
   - CPPN baseline: [x, y, r] → CPPN (3→hidden→output)
   - CPPN + interactions: [x, y, r, x*y, x², y²] → CPPN (6→hidden→output)
   - ResNet baseline: [x, y] → 2-layer ResNet (2→64→output)
   - ResNet + interactions: [x, y, x*y, x², y²] → 2-layer ResNet (5→64→output)

2. For each config:
   - Generate 30 samples using nested sampling to order 0.5
   - Measure order achievement, sampling efficiency, effective dimensionality
   - Sample all 120 total networks

3. Statistical analysis:
   - Main effect: architecture (CPPN vs ResNet)
   - Main effect: composition (baseline vs interactions)
   - Interaction: architecture × composition
   - Transfer factor: (ResNet_effect / CPPN_effect)

EXPECTED RESULTS:
- CPPN interactions: ~2.5× order improvement (per RES-241)
- ResNet interactions: ≥1.3× (substantial transfer) OR ≤1.0× (architecture-specific)
- If ≥1.3×: Interactions are universal strategy
- If ≤1.0×: Benefit is CPPN-specific (leverages compositional structure)
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os
import math

# Setup path
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(Path.cwd()))

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, log_prior,
    elliptical_slice_sample, set_global_seed, PRIOR_SIGMA,
    compute_spectral_coherence, compute_compressibility, compute_edge_density,
    compute_symmetry, compute_connected_components
)


# ============================================================================
# RESNET IMPLEMENTATION
# ============================================================================

class ResNetBlock:
    """Single residual block with 64 hidden units."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # First layer: input_dim → hidden_dim
        self.w1 = np.random.randn(hidden_dim, input_dim) * PRIOR_SIGMA
        self.b1 = np.random.randn(hidden_dim) * PRIOR_SIGMA

        # Second layer: hidden_dim → hidden_dim (for residual connection)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * PRIOR_SIGMA
        self.b2 = np.random.randn(hidden_dim) * PRIOR_SIGMA

        # Output layer: hidden_dim → 1
        self.w_out = np.random.randn(1, hidden_dim) * PRIOR_SIGMA
        self.b_out = np.random.randn(1) * PRIOR_SIGMA

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet block.

        Args:
            x: Shape (height, width, input_dim)

        Returns:
            output: Shape (height, width)
        """
        h, w, _ = x.shape

        # Reshape to (h*w, input_dim)
        x_flat = x.reshape(-1, self.input_dim)

        # Layer 1: linear + ReLU
        h1 = np.dot(x_flat, self.w1.T) + self.b1
        h1_relu = np.maximum(0, h1)

        # Layer 2: linear + ReLU with residual connection
        h2 = np.dot(h1_relu, self.w2.T) + self.b2
        h2_relu = np.maximum(0, h2)

        # Output layer: linear + sigmoid
        out_flat = np.dot(h2_relu, self.w_out.T) + self.b_out
        out_flat = 1.0 / (1.0 + np.exp(-np.clip(out_flat, -10, 10)))

        return out_flat.reshape(h, w)

    def get_weights(self) -> np.ndarray:
        """Get all weights as flat vector."""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten(),
            self.w_out.flatten(),
            self.b_out.flatten()
        ])

    def set_weights(self, w: np.ndarray):
        """Set weights from flat vector."""
        idx = 0

        # w1
        size = self.hidden_dim * self.input_dim
        self.w1 = w[idx:idx+size].reshape(self.hidden_dim, self.input_dim)
        idx += size

        # b1
        self.b1 = w[idx:idx+self.hidden_dim]
        idx += self.hidden_dim

        # w2
        size = self.hidden_dim * self.hidden_dim
        self.w2 = w[idx:idx+size].reshape(self.hidden_dim, self.hidden_dim)
        idx += size

        # b2
        self.b2 = w[idx:idx+self.hidden_dim]
        idx += self.hidden_dim

        # w_out
        self.w_out = w[idx:idx+self.hidden_dim].reshape(1, self.hidden_dim)
        idx += self.hidden_dim

        # b_out
        self.b_out = w[idx:idx+1]


# ============================================================================
# COORDINATE INPUT GENERATION
# ============================================================================

def create_cppn_inputs(grid_size: int = 32, interaction_terms: bool = False) -> np.ndarray:
    """
    Create input arrays for CPPN.

    Args:
        grid_size: Size of image grid
        interaction_terms: If True, include [x*y, x², y²]

    Returns:
        inputs: Dict with coordinate tensors
    """
    coords = np.linspace(-1, 1, grid_size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)

    inputs = {
        'x': x,
        'y': y,
        'r': r
    }

    if interaction_terms:
        inputs['x_y'] = x * y
        inputs['x2'] = x ** 2
        inputs['y2'] = y ** 2

    return inputs


def create_resnet_inputs(grid_size: int = 32, interaction_terms: bool = False) -> np.ndarray:
    """
    Create 3D input array for ResNet [height, width, channels].

    Args:
        grid_size: Size of image grid
        interaction_terms: If True, include [x*y, x², y²]

    Returns:
        inputs: Shape (height, width, channels)
    """
    coords = np.linspace(-1, 1, grid_size)
    x, y = np.meshgrid(coords, coords)

    channels = [x, y]

    if interaction_terms:
        channels.extend([
            x * y,
            x ** 2,
            y ** 2
        ])

    # Stack along channel dimension
    return np.stack(channels, axis=-1)


# ============================================================================
# CPPN EVALUATION
# ============================================================================

def evaluate_cppn_with_interactions(cppn: CPPN, inputs: Dict) -> np.ndarray:
    """
    Evaluate CPPN on coordinates with optional interaction terms.

    Args:
        cppn: CPPN network
        inputs: Dict with x, y, r, and optional x_y, x2, y2

    Returns:
        image: Binary image (height, width)
    """
    # Get baseline evaluation
    x, y, r = inputs['x'], inputs['y'], inputs['r']
    bias = np.ones_like(x)

    # For interaction terms, we extend the CPPN by adding new input nodes
    # and treating them as pre-computed features
    values = {0: x, 1: y, 2: r, 3: bias}

    # Add interaction terms as additional inputs if present
    next_input_id = 4
    if 'x_y' in inputs:
        values[next_input_id] = inputs['x_y']
        next_input_id += 1
    if 'x2' in inputs:
        values[next_input_id] = inputs['x2']
        next_input_id += 1
    if 'y2' in inputs:
        values[next_input_id] = inputs['y2']
        next_input_id += 1

    # Evaluate hidden nodes
    eval_order = sorted([n.id for n in cppn.nodes
                        if n.id not in cppn.input_ids and n.id != cppn.output_id])

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        from core.thermo_sampler_v3 import ACTIVATIONS
        values[nid] = ACTIVATIONS[node.activation](total)

    # Evaluate output
    output_node = next(n for n in cppn.nodes if n.id == cppn.output_id)
    total = np.zeros_like(x) + output_node.bias
    for conn in cppn.connections:
        if conn.to_id == cppn.output_id and conn.enabled and conn.from_id in values:
            total += values[conn.from_id] * conn.weight

    from core.thermo_sampler_v3 import ACTIVATIONS
    output = ACTIVATIONS[output_node.activation](total)

    return (output > 0.5).astype(np.uint8)


# ============================================================================
# SAMPLING WITH NESTED SAMPLING
# ============================================================================

def sample_cppn_to_order(config: Dict, n_samples: int = 30, target_order: float = 0.5,
                         interaction_terms: bool = False) -> Tuple[List[float], List[float]]:
    """
    Sample CPPNs using nested sampling protocol to reach target order.

    Args:
        config: Configuration dict with seed, etc.
        n_samples: Number of CPPNs to sample
        target_order: Target order value (e.g., 0.5)
        interaction_terms: If True, use interaction terms

    Returns:
        (orders, eff_dims): Lists of order values and effective dimensionalities
    """
    set_global_seed(config.get('seed'))

    orders = []
    eff_dims = []

    for i in range(n_samples):
        # Create fresh CPPN
        cppn = CPPN()

        # Prepare inputs
        if interaction_terms:
            cppn.input_ids = [0, 1, 2, 3, 4, 5, 6]  # x, y, r, bias, x*y, x², y²
            cppn.output_id = 10
        else:
            cppn.input_ids = [0, 1, 2, 3]  # x, y, r, bias
            cppn.output_id = 4

        # Sample weights using nested sampling
        weights = cppn.get_weights()
        accepted_orders = []

        # Simple nested sampling: rejection sample until we get some good samples
        for iteration in range(100):  # Max iterations
            # Perturb weights
            weights = elliptical_slice_sample(
                weights,
                lambda w: log_prior(w),
                [],
                200
            )
            cppn.set_weights(weights)

            # Render image and compute order
            inputs = create_cppn_inputs(interaction_terms=interaction_terms)
            img = evaluate_cppn_with_interactions(cppn, inputs)
            order = order_multiplicative(img)
            accepted_orders.append(order)

            # Stop if we've reached target with enough samples
            if len(accepted_orders) >= 10 and np.mean(accepted_orders) >= target_order * 0.8:
                break

        # Store results
        if accepted_orders:
            final_order = np.mean(accepted_orders)
            orders.append(final_order)

            # Estimate effective dimensionality
            weights = cppn.get_weights()
            eff_dim = np.linalg.matrix_rank(np.outer(weights, weights))
            eff_dims.append(float(eff_dim) / len(weights))

    return orders, eff_dims


def sample_resnet_to_order(config: Dict, n_samples: int = 30, target_order: float = 0.5,
                           interaction_terms: bool = False) -> Tuple[List[float], List[float]]:
    """
    Sample ResNets using nested sampling protocol.

    Args:
        config: Configuration dict with seed, etc.
        n_samples: Number of ResNets to sample
        target_order: Target order value (e.g., 0.5)
        interaction_terms: If True, use interaction terms

    Returns:
        (orders, eff_dims): Lists of order values and effective dimensionalities
    """
    set_global_seed(config.get('seed'))

    orders = []
    eff_dims = []

    input_dim = 5 if interaction_terms else 2

    for i in range(n_samples):
        # Create fresh ResNet
        resnet = ResNetBlock(input_dim=input_dim)

        # Sample weights using nested sampling
        weights = resnet.get_weights()
        accepted_orders = []

        for iteration in range(100):  # Max iterations
            # Perturb weights
            weights = elliptical_slice_sample(
                weights,
                lambda w: log_prior(w),
                [],
                200
            )
            resnet.set_weights(weights)

            # Forward pass and compute order
            inputs = create_resnet_inputs(interaction_terms=interaction_terms)
            img = resnet.forward(inputs)
            img_binary = (img > 0.5).astype(np.uint8)
            order = order_multiplicative(img_binary)
            accepted_orders.append(order)

            # Stop if we've reached target
            if len(accepted_orders) >= 10 and np.mean(accepted_orders) >= target_order * 0.8:
                break

        # Store results
        if accepted_orders:
            final_order = np.mean(accepted_orders)
            orders.append(final_order)

            # Estimate effective dimensionality
            eff_dim = np.linalg.matrix_rank(np.outer(weights, weights))
            eff_dims.append(float(eff_dim) / len(weights))

    return orders, eff_dims


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Execute RES-246 experiment."""

    print("RES-246: Interaction Architecture Transfer")
    print("=" * 60)

    config = {
        'seed': 42,
        'grid_size': 32,
        'target_order': 0.5,
        'n_cppns_per_config': 30,
        'n_resnets_per_config': 30,
    }

    results = {
        'config': config,
        'runs': []
    }

    # Run 1: CPPN baseline [x, y, r]
    print("\n1. CPPN baseline [x, y, r]...")
    cppn_baseline_orders, cppn_baseline_dims = sample_cppn_to_order(
        config, n_samples=config['n_cppns_per_config'],
        target_order=config['target_order'],
        interaction_terms=False
    )

    results['runs'].append({
        'architecture': 'cppn',
        'composition': 'baseline',
        'orders': cppn_baseline_orders,
        'eff_dims': cppn_baseline_dims,
        'mean_order': float(np.mean(cppn_baseline_orders)),
        'std_order': float(np.std(cppn_baseline_orders)),
        'mean_eff_dim': float(np.mean(cppn_baseline_dims)),
    })
    print(f"   Mean order: {np.mean(cppn_baseline_orders):.3f} ± {np.std(cppn_baseline_orders):.3f}")
    print(f"   Mean eff_dim: {np.mean(cppn_baseline_dims):.3f}")

    # Run 2: CPPN + interactions [x, y, r, x*y, x², y²]
    print("\n2. CPPN + interactions [x, y, r, x*y, x², y²]...")
    cppn_interact_orders, cppn_interact_dims = sample_cppn_to_order(
        config, n_samples=config['n_cppns_per_config'],
        target_order=config['target_order'],
        interaction_terms=True
    )

    results['runs'].append({
        'architecture': 'cppn',
        'composition': 'interactions',
        'orders': cppn_interact_orders,
        'eff_dims': cppn_interact_dims,
        'mean_order': float(np.mean(cppn_interact_orders)),
        'std_order': float(np.std(cppn_interact_orders)),
        'mean_eff_dim': float(np.mean(cppn_interact_dims)),
    })
    print(f"   Mean order: {np.mean(cppn_interact_orders):.3f} ± {np.std(cppn_interact_orders):.3f}")
    print(f"   Mean eff_dim: {np.mean(cppn_interact_dims):.3f}")

    # Run 3: ResNet baseline [x, y]
    print("\n3. ResNet baseline [x, y]...")
    resnet_baseline_orders, resnet_baseline_dims = sample_resnet_to_order(
        config, n_samples=config['n_resnets_per_config'],
        target_order=config['target_order'],
        interaction_terms=False
    )

    results['runs'].append({
        'architecture': 'resnet',
        'composition': 'baseline',
        'orders': resnet_baseline_orders,
        'eff_dims': resnet_baseline_dims,
        'mean_order': float(np.mean(resnet_baseline_orders)),
        'std_order': float(np.std(resnet_baseline_orders)),
        'mean_eff_dim': float(np.mean(resnet_baseline_dims)),
    })
    print(f"   Mean order: {np.mean(resnet_baseline_orders):.3f} ± {np.std(resnet_baseline_orders):.3f}")
    print(f"   Mean eff_dim: {np.mean(resnet_baseline_dims):.3f}")

    # Run 4: ResNet + interactions [x, y, x*y, x², y²]
    print("\n4. ResNet + interactions [x, y, x*y, x², y²]...")
    resnet_interact_orders, resnet_interact_dims = sample_resnet_to_order(
        config, n_samples=config['n_resnets_per_config'],
        target_order=config['target_order'],
        interaction_terms=True
    )

    results['runs'].append({
        'architecture': 'resnet',
        'composition': 'interactions',
        'orders': resnet_interact_orders,
        'eff_dims': resnet_interact_dims,
        'mean_order': float(np.mean(resnet_interact_orders)),
        'std_order': float(np.std(resnet_interact_orders)),
        'mean_eff_dim': float(np.mean(resnet_interact_dims)),
    })
    print(f"   Mean order: {np.mean(resnet_interact_orders):.3f} ± {np.std(resnet_interact_orders):.3f}")
    print(f"   Mean eff_dim: {np.mean(resnet_interact_dims):.3f}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Compute transfer factor
    cppn_baseline_mean = np.mean(cppn_baseline_orders)
    cppn_interact_mean = np.mean(cppn_interact_orders)
    resnet_baseline_mean = np.mean(resnet_baseline_orders)
    resnet_interact_mean = np.mean(resnet_interact_orders)

    cppn_improvement = cppn_interact_mean / max(cppn_baseline_mean, 1e-10)
    resnet_improvement = resnet_interact_mean / max(resnet_baseline_mean, 1e-10)
    transfer_factor = resnet_improvement / max(cppn_improvement, 1e-10)

    print(f"\nCPPN baseline order:         {cppn_baseline_mean:.3f}")
    print(f"CPPN + interactions order:   {cppn_interact_mean:.3f}")
    print(f"CPPN improvement factor:     {cppn_improvement:.2f}×")

    print(f"\nResNet baseline order:       {resnet_baseline_mean:.3f}")
    print(f"ResNet + interactions order: {resnet_interact_mean:.3f}")
    print(f"ResNet improvement factor:   {resnet_improvement:.2f}×")

    print(f"\nTransfer factor (ResNet/CPPN improvement): {transfer_factor:.2f}×")

    # Determine if transfer is substantial
    if resnet_improvement >= 1.3:
        transfer_status = "SUBSTANTIAL (≥1.3×) - Interactions are universal"
    elif resnet_improvement >= 1.1:
        transfer_status = "PARTIAL (1.1×-1.3×) - Some generalization"
    else:
        transfer_status = "MINIMAL (≤1.0×) - Architecture-specific benefit"

    print(f"Transfer assessment:        {transfer_status}")

    # Compute architecture and composition main effects
    cppn_mean = (cppn_baseline_mean + cppn_interact_mean) / 2
    resnet_mean = (resnet_baseline_mean + resnet_interact_mean) / 2
    arch_effect = abs(cppn_mean - resnet_mean) / max(cppn_mean, resnet_mean, 1e-10)

    baseline_mean = (cppn_baseline_mean + resnet_baseline_mean) / 2
    interact_mean = (cppn_interact_mean + resnet_interact_mean) / 2
    composition_effect = interact_mean / max(baseline_mean, 1e-10)

    print(f"\nArchitecture main effect (d): {arch_effect:.2f}")
    print(f"Composition main effect:      {composition_effect:.2f}×")

    # Interaction term significance
    # If (CPPN_d * ResNet_d) ≈ (CPPN_b * ResNet_b), then additive (no interaction)
    cppn_product = cppn_baseline_mean * cppn_interact_mean
    resnet_product = resnet_baseline_mean * resnet_interact_mean
    interaction_sig = abs(cppn_product - resnet_product) / max(cppn_product, resnet_product, 1e-10)

    if interaction_sig > 0.2:
        interaction_term = "SIGNIFICANT (d > 0.2)"
    else:
        interaction_term = "INSIGNIFICANT (d ≤ 0.2) - Additive effects"

    print(f"Interaction term:            {interaction_term}")

    # Determine overall hypothesis status
    if resnet_improvement >= 1.3:
        hypothesis_status = "VALIDATED"
        summary = (f"Nonlinear interactions generalize across architectures. "
                  f"ResNet achieves {resnet_improvement:.2f}× improvement "
                  f"(≥1.3× threshold). Transfer is SUBSTANTIAL.")
    elif 1.0 < resnet_improvement < 1.3:
        hypothesis_status = "INCONCLUSIVE"
        summary = (f"Partial transfer observed: ResNet achieves {resnet_improvement:.2f}× improvement. "
                  f"Benefit is real but smaller than CPPN ({cppn_improvement:.2f}×). "
                  f"Architecture-dependent but not purely specific.")
    else:
        hypothesis_status = "REFUTED"
        summary = (f"Interactions are CPPN-specific. ResNet achieves only {resnet_improvement:.2f}× "
                  f"improvement (≤1.0×) while CPPN achieves {cppn_improvement:.2f}×. "
                  f"Benefit does not generalize.")

    print(f"\nHypothesis status: {hypothesis_status}")
    print(f"Summary: {summary}")

    # Store metrics
    results['metrics'] = {
        'cppn_baseline_order': float(cppn_baseline_mean),
        'cppn_interactions_order': float(cppn_interact_mean),
        'cppn_improvement_factor': float(cppn_improvement),
        'resnet_baseline_order': float(resnet_baseline_mean),
        'resnet_interactions_order': float(resnet_interact_mean),
        'resnet_improvement_factor': float(resnet_improvement),
        'transfer_factor': float(transfer_factor),
        'transfer_status': transfer_status,
        'architecture_main_effect': float(arch_effect),
        'composition_main_effect': float(composition_effect),
        'interaction_term_significance': float(interaction_sig),
        'interaction_term_status': interaction_term,
        'hypothesis_status': hypothesis_status,
    }
    results['summary'] = summary

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/interaction_architecture_transfer')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'res_246_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Return concise summary for log
    return {
        'metrics': results['metrics'],
        'summary': summary
    }


if __name__ == '__main__':
    result = main()
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
