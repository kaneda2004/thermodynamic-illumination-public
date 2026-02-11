#!/usr/bin/env python3
"""
RES-060: Does CPPN weight sparsity correlate with output order?

Hypothesis: CPPNs with sparser weights (more near-zero weights) correlate
with different output image order.

Method:
1. Generate many random CPPNs with varying architectures
2. Measure weight sparsity at multiple thresholds
3. Measure output image order (using multiplicative metric)
4. Compute Spearman correlation and effect size
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, Node, Connection, ACTIVATIONS, PRIOR_SIGMA

def compute_weight_sparsity(cppn: CPPN, threshold: float = 0.1) -> float:
    """Fraction of weights with |w| < threshold."""
    weights = cppn.get_weights()
    if len(weights) == 0:
        return 1.0
    return np.mean(np.abs(weights) < threshold)

def compute_weight_l1_norm(cppn: CPPN) -> float:
    """L1 norm of weights (alternative sparsity measure)."""
    weights = cppn.get_weights()
    if len(weights) == 0:
        return 0.0
    return np.mean(np.abs(weights))

def create_random_cppn(n_hidden: int = 0) -> CPPN:
    """Create CPPN with random architecture."""
    cppn = CPPN()  # Start with default structure

    # Add hidden nodes
    next_id = 5
    hidden_ids = []
    activations = list(ACTIVATIONS.keys())

    for _ in range(n_hidden):
        act = np.random.choice(activations)
        bias = np.random.randn() * PRIOR_SIGMA
        cppn.nodes.append(Node(next_id, act, bias))
        hidden_ids.append(next_id)
        next_id += 1

    # Add connections to/from hidden nodes
    for hid in hidden_ids:
        # Input -> hidden
        for inp in cppn.input_ids:
            if np.random.random() > 0.5:
                cppn.connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))
        # Hidden -> output
        cppn.connections.append(Connection(hid, cppn.output_id, np.random.randn() * PRIOR_SIGMA))

    return cppn

def run_experiment(n_samples: int = 1000, seed: int = 42):
    """Run the weight sparsity vs order correlation experiment."""
    np.random.seed(seed)

    # Multiple sparsity measures
    sparsity_01 = []  # threshold=0.1
    sparsity_05 = []  # threshold=0.5
    l1_norms = []
    orders = []
    n_weights = []

    print(f"Generating {n_samples} random CPPNs...")

    for i in range(n_samples):
        # Vary architecture complexity
        n_hidden = np.random.randint(0, 8)

        cppn = create_random_cppn(n_hidden=n_hidden)

        # Vary weight scales to induce sparsity variation
        weights = cppn.get_weights()
        if len(weights) > 0:
            scale = np.random.uniform(0.05, 5.0)  # Wide range
            cppn.set_weights(weights * scale)

        s01 = compute_weight_sparsity(cppn, threshold=0.1)
        s05 = compute_weight_sparsity(cppn, threshold=0.5)
        l1 = compute_weight_l1_norm(cppn)
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        sparsity_01.append(s01)
        sparsity_05.append(s05)
        l1_norms.append(l1)
        orders.append(order)
        n_weights.append(len(cppn.get_weights()))

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

    # Convert to arrays
    sparsity_01 = np.array(sparsity_01)
    sparsity_05 = np.array(sparsity_05)
    l1_norms = np.array(l1_norms)
    orders = np.array(orders)
    n_weights = np.array(n_weights)

    # Remove NaN/Inf values
    valid = np.isfinite(sparsity_01) & np.isfinite(orders) & np.isfinite(l1_norms)
    sparsity_01 = sparsity_01[valid]
    sparsity_05 = sparsity_05[valid]
    l1_norms = l1_norms[valid]
    orders = orders[valid]
    n_weights = n_weights[valid]

    print(f"\nValid samples: {len(orders)}")
    print(f"Sparsity (t=0.1) range: [{sparsity_01.min():.3f}, {sparsity_01.max():.3f}]")
    print(f"Sparsity (t=0.5) range: [{sparsity_05.min():.3f}, {sparsity_05.max():.3f}]")
    print(f"L1 norm range: [{l1_norms.min():.3f}, {l1_norms.max():.3f}]")
    print(f"Order range: [{orders.min():.3f}, {orders.max():.3f}]")

    print(f"\n=== CORRELATION ANALYSIS ===")

    results = {}

    # Test multiple sparsity measures
    for name, sparsity in [('sparsity_t01', sparsity_01),
                           ('sparsity_t05', sparsity_05),
                           ('l1_norm', l1_norms)]:
        rho, p_value = stats.spearmanr(sparsity, orders)
        r_pearson, p_pearson = stats.pearsonr(sparsity, orders)
        effect_size = abs(np.arctanh(rho)) if abs(rho) < 0.999 else abs(rho)

        print(f"\n{name}:")
        print(f"  Spearman rho: {rho:.4f} (p={p_value:.2e})")
        print(f"  Pearson r: {r_pearson:.4f} (p={p_pearson:.2e})")
        print(f"  Effect size: {effect_size:.4f}")

        results[name] = {
            'spearman_rho': float(rho),
            'spearman_p': float(p_value),
            'effect_size': float(effect_size)
        }

    # Also check if network size confounds
    rho_size, p_size = stats.spearmanr(n_weights, orders)
    print(f"\nConfound check - n_weights vs order:")
    print(f"  Spearman rho: {rho_size:.4f} (p={p_size:.2e})")

    # Primary result: sparsity at threshold 0.1
    rho = results['sparsity_t01']['spearman_rho']
    p_value = results['sparsity_t01']['spearman_p']
    effect_size = results['sparsity_t01']['effect_size']

    # Determine status with Bonferroni correction (3 tests)
    p_bonferroni = p_value * 3

    if p_bonferroni < 0.01 and effect_size > 0.1:
        status = "VALIDATED" if rho > 0 else "VALIDATED_NEGATIVE"
        conclusion = f"Significant correlation: rho={rho:.3f}, p={p_bonferroni:.2e}"
    elif p_bonferroni < 0.05 and effect_size > 0.05:
        status = "INCONCLUSIVE"
        conclusion = f"Weak correlation: rho={rho:.3f}, p={p_bonferroni:.2e}"
    else:
        status = "REFUTED"
        conclusion = f"No significant correlation (rho={rho:.3f}, p={p_bonferroni:.2e})"

    print(f"\n{'='*50}")
    print(f"STATUS: {status}")
    print(f"CONCLUSION: {conclusion}")

    return {
        'n_samples': len(orders),
        'primary_spearman_rho': float(rho),
        'primary_p_value': float(p_value),
        'primary_p_bonferroni': float(p_bonferroni),
        'primary_effect_size': float(effect_size),
        'l1_spearman_rho': float(results['l1_norm']['spearman_rho']),
        'l1_p_value': float(results['l1_norm']['spearman_p']),
        'status': status,
        'conclusion': conclusion
    }

if __name__ == '__main__':
    results = run_experiment(n_samples=1000, seed=42)
    print("\n" + "="*50)
    print("Final Results Dict:")
    for k, v in results.items():
        print(f"  {k}: {v}")
