"""
RES-048: Activation Variance vs Output Order

Hypothesis: The variance of internal node activations in CPPNs correlates
negatively with output image order - high-order images emerge from CPPNs
where intermediate computations have low variance (saturated activations),
while low-order images have high activation variance.

Domain: activation_statistics
"""

import numpy as np
import json
import os
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, ACTIVATIONS, order_multiplicative as compute_order, Node, Connection


@dataclass
class ActivationStats:
    """Statistics for activations at a single node."""
    node_id: int
    activation_func: str
    mean: float
    variance: float
    min_val: float
    max_val: float
    saturation_ratio: float  # fraction of values near 0 or 1


def compute_activation_statistics(cppn: CPPN, size: int = 32) -> Dict[int, ActivationStats]:
    """
    Compute activation statistics for all nodes in a CPPN.

    Returns dict mapping node_id -> ActivationStats
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    stats_dict = {}

    # Get evaluation order (hidden nodes then output)
    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)

        # Compute pre-activation sum
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        # Apply activation
        activated = ACTIVATIONS[node.activation](total)
        values[nid] = activated

        # Compute statistics
        flat = activated.flatten()
        var = np.var(flat)

        # Saturation: values near 0 or 1 (for bounded activations)
        saturation = np.mean((flat < 0.1) | (flat > 0.9))

        stats_dict[nid] = ActivationStats(
            node_id=nid,
            activation_func=node.activation,
            mean=float(np.mean(flat)),
            variance=float(var),
            min_val=float(np.min(flat)),
            max_val=float(np.max(flat)),
            saturation_ratio=float(saturation)
        )

    return stats_dict


def sample_cppns_with_varying_order(n_samples: int = 500, size: int = 32, seed: int = 42) -> List[Tuple[CPPN, float, Dict]]:
    """
    Sample CPPNs and compute their order and activation statistics.
    Returns list of (cppn, order, activation_stats).
    """
    np.random.seed(seed)
    results = []

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization from prior
        img = cppn.render(size)
        order = compute_order(img)
        act_stats = compute_activation_statistics(cppn, size)
        results.append((cppn, order, act_stats))

    return results


def run_experiment():
    """Main experiment: test correlation between activation variance and order."""
    print("=" * 60)
    print("RES-048: Activation Variance vs Output Order")
    print("=" * 60)

    # Parameters
    n_samples = 500
    size = 32
    seed = 42

    print(f"\nSampling {n_samples} CPPNs...")
    samples = sample_cppns_with_varying_order(n_samples, size, seed)

    # Extract data
    orders = np.array([s[1] for s in samples])

    # For basic CPPN, output node is id=4
    output_variances = np.array([s[2][4].variance for s in samples])
    output_saturations = np.array([s[2][4].saturation_ratio for s in samples])

    print(f"\nOrder statistics: mean={orders.mean():.3f}, std={orders.std():.3f}")
    print(f"  min={orders.min():.3f}, max={orders.max():.3f}")
    print(f"\nOutput variance statistics: mean={output_variances.mean():.4f}, std={output_variances.std():.4f}")
    print(f"Output saturation statistics: mean={output_saturations.mean():.3f}, std={output_saturations.std():.3f}")

    # Test 1: Correlation between output variance and order
    corr_var, p_var = stats.pearsonr(output_variances, orders)
    spearman_var, sp_p_var = stats.spearmanr(output_variances, orders)

    print("\n--- Correlation: Output Variance vs Order ---")
    print(f"Pearson r = {corr_var:.4f}, p = {p_var:.2e}")
    print(f"Spearman rho = {spearman_var:.4f}, p = {sp_p_var:.2e}")

    # Test 2: Correlation between saturation ratio and order
    corr_sat, p_sat = stats.pearsonr(output_saturations, orders)
    spearman_sat, sp_p_sat = stats.spearmanr(output_saturations, orders)

    print("\n--- Correlation: Saturation Ratio vs Order ---")
    print(f"Pearson r = {corr_sat:.4f}, p = {p_sat:.2e}")
    print(f"Spearman rho = {spearman_sat:.4f}, p = {sp_p_sat:.2e}")

    # Test 3: Compare high-order vs low-order groups
    median_order = np.median(orders)
    high_order_mask = orders > median_order
    low_order_mask = orders <= median_order

    high_var = output_variances[high_order_mask]
    low_var = output_variances[low_order_mask]

    # T-test between groups
    t_stat, t_p = stats.ttest_ind(high_var, low_var)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_var.std()**2 + low_var.std()**2) / 2)
    cohens_d = (high_var.mean() - low_var.mean()) / pooled_std if pooled_std > 0 else 0

    print("\n--- High vs Low Order Groups ---")
    print(f"High-order variance: mean={high_var.mean():.4f}, std={high_var.std():.4f}")
    print(f"Low-order variance: mean={low_var.mean():.4f}, std={low_var.std():.4f}")
    print(f"T-test: t={t_stat:.3f}, p={t_p:.2e}")
    print(f"Cohen's d: {cohens_d:.3f}")

    # Test 4: Also look at pre-activation statistics
    # For basic CPPN, we can analyze the weighted sum before activation
    print("\n--- Additional Analysis: Pre-activation Statistics ---")

    # Recompute with pre-activation tracking
    pre_act_vars = []
    for cppn, order, _ in samples:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        bias_input = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias_input}

        # Compute pre-activation for output node
        output_node = next(n for n in cppn.nodes if n.id == cppn.output_id)
        total = np.zeros_like(x) + output_node.bias
        for conn in cppn.connections:
            if conn.to_id == cppn.output_id and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        pre_act_vars.append(np.var(total.flatten()))

    pre_act_vars = np.array(pre_act_vars)
    corr_pre, p_pre = stats.pearsonr(pre_act_vars, orders)

    print(f"Pre-activation variance vs Order: r={corr_pre:.4f}, p={p_pre:.2e}")

    # Determine result status
    # Key prediction: variance should NEGATIVELY correlate with order
    # (hypothesis predicts low variance -> high order)

    is_significant = p_var < 0.01  # Strict threshold
    has_large_effect = abs(cohens_d) > 0.5
    direction_matches = corr_var < 0  # Negative correlation as hypothesized

    if is_significant and direction_matches and has_large_effect:
        status = "validated"
        conclusion = "Strong negative correlation between activation variance and order"
    elif is_significant and not direction_matches:
        status = "refuted"
        conclusion = f"Significant correlation but POSITIVE (r={corr_var:.3f}), opposite to hypothesis"
    elif is_significant:
        status = "validated"
        conclusion = f"Significant correlation (r={corr_var:.3f}) with effect size d={cohens_d:.3f}"
    else:
        status = "inconclusive"
        conclusion = f"No significant correlation found (r={corr_var:.3f}, p={p_var:.3f})"

    print("\n" + "=" * 60)
    print(f"STATUS: {status.upper()}")
    print(f"CONCLUSION: {conclusion}")
    print("=" * 60)

    # Save results
    results = {
        "experiment_id": "RES-048",
        "hypothesis": "Activation variance negatively correlates with output order",
        "n_samples": n_samples,
        "image_size": size,
        "metrics": {
            "pearson_r_var_order": float(corr_var),
            "pearson_p_var_order": float(p_var),
            "spearman_rho_var_order": float(spearman_var),
            "spearman_p_var_order": float(sp_p_var),
            "pearson_r_sat_order": float(corr_sat),
            "pearson_p_sat_order": float(p_sat),
            "cohens_d": float(cohens_d),
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "pre_activation_corr": float(corr_pre),
            "pre_activation_p": float(p_pre),
        },
        "group_stats": {
            "high_order_variance_mean": float(high_var.mean()),
            "high_order_variance_std": float(high_var.std()),
            "low_order_variance_mean": float(low_var.mean()),
            "low_order_variance_std": float(low_var.std()),
        },
        "status": status,
        "conclusion": conclusion
    }

    # Save results
    os.makedirs("results/activation_statistics", exist_ok=True)
    with open("results/activation_statistics/res048_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/activation_statistics/res048_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
