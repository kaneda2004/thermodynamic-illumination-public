"""
RES-103: Activation Saturation Pattern vs Output Order

Hypothesis: Layer-wise saturation order (early vs late layers saturating first)
predicts CPPN output order - high-order CPPNs have earlier saturation in
input-proximal layers while low-order CPPNs show uniform or late saturation.

Domain: activation_saturation_pattern

Key metric: "saturation_depth_index" - weighted average of layer indices where
saturation occurs, normalized by total depth. Lower values = earlier saturation.
"""

import numpy as np
import json
import os
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, ACTIVATIONS, order_multiplicative, Node, Connection


@dataclass
class LayerSaturationProfile:
    """Saturation statistics for a single layer."""
    layer_idx: int
    node_ids: List[int]
    mean_saturation: float  # Average saturation ratio across nodes
    max_saturation: float  # Max saturation in this layer
    activations_near_bounds: float  # Fraction of activations near 0 or 1


def compute_saturation_ratio(values: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute fraction of values that are "saturated" (near 0 or 1).
    For sigmoid-like outputs, saturation means near 0 or 1.
    For unbounded activations, we use relative saturation (near min/max).
    """
    flat = values.flatten()
    vmin, vmax = np.min(flat), np.max(flat)
    vrange = vmax - vmin

    if vrange < 1e-6:
        # Constant output = fully saturated
        return 1.0

    # For bounded activations like sigmoid (0-1), check absolute bounds
    # For unbounded, check relative to range
    if vmin >= -0.1 and vmax <= 1.1:
        # Bounded activation - check absolute saturation
        near_zero = np.mean(flat < threshold)
        near_one = np.mean(flat > (1 - threshold))
        return near_zero + near_one
    else:
        # Unbounded activation - check relative saturation
        near_min = np.mean(flat < vmin + threshold * vrange)
        near_max = np.mean(flat > vmax - threshold * vrange)
        return near_min + near_max


def create_deep_cppn(num_hidden_layers: int = 4, nodes_per_layer: int = 3) -> Tuple[CPPN, List[List[int]]]:
    """
    Create a CPPN with explicit layer structure.
    Returns (cppn, layer_node_ids) where layer_node_ids[i] = node IDs in layer i.
    Layer 0 = inputs (not tracked), Layer 1...n = hidden layers, Layer n+1 = output.
    """
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    input_ids = [0, 1, 2, 3]
    next_id = 5  # Output is 4

    layer_node_ids = []  # Will store node IDs for each computational layer
    prev_layer_ids = input_ids

    for layer_idx in range(num_hidden_layers):
        current_layer_ids = []
        for _ in range(nodes_per_layer):
            activation = np.random.choice(['sin', 'tanh', 'gauss', 'sigmoid'])
            nodes.append(Node(next_id, activation, np.random.randn() * 1.0))
            current_layer_ids.append(next_id)
            # Connect from previous layer
            for prev_id in prev_layer_ids:
                connections.append(Connection(prev_id, next_id, np.random.randn() * 1.0))
            next_id += 1
        layer_node_ids.append(current_layer_ids)
        prev_layer_ids = current_layer_ids

    # Output node
    nodes.append(Node(4, 'sigmoid', np.random.randn() * 1.0))
    for prev_id in prev_layer_ids:
        connections.append(Connection(prev_id, 4, np.random.randn() * 1.0))
    layer_node_ids.append([4])  # Output layer

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=4)
    return cppn, layer_node_ids


def compute_layer_saturations(cppn: CPPN, layer_node_ids: List[List[int]], size: int = 32) -> List[LayerSaturationProfile]:
    """
    Compute saturation profile for each layer in the CPPN.
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    # Store activations for all nodes
    values = {0: x, 1: y, 2: r, 3: bias}

    # Forward pass through all nodes in evaluation order
    eval_order = cppn._get_eval_order()
    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[nid] = ACTIVATIONS[node.activation](total)

    # Compute saturation for each layer
    profiles = []
    for layer_idx, node_ids in enumerate(layer_node_ids):
        saturations = []
        for nid in node_ids:
            if nid in values:
                sat = compute_saturation_ratio(values[nid])
                saturations.append(sat)

        if saturations:
            profiles.append(LayerSaturationProfile(
                layer_idx=layer_idx,
                node_ids=node_ids,
                mean_saturation=float(np.mean(saturations)),
                max_saturation=float(np.max(saturations)),
                activations_near_bounds=float(np.mean(saturations))  # Same for now
            ))

    return profiles


def compute_saturation_depth_index(profiles: List[LayerSaturationProfile], saturation_threshold: float = 0.3) -> float:
    """
    Compute a single metric: weighted average layer index where saturation exceeds threshold.

    Lower values = saturation happens early (input-proximal layers)
    Higher values = saturation happens late (output-proximal layers)

    If no layer exceeds threshold, return the layer with highest saturation.
    """
    n_layers = len(profiles)
    if n_layers == 0:
        return 0.5

    # Find layers that exceed threshold
    saturated_layers = [(p.layer_idx, p.mean_saturation)
                        for p in profiles if p.mean_saturation > saturation_threshold]

    if saturated_layers:
        # Weighted average of saturated layer indices (weight = saturation amount)
        total_weight = sum(s for _, s in saturated_layers)
        weighted_idx = sum(idx * sat for idx, sat in saturated_layers) / total_weight
        return weighted_idx / (n_layers - 1) if n_layers > 1 else 0.5
    else:
        # Use layer with max saturation
        max_layer = max(profiles, key=lambda p: p.mean_saturation)
        return max_layer.layer_idx / (n_layers - 1) if n_layers > 1 else 0.5


def compute_early_late_ratio(profiles: List[LayerSaturationProfile]) -> float:
    """
    Ratio of early-layer saturation to late-layer saturation.
    Higher ratio = more early saturation.
    """
    n_layers = len(profiles)
    if n_layers < 2:
        return 1.0

    mid = n_layers // 2
    early_sat = np.mean([p.mean_saturation for p in profiles[:mid]])
    late_sat = np.mean([p.mean_saturation for p in profiles[mid:]])

    return early_sat / (late_sat + 1e-6)


def run_experiment(n_samples: int = 500, n_hidden_layers: int = 4, nodes_per_layer: int = 3, seed: int = 42):
    """Main experiment."""
    print("=" * 70)
    print("RES-103: Activation Saturation Pattern vs Output Order")
    print("=" * 70)

    np.random.seed(seed)

    print(f"\nSampling {n_samples} CPPNs with {n_hidden_layers} hidden layers...")

    # Collect data
    orders = []
    saturation_depth_indices = []
    early_late_ratios = []
    total_saturations = []

    valid_count = 0
    attempts = 0
    max_attempts = n_samples * 5

    while valid_count < n_samples and attempts < max_attempts:
        attempts += 1

        cppn, layer_node_ids = create_deep_cppn(n_hidden_layers, nodes_per_layer)
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute saturation profiles
        profiles = compute_layer_saturations(cppn, layer_node_ids)

        if not profiles:
            continue

        # Compute metrics
        depth_idx = compute_saturation_depth_index(profiles)
        el_ratio = compute_early_late_ratio(profiles)
        total_sat = np.mean([p.mean_saturation for p in profiles])

        orders.append(order)
        saturation_depth_indices.append(depth_idx)
        early_late_ratios.append(el_ratio)
        total_saturations.append(total_sat)

        valid_count += 1
        if valid_count % 100 == 0:
            print(f"  Processed {valid_count}/{n_samples}...")

    print(f"\nCollected {valid_count} valid samples")

    orders = np.array(orders)
    saturation_depth_indices = np.array(saturation_depth_indices)
    early_late_ratios = np.array(early_late_ratios)
    total_saturations = np.array(total_saturations)

    # Statistics
    print(f"\nOrder: mean={orders.mean():.3f}, std={orders.std():.3f}")
    print(f"Saturation Depth Index: mean={saturation_depth_indices.mean():.3f}, std={saturation_depth_indices.std():.3f}")
    print(f"Early/Late Ratio: mean={early_late_ratios.mean():.3f}, std={early_late_ratios.std():.3f}")

    # Test 1: Correlation between saturation depth index and order
    # Hypothesis: NEGATIVE correlation (early saturation -> high order)
    corr_depth, p_depth = stats.pearsonr(saturation_depth_indices, orders)
    spearman_depth, sp_p_depth = stats.spearmanr(saturation_depth_indices, orders)

    print("\n--- Saturation Depth Index vs Order ---")
    print(f"Pearson r = {corr_depth:.4f}, p = {p_depth:.2e}")
    print(f"Spearman rho = {spearman_depth:.4f}, p = {sp_p_depth:.2e}")

    # Test 2: Early/Late ratio vs order
    corr_ratio, p_ratio = stats.pearsonr(early_late_ratios, orders)
    spearman_ratio, sp_p_ratio = stats.spearmanr(early_late_ratios, orders)

    print("\n--- Early/Late Saturation Ratio vs Order ---")
    print(f"Pearson r = {corr_ratio:.4f}, p = {p_ratio:.2e}")
    print(f"Spearman rho = {spearman_ratio:.4f}, p = {sp_p_ratio:.2e}")

    # Test 3: Compare high-order vs low-order groups
    median_order = np.median(orders)
    high_order_mask = orders > median_order
    low_order_mask = orders <= median_order

    high_depth = saturation_depth_indices[high_order_mask]
    low_depth = saturation_depth_indices[low_order_mask]

    high_ratio = early_late_ratios[high_order_mask]
    low_ratio = early_late_ratios[low_order_mask]

    # T-tests
    t_depth, p_t_depth = stats.ttest_ind(high_depth, low_depth)
    t_ratio, p_t_ratio = stats.ttest_ind(high_ratio, low_ratio)

    # Effect sizes (Cohen's d)
    def cohens_d(g1, g2):
        pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
        return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

    d_depth = cohens_d(high_depth, low_depth)
    d_ratio = cohens_d(high_ratio, low_ratio)

    print("\n--- High vs Low Order Groups ---")
    print(f"Depth Index: high={high_depth.mean():.3f}, low={low_depth.mean():.3f}, d={d_depth:.3f}, p={p_t_depth:.2e}")
    print(f"Early/Late Ratio: high={high_ratio.mean():.3f}, low={low_ratio.mean():.3f}, d={d_ratio:.3f}, p={p_t_ratio:.2e}")

    # Test 4: Control for total saturation
    # Does the PATTERN matter beyond just having saturation?
    partial_corr_numerator = corr_depth - stats.pearsonr(saturation_depth_indices, total_saturations)[0] * stats.pearsonr(total_saturations, orders)[0]
    partial_corr_denom = np.sqrt((1 - stats.pearsonr(saturation_depth_indices, total_saturations)[0]**2) *
                                  (1 - stats.pearsonr(total_saturations, orders)[0]**2))
    partial_corr = partial_corr_numerator / partial_corr_denom if partial_corr_denom > 0 else 0

    print(f"\n--- Partial Correlation (controlling for total saturation) ---")
    print(f"Partial r (depth vs order | total_sat): {partial_corr:.4f}")

    # Determine status
    # Key predictions:
    # 1. Depth index should NEGATIVELY correlate with order (early sat -> high order)
    # 2. Early/Late ratio should POSITIVELY correlate with order

    depth_significant = abs(p_depth) < 0.01
    depth_large_effect = abs(d_depth) > 0.5
    depth_correct_direction = corr_depth < 0  # Negative = early saturation -> high order

    ratio_significant = abs(p_ratio) < 0.01
    ratio_large_effect = abs(d_ratio) > 0.5
    ratio_correct_direction = corr_ratio > 0  # Positive = high ratio -> high order

    if (depth_significant and depth_large_effect and depth_correct_direction) or \
       (ratio_significant and ratio_large_effect and ratio_correct_direction):
        status = "validated"
        if depth_correct_direction and ratio_correct_direction:
            conclusion = f"Early saturation predicts high order: depth_idx r={corr_depth:.3f}, ratio r={corr_ratio:.3f}"
        elif depth_correct_direction:
            conclusion = f"Depth index predicts order (r={corr_depth:.3f}, d={d_depth:.3f})"
        else:
            conclusion = f"Early/late ratio predicts order (r={corr_ratio:.3f}, d={d_ratio:.3f})"
    elif (depth_significant and not depth_correct_direction) or (ratio_significant and not ratio_correct_direction):
        status = "refuted"
        conclusion = f"Pattern opposite to hypothesis: depth_idx r={corr_depth:.3f}, ratio r={corr_ratio:.3f}"
    elif depth_significant or ratio_significant:
        status = "inconclusive"
        conclusion = f"Weak effect: depth d={d_depth:.3f}, ratio d={d_ratio:.3f}"
    else:
        status = "refuted"
        conclusion = f"No significant correlation: depth p={p_depth:.3f}, ratio p={p_ratio:.3f}"

    print("\n" + "=" * 70)
    print(f"STATUS: {status.upper()}")
    print(f"CONCLUSION: {conclusion}")
    print("=" * 70)

    # Save results
    results = {
        "experiment_id": "RES-103",
        "domain": "activation_saturation_pattern",
        "hypothesis": "Early saturation (input-proximal layers) predicts high output order",
        "n_samples": valid_count,
        "n_hidden_layers": n_hidden_layers,
        "nodes_per_layer": nodes_per_layer,
        "metrics": {
            "pearson_r_depth_order": float(corr_depth),
            "pearson_p_depth_order": float(p_depth),
            "spearman_rho_depth_order": float(spearman_depth),
            "pearson_r_ratio_order": float(corr_ratio),
            "pearson_p_ratio_order": float(p_ratio),
            "spearman_rho_ratio_order": float(spearman_ratio),
            "cohens_d_depth": float(d_depth),
            "cohens_d_ratio": float(d_ratio),
            "partial_corr_depth": float(partial_corr),
        },
        "group_stats": {
            "high_order_depth_mean": float(high_depth.mean()),
            "low_order_depth_mean": float(low_depth.mean()),
            "high_order_ratio_mean": float(high_ratio.mean()),
            "low_order_ratio_mean": float(low_ratio.mean()),
        },
        "status": status,
        "conclusion": conclusion
    }

    os.makedirs("results/activation_saturation", exist_ok=True)
    with open("results/activation_saturation/res103_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/activation_saturation/res103_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(n_samples=500, n_hidden_layers=4, nodes_per_layer=3, seed=42)
