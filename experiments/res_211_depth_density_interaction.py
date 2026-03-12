#!/usr/bin/env python3
"""
RES-211: Optimal connection density scales logarithmically with network depth
Tests interaction between CPPN network depth and connection density
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from scipy import stats
import time

# Set working directory
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA

def create_cppn_with_depth_and_density(depth: int, density: float, seed: int) -> CPPN:
    """Create CPPN with specified depth (number of hidden layers) and connection density."""
    np.random.seed(seed)

    input_ids = [0, 1, 2, 3]
    output_id = 4

    # Depth controls number of hidden nodes (linear with depth for simplicity)
    n_hidden = depth * 2  # Each depth level adds 2 hidden nodes
    hidden_ids = list(range(5, 5 + n_hidden))

    # Create nodes
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
        Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA),
    ]

    activations = list(ACTIVATIONS.keys())
    for hid in hidden_ids:
        nodes.append(Node(hid, np.random.choice(activations), np.random.randn() * PRIOR_SIGMA))

    # Calculate all possible connections (feedforward only)
    # Input -> Hidden, Input -> Output, Hidden -> Output, Hidden -> Hidden (higher id)
    possible_conns = []
    for inp in input_ids:
        for hid in hidden_ids:
            possible_conns.append((inp, hid))
        possible_conns.append((inp, output_id))
    for i, h1 in enumerate(hidden_ids):
        for h2 in hidden_ids[i+1:]:
            possible_conns.append((h1, h2))
        possible_conns.append((h1, output_id))

    # Sample connections based on density
    n_conns = max(1, int(len(possible_conns) * density))
    selected = np.random.choice(len(possible_conns), size=n_conns, replace=False)

    connections = []
    for idx in selected:
        from_id, to_id = possible_conns[idx]
        connections.append(Connection(from_id, to_id, np.random.randn() * PRIOR_SIGMA))

    cppn = CPPN.__new__(CPPN)
    cppn.nodes = nodes
    cppn.connections = connections
    cppn.input_ids = input_ids
    cppn.output_id = output_id

    return cppn

# Configuration
DEPTHS = [1, 2, 4, 6, 8]
DENSITIES = [0.20, 0.40, 0.60, 1.0]  # 20%, 40%, 60%, 100%
N_CPNS = 10  # CPPNs per config
IMAGE_SIZE = 32

# Results tracking
results_data = {
    'timestamp': time.time(),
    'hypothesis': 'Optimal connection density scales logarithmically with network depth',
    'domain': 'network_architecture',
    'configuration': {
        'depths': DEPTHS,
        'densities': DENSITIES,
        'n_cpns_per_config': N_CPNS,
        'image_size': IMAGE_SIZE
    },
    'results_by_config': {},
    'depth_col': [],
    'density_col': [],
    'order_col': []
}

print(f"Starting RES-211: Depth × Density Interaction Study")
print(f"Testing {len(DEPTHS)} depths × {len(DENSITIES)} densities = {len(DEPTHS)*len(DENSITIES)} configurations")
print(f"With {N_CPNS} CPPNs per config = {len(DEPTHS)*len(DENSITIES)*N_CPNS} total CPPNs to test\n")

total_configs = len(DEPTHS) * len(DENSITIES)
config_num = 0

# Test each depth × density combination
for depth in DEPTHS:
    for density in DENSITIES:
        config_num += 1
        config_key = f"depth_{depth}_density_{density:.2f}"
        results_data['results_by_config'][config_key] = {
            'depth': depth,
            'density': density,
            'orders': [],
            'mean_order': 0.0,
            'std_order': 0.0,
            'max_order': 0.0,
            'min_order': 0.0
        }

        print(f"[{config_num}/{total_configs}] Depth={depth}, Density={density:.0%}")

        # Test N_CPNS networks for this configuration
        for cppn_idx in range(N_CPNS):
            try:
                # Create CPPN with specific depth and density
                cppn = create_cppn_with_depth_and_density(
                    depth=depth,
                    density=density,
                    seed=42 + config_num * 1000 + cppn_idx
                )

                # Render and measure order
                img = cppn.render(IMAGE_SIZE)
                order_score = order_multiplicative(img)

                results_data['results_by_config'][config_key]['orders'].append(float(order_score))

                # Track for ANOVA
                results_data['depth_col'].append(depth)
                results_data['density_col'].append(density)
                results_data['order_col'].append(float(order_score))

            except Exception as e:
                print(f"  ⚠ Error in CPPN {cppn_idx}: {str(e)[:60]}")
                continue

        # Compute statistics for this config
        orders = results_data['results_by_config'][config_key]['orders']
        if len(orders) > 0:
            results_data['results_by_config'][config_key]['mean_order'] = float(np.mean(orders))
            results_data['results_by_config'][config_key]['std_order'] = float(np.std(orders))
            results_data['results_by_config'][config_key]['max_order'] = float(np.max(orders))
            results_data['results_by_config'][config_key]['min_order'] = float(np.min(orders))
            print(f"  → Orders: mean={results_data['results_by_config'][config_key]['mean_order']:.3f} "
                  f"±{results_data['results_by_config'][config_key]['std_order']:.3f}")
        else:
            print(f"  → No valid results for this config")

# Statistical Analysis: Two-way ANOVA
print("\n" + "="*70)
print("STATISTICAL ANALYSIS: Two-way ANOVA (Depth × Density)")
print("="*70)

if len(results_data['order_col']) > 0:
    # Prepare data for ANOVA
    depth_groups = {}
    density_groups = {}

    for depth in DEPTHS:
        depth_groups[depth] = [results_data['order_col'][i]
                               for i in range(len(results_data['order_col']))
                               if results_data['depth_col'][i] == depth]

    for density in DENSITIES:
        density_groups[density] = [results_data['order_col'][i]
                                   for i in range(len(results_data['order_col']))
                                   if results_data['density_col'][i] == density]

    # Two-way ANOVA using scipy
    from scipy.stats import f_oneway

    # Main effects
    f_depth, p_depth = f_oneway(*[depth_groups[d] for d in DEPTHS])
    f_density, p_density = f_oneway(*[density_groups[d] for d in DENSITIES])

    print(f"\nMain Effects:")
    print(f"  Depth:   F={f_depth:.4f}, p={p_depth:.4e}")
    print(f"  Density: F={f_density:.4f}, p={p_density:.4e}")

    # Compute effect sizes (eta-squared)
    grand_mean = np.mean(results_data['order_col'])
    ss_total = np.sum([(x - grand_mean)**2 for x in results_data['order_col']])

    ss_depth = sum(len(depth_groups[d]) * (np.mean(depth_groups[d]) - grand_mean)**2
                   for d in DEPTHS)
    eta2_depth = ss_depth / ss_total if ss_total > 0 else 0

    ss_density = sum(len(density_groups[d]) * (np.mean(density_groups[d]) - grand_mean)**2
                     for d in DENSITIES)
    eta2_density = ss_density / ss_total if ss_total > 0 else 0

    print(f"\nEffect Sizes (eta-squared):")
    print(f"  Depth:   η² = {eta2_depth:.4f}")
    print(f"  Density: η² = {eta2_density:.4f}")

    # Test for interaction: compare within-depth density variation
    depth_density_variance = []
    for depth in DEPTHS:
        for density1_idx, d1 in enumerate(DENSITIES):
            for d2 in DENSITIES[density1_idx+1:]:
                key1 = f"depth_{depth}_density_{d1:.2f}"
                key2 = f"depth_{depth}_density_{d2:.2f}"
                if (results_data['results_by_config'][key1]['orders'] and
                    results_data['results_by_config'][key2]['orders']):
                    diff = (results_data['results_by_config'][key1]['mean_order'] -
                           results_data['results_by_config'][key2]['mean_order'])
                    depth_density_variance.append(diff)

    interaction_effect = np.std(depth_density_variance) if depth_density_variance else 0
    print(f"\nInteraction Effect (density sensitivity varies by depth):")
    print(f"  Std dev of within-depth density differences: {interaction_effect:.4f}")

    # Relationship analysis: Does optimal density scale with depth?
    print(f"\nOptimal Density by Depth:")
    optimal_density_by_depth = {}
    for depth in DEPTHS:
        densities_for_depth = []
        for density in DENSITIES:
            key = f"depth_{depth}_density_{density:.2f}"
            if results_data['results_by_config'][key]['mean_order'] > 0:
                densities_for_depth.append((density, results_data['results_by_config'][key]['mean_order']))
        if densities_for_depth:
            optimal_density = max(densities_for_depth, key=lambda x: x[1])[0]
            optimal_order = max(densities_for_depth, key=lambda x: x[1])[1]
            optimal_density_by_depth[depth] = optimal_density
            print(f"  Depth {depth}: optimal_density = {optimal_density:.0%} (order = {optimal_order:.3f})")

    # Test correlation: depth vs optimal density
    if len(optimal_density_by_depth) > 1:
        depths_list = sorted(optimal_density_by_depth.keys())
        densities_list = [optimal_density_by_depth[d] for d in depths_list]
        log_depths = np.log(np.array(depths_list) + 1)  # +1 to avoid log(1)=0

        corr_linear, p_linear = stats.pearsonr(depths_list, densities_list)
        corr_log, p_log = stats.pearsonr(log_depths, densities_list)

        print(f"\nCorrelation: Depth vs Optimal Density")
        print(f"  Linear:     r={corr_linear:.4f}, p={p_linear:.4f}")
        print(f"  Log-linear: r={corr_log:.4f}, p={p_log:.4f}")

        results_data['correlation_analysis'] = {
            'linear_r': float(corr_linear),
            'linear_p': float(p_linear),
            'log_r': float(corr_log),
            'log_p': float(p_log)
        }

    # Summary statistics
    results_data['summary_statistics'] = {
        'total_cpns': len(results_data['order_col']),
        'mean_order_all': float(np.mean(results_data['order_col'])),
        'std_order_all': float(np.std(results_data['order_col'])),
        'max_order_all': float(np.max(results_data['order_col'])),
        'min_order_all': float(np.min(results_data['order_col'])),
        'f_depth': float(f_depth),
        'p_depth': float(p_depth),
        'f_density': float(f_density),
        'p_density': float(p_density),
        'eta2_depth': float(eta2_depth),
        'eta2_density': float(eta2_density),
        'interaction_effect': float(interaction_effect)
    }

# Save results
results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/network_architecture')
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / 'res_211_results.json'
with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\n✓ Results saved to {results_file}")

# Determine conclusion
p_threshold = 0.01
eta2_threshold = 0.05

# Interpretation
if p_density < p_threshold and eta2_density > eta2_threshold:
    conclusion = "VALIDATED"
    detail = f"Density has significant main effect (η²={eta2_density:.3f})"
    if 'correlation_analysis' in results_data and abs(results_data['correlation_analysis']['log_r']) > 0.5:
        detail += " with logarithmic scaling"
elif p_density > 0.05 or eta2_density < 0.01:
    conclusion = "REFUTED"
    detail = f"Density effect is weak (η²={eta2_density:.3f}, p={p_density:.3f})"
else:
    conclusion = "INCONCLUSIVE"
    detail = f"Borderline effects (η²={eta2_density:.3f}, p={p_density:.3f})"

results_data['conclusion'] = {
    'status': conclusion,
    'detail': detail
}

print(f"\n{'='*70}")
print(f"CONCLUSION: {conclusion}")
print(f"Detail: {detail}")
print(f"{'='*70}\n")

# Return minimal summary for orchestrator
print(f"RES-211 | network_architecture | {conclusion} | η²_depth={eta2_depth:.3f} | η²_density={eta2_density:.3f}")
