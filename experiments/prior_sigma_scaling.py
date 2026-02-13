"""
RES-152: Test whether larger prior std produces higher order via activation saturation.

Hypothesis: Larger prior std (sigma=2) produces higher order than sigma=1 via saturation

Rationale:
- RES-079 showed coordinate range scaling affects order through saturation
- Weight magnitudes also control pre-activation scale
- Larger weights -> more saturation -> potentially higher order
- But RES-130 showed saturation NEGATIVELY correlates with order
- This tests whether prior scale matters independently
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative

def create_cppn_with_sigma(sigma: float, depth: int = 2) -> CPPN:
    """Create CPPN with weights sampled from N(0, sigma)."""
    # Build architecture with hidden layers
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    input_ids = [0, 1, 2, 3]
    next_id = 4
    prev_layer = input_ids

    activations = ['tanh', 'sin', 'sigmoid', 'gauss']

    for d in range(depth):
        layer_size = 5
        layer_ids = []
        for i in range(layer_size):
            act = activations[(d + i) % len(activations)]
            bias = np.random.randn() * sigma
            nodes.append(Node(next_id, act, bias))
            layer_ids.append(next_id)
            # Connect from previous layer
            for prev_id in prev_layer:
                weight = np.random.randn() * sigma
                connections.append(Connection(prev_id, next_id, weight))
            next_id += 1
        prev_layer = layer_ids

    # Output node
    output_id = next_id
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * sigma))
    for prev_id in prev_layer:
        weight = np.random.randn() * sigma
        connections.append(Connection(prev_id, output_id, weight))

    cppn = CPPN()
    cppn.nodes = nodes
    cppn.connections = connections
    cppn.input_ids = input_ids
    cppn.output_id = output_id
    return cppn

def measure_saturation(cppn: CPPN, size: int = 32) -> float:
    """Measure activation saturation (fraction near 0 or 1)."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    # Get raw activations (before thresholding)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)
    values = {0: x, 1: y, 2: r, 3: bias}

    from core.thermo_sampler_v3 import ACTIVATIONS

    for nid in cppn._get_eval_order():
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[nid] = ACTIVATIONS[node.activation](total)

    output = values[cppn.output_id]
    # Saturation: fraction near 0 or 1 (within 0.05)
    saturated = (output < 0.05) | (output > 0.95)
    return np.mean(saturated)

def run_experiment():
    np.random.seed(42)

    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
    n_samples = 200

    results = {sigma: {'orders': [], 'saturations': []} for sigma in sigmas}

    print("Testing prior sigma scaling effect on order...")
    print(f"Sigmas: {sigmas}, N={n_samples} per condition")

    for sigma in sigmas:
        for _ in range(n_samples):
            cppn = create_cppn_with_sigma(sigma, depth=2)
            img = cppn.render(size=32)
            order = order_multiplicative(img)
            saturation = measure_saturation(cppn)
            results[sigma]['orders'].append(order)
            results[sigma]['saturations'].append(saturation)

        orders = results[sigma]['orders']
        sats = results[sigma]['saturations']
        print(f"  sigma={sigma}: order={np.mean(orders):.4f}+-{np.std(orders):.4f}, "
              f"sat={np.mean(sats):.3f}")

    # Statistical tests
    print("\n=== Statistical Analysis ===")

    # 1. Kruskal-Wallis test across all groups
    all_orders = [results[s]['orders'] for s in sigmas]
    H, p_kruskal = stats.kruskal(*all_orders)
    print(f"Kruskal-Wallis: H={H:.2f}, p={p_kruskal:.2e}")

    # 2. Pairwise comparison: sigma=1 vs sigma=2 (primary hypothesis)
    o1 = np.array(results[1.0]['orders'])
    o2 = np.array(results[2.0]['orders'])

    # Mann-Whitney U test
    U, p_mw = stats.mannwhitneyu(o1, o2, alternative='two-sided')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(o1) + np.var(o2)) / 2)
    cohens_d = (np.mean(o2) - np.mean(o1)) / pooled_std if pooled_std > 0 else 0

    print(f"\nPrimary comparison (sigma=1 vs sigma=2):")
    print(f"  Mann-Whitney U: p={p_mw:.4e}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mean order sigma=1: {np.mean(o1):.4f}")
    print(f"  Mean order sigma=2: {np.mean(o2):.4f}")

    # 3. Correlation between sigma and order
    all_sigmas_flat = []
    all_orders_flat = []
    for sigma in sigmas:
        all_sigmas_flat.extend([sigma] * n_samples)
        all_orders_flat.extend(results[sigma]['orders'])

    rho, p_corr = stats.spearmanr(all_sigmas_flat, all_orders_flat)
    print(f"\nSigma-Order correlation: rho={rho:.3f}, p={p_corr:.2e}")

    # 4. Correlation between saturation and order (within each sigma)
    print("\nSaturation-Order correlations by sigma:")
    for sigma in sigmas:
        r, p = stats.spearmanr(results[sigma]['saturations'], results[sigma]['orders'])
        print(f"  sigma={sigma}: r={r:.3f}, p={p:.3e}")

    # 5. Overall saturation-order correlation
    all_sats = []
    for sigma in sigmas:
        all_sats.extend(results[sigma]['saturations'])
    r_sat, p_sat = stats.spearmanr(all_sats, all_orders_flat)
    print(f"\nOverall saturation-order: r={r_sat:.3f}, p={p_sat:.2e}")

    # Summary
    print("\n=== SUMMARY ===")
    if p_corr < 0.01 and abs(cohens_d) > 0.5:
        direction = "HIGHER" if cohens_d > 0 else "LOWER"
        print(f"VALIDATED: Larger sigma produces {direction} order (d={cohens_d:.2f})")
    elif p_corr < 0.01 and abs(cohens_d) < 0.5:
        print(f"WEAK EFFECT: Significant but small (d={cohens_d:.2f})")
    else:
        print(f"REFUTED: No significant effect of prior sigma on order (d={cohens_d:.2f})")

    return {
        'kruskal_H': H,
        'kruskal_p': p_kruskal,
        'cohens_d': cohens_d,
        'p_primary': p_mw,
        'rho_sigma_order': rho,
        'p_corr': p_corr,
        'r_saturation_order': r_sat,
        'mean_orders': {s: np.mean(results[s]['orders']) for s in sigmas},
        'mean_saturations': {s: np.mean(results[s]['saturations']) for s in sigmas}
    }

if __name__ == '__main__':
    results = run_experiment()
