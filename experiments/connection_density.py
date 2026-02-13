"""
RES-080: Connection Density vs Order
Tests whether connection density affects order independently of network complexity.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA

def create_cppn_with_density(n_hidden: int, density: float, seed: int) -> CPPN:
    """Create CPPN with specified hidden nodes and connection density."""
    np.random.seed(seed)

    input_ids = [0, 1, 2, 3]
    output_id = 4
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

def run_experiment():
    """Test connection density effect on order."""
    n_samples = 100
    n_hidden = 4  # Fixed complexity
    densities = [0.2, 0.4, 0.6, 0.8, 1.0]

    results = {d: [] for d in densities}

    for seed in range(n_samples):
        for density in densities:
            cppn = create_cppn_with_density(n_hidden, density, seed * 1000 + int(density * 100))
            img = cppn.render(32)
            order = order_multiplicative(img)
            results[density].append(order)

    # Compute statistics
    print("Connection Density vs Order")
    print("=" * 50)

    means = []
    for d in densities:
        m = np.mean(results[d])
        s = np.std(results[d])
        means.append(m)
        print(f"Density {d:.1f}: mean={m:.4f}, std={s:.4f}")

    # Correlation test
    all_densities = []
    all_orders = []
    for d in densities:
        all_densities.extend([d] * len(results[d]))
        all_orders.extend(results[d])

    r, p = stats.spearmanr(all_densities, all_orders)
    print(f"\nSpearman correlation: r={r:.4f}, p={p:.2e}")

    # Effect size: compare lowest vs highest density
    low = results[densities[0]]
    high = results[densities[-1]]
    cohens_d = (np.mean(high) - np.mean(low)) / np.sqrt((np.var(low) + np.var(high)) / 2)
    print(f"Cohen's d (0.2 vs 1.0): {cohens_d:.4f}")

    # Statistical test
    _, p_mw = stats.mannwhitneyu(low, high, alternative='two-sided')
    print(f"Mann-Whitney p-value: {p_mw:.2e}")

    # Verdict
    validated = abs(r) > 0.1 and p < 0.01 and abs(cohens_d) > 0.5
    print(f"\nVERDICT: {'VALIDATED' if validated else 'REFUTED'}")
    print(f"  |r|={abs(r):.3f} > 0.1: {abs(r) > 0.1}")
    print(f"  p={p:.2e} < 0.01: {p < 0.01}")
    print(f"  |d|={abs(cohens_d):.3f} > 0.5: {abs(cohens_d) > 0.5}")

    return {
        'r': float(r),
        'p': float(p),
        'cohens_d': float(cohens_d),
        'validated': bool(validated),
        'means': {d: float(np.mean(results[d])) for d in densities}
    }

if __name__ == '__main__':
    run_experiment()
