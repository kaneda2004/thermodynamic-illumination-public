"""
RES-148: High-order CPPNs compress information faster in early layers than low-order CPPNs

Hypothesis: The rate of information loss (MI decay) differs between high and low order CPPNs,
with high-order CPPNs compressing more aggressively in early layers.

Building on:
- RES-122: MI(input, layer) monotonically decreases through CPPN layers
- RES-087: High-order CPPNs show lower inter-layer MI
- RES-138: Spatial variance of hidden activations correlates with order

Method:
1. Generate CPPNs with hidden layers and compute order
2. For each CPPN, estimate MI between input coordinates and each hidden layer
3. Compute the layer-to-layer MI decay rate
4. Test if early-layer compression rate correlates with final output order
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, Node, Connection, ACTIVATIONS, order_multiplicative
from scipy import stats
from dataclasses import dataclass, field
import json


def estimate_mi_binned(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Estimate mutual information using binned histogram method."""
    # Flatten if needed
    x = x.flatten()
    y = y.flatten()

    # Bin the continuous values
    x_bins = np.digitize(x, np.linspace(x.min() - 1e-10, x.max() + 1e-10, bins + 1))
    y_bins = np.digitize(y, np.linspace(y.min() - 1e-10, y.max() + 1e-10, bins + 1))

    # Joint and marginal histograms
    joint_hist = np.histogram2d(x_bins, y_bins, bins=bins)[0]
    joint_hist = joint_hist / joint_hist.sum()
    joint_hist = joint_hist + 1e-10  # Avoid log(0)

    p_x = joint_hist.sum(axis=1)
    p_y = joint_hist.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 1e-9:
                mi += joint_hist[i, j] * np.log2(joint_hist[i, j] / (p_x[i] * p_y[j] + 1e-10))
    return max(0, mi)


@dataclass
class DeepCPPN:
    """CPPN with multiple hidden layers for information flow analysis."""
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = -1
    hidden_layers: list = field(default_factory=list)  # List of lists of node ids per layer

    @classmethod
    def create(cls, depth: int = 4, width: int = 4, seed: int = None):
        """Create a deep CPPN with specified depth and width."""
        if seed is not None:
            np.random.seed(seed)

        nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
        ]
        connections = []
        hidden_layers = []

        next_id = 4
        prev_layer = [0, 1, 2, 3]

        # Create hidden layers
        activations = ['tanh', 'sin', 'gauss', 'sigmoid']
        for layer_idx in range(depth):
            current_layer = []
            for w in range(width):
                act = activations[(layer_idx + w) % len(activations)]
                nodes.append(Node(next_id, act, np.random.randn()))
                current_layer.append(next_id)
                # Connect from all previous layer nodes
                for prev_id in prev_layer:
                    connections.append(Connection(prev_id, next_id, np.random.randn()))
                next_id += 1
            hidden_layers.append(current_layer)
            prev_layer = current_layer

        # Output node
        output_id = next_id
        nodes.append(Node(output_id, 'sigmoid', np.random.randn()))
        for prev_id in prev_layer:
            connections.append(Connection(prev_id, output_id, np.random.randn()))

        return cls(nodes=nodes, connections=connections, output_id=output_id,
                   hidden_layers=hidden_layers)

    def get_layer_activations(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Get activations at each layer."""
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}

        layer_activations = {'input': np.stack([x, y, r, bias], axis=-1)}

        # Process hidden layers
        for layer_idx, layer_ids in enumerate(self.hidden_layers):
            for nid in layer_ids:
                node = next(n for n in self.nodes if n.id == nid)
                total = np.zeros_like(x) + node.bias
                for conn in self.connections:
                    if conn.to_id == nid and conn.enabled and conn.from_id in values:
                        total += values[conn.from_id] * conn.weight
                values[nid] = ACTIVATIONS[node.activation](total)

            # Stack layer activations
            layer_acts = np.stack([values[nid] for nid in layer_ids], axis=-1)
            layer_activations[f'layer_{layer_idx}'] = layer_acts

        # Output
        node = next(n for n in self.nodes if n.id == self.output_id)
        total = np.zeros_like(x) + node.bias
        for conn in self.connections:
            if conn.to_id == self.output_id and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[self.output_id] = ACTIVATIONS[node.activation](total)
        layer_activations['output'] = values[self.output_id]

        return layer_activations

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        activations = self.get_layer_activations(x, y)
        return (activations['output'] > 0.5).astype(np.uint8)


def compute_mi_profile(cppn: DeepCPPN, size: int = 32) -> dict:
    """Compute MI between input coordinates and each layer."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    activations = cppn.get_layer_activations(x, y)

    # Combine input coordinates into single representation
    input_combined = x.flatten() + y.flatten() * 2  # Simple combination

    mi_profile = {}

    for layer_name, layer_act in activations.items():
        if layer_name == 'input':
            continue

        if layer_act.ndim == 3:  # Hidden layers have shape (h, w, width)
            # Average MI across channels
            mi_values = []
            for ch in range(layer_act.shape[-1]):
                mi_val = estimate_mi_binned(input_combined, layer_act[:, :, ch].flatten())
                mi_values.append(mi_val)
            mi_profile[layer_name] = np.mean(mi_values)
        else:  # Output is 2D
            mi_profile[layer_name] = estimate_mi_binned(input_combined, layer_act.flatten())

    return mi_profile


def compute_compression_rate(mi_profile: dict, depth: int) -> dict:
    """Compute per-layer compression rates (MI decay)."""
    # Get MI values in order
    mi_values = []
    for i in range(depth):
        mi_values.append(mi_profile.get(f'layer_{i}', 0))
    mi_values.append(mi_profile.get('output', 0))

    # Compute decay rates between consecutive layers
    compression_rates = {}
    for i in range(len(mi_values) - 1):
        if mi_values[i] > 1e-6:
            rate = (mi_values[i] - mi_values[i + 1]) / mi_values[i]
        else:
            rate = 0.0
        compression_rates[f'rate_{i}to{i+1}'] = rate

    # Early vs late compression
    early_rates = [compression_rates[f'rate_{i}to{i+1}'] for i in range(depth // 2)]
    late_rates = [compression_rates[f'rate_{i}to{i+1}'] for i in range(depth // 2, depth)]

    compression_rates['early_mean'] = np.mean(early_rates) if early_rates else 0
    compression_rates['late_mean'] = np.mean(late_rates) if late_rates else 0
    compression_rates['early_late_ratio'] = (
        compression_rates['early_mean'] / (compression_rates['late_mean'] + 1e-6)
    )

    return compression_rates


def main():
    print("RES-148: Information compression rate in CPPN layers")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500
    depth = 4
    width = 4

    results = []

    print(f"Generating {n_samples} deep CPPNs (depth={depth}, width={width})...")

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_samples}")

        cppn = DeepCPPN.create(depth=depth, width=width, seed=i)
        img = cppn.render(32)
        order = order_multiplicative(img)

        mi_profile = compute_mi_profile(cppn, 32)
        compression = compute_compression_rate(mi_profile, depth)

        results.append({
            'seed': i,
            'order': order,
            **mi_profile,
            **compression
        })

    # Convert to arrays for analysis
    orders = np.array([r['order'] for r in results])
    early_rates = np.array([r['early_mean'] for r in results])
    late_rates = np.array([r['late_mean'] for r in results])
    early_late_ratios = np.array([r['early_late_ratio'] for r in results])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Primary hypothesis: early compression rate correlates with order
    r_early, p_early = stats.spearmanr(orders, early_rates)
    r_late, p_late = stats.spearmanr(orders, late_rates)
    r_ratio, p_ratio = stats.spearmanr(orders, early_late_ratios)

    print(f"\nCorrelations with order:")
    print(f"  Early compression rate:  r={r_early:.3f}, p={p_early:.2e}")
    print(f"  Late compression rate:   r={r_late:.3f}, p={p_late:.2e}")
    print(f"  Early/Late ratio:        r={r_ratio:.3f}, p={p_ratio:.2e}")

    # Split by order tertiles
    order_tertiles = np.percentile(orders, [33, 66])
    low_mask = orders < order_tertiles[0]
    high_mask = orders > order_tertiles[1]

    low_early = early_rates[low_mask]
    high_early = early_rates[high_mask]

    t_stat, p_val = stats.ttest_ind(high_early, low_early)
    cohens_d = (np.mean(high_early) - np.mean(low_early)) / np.sqrt(
        (np.var(high_early) + np.var(low_early)) / 2
    )

    print(f"\nHigh vs Low order comparison (early compression rate):")
    print(f"  High-order mean: {np.mean(high_early):.4f} (std={np.std(high_early):.4f})")
    print(f"  Low-order mean:  {np.mean(low_early):.4f} (std={np.std(low_early):.4f})")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_val:.2e}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Per-layer MI correlation with order
    print("\nPer-layer MI correlation with order:")
    for layer_idx in range(depth + 1):
        layer_name = f'layer_{layer_idx}' if layer_idx < depth else 'output'
        mi_vals = np.array([r.get(layer_name, 0) for r in results])
        r, p = stats.spearmanr(orders, mi_vals)
        print(f"  {layer_name}: r={r:.3f}, p={p:.2e}")

    # Effect size threshold check
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    effect_threshold = 0.5
    p_threshold = 0.01

    is_significant = p_early < p_threshold
    is_large_effect = abs(cohens_d) > effect_threshold
    direction = "higher" if r_early > 0 else "lower"

    print(f"Primary hypothesis (early compression correlates with order):")
    print(f"  Significant (p<0.01): {is_significant}")
    print(f"  Large effect (|d|>0.5): {is_large_effect}")
    print(f"  Direction: High-order CPPNs have {direction} early compression rates")

    if is_significant and is_large_effect:
        status = "VALIDATED"
    elif is_significant and not is_large_effect:
        status = "REFUTED (significant but small effect)"
    else:
        status = "REFUTED"

    print(f"\nFinal status: {status}")

    # Save summary
    summary = {
        'hypothesis': 'High-order CPPNs compress information faster in early layers',
        'n_samples': n_samples,
        'depth': depth,
        'width': width,
        'r_early_order': float(r_early),
        'p_early_order': float(p_early),
        'r_late_order': float(r_late),
        'p_late_order': float(p_late),
        'cohens_d': float(cohens_d),
        't_stat': float(t_stat),
        'p_val_ttest': float(p_val),
        'high_early_mean': float(np.mean(high_early)),
        'low_early_mean': float(np.mean(low_early)),
        'status': status
    }

    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    return summary


if __name__ == '__main__':
    main()
