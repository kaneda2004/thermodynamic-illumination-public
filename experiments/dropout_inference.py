#!/usr/bin/env python3
"""
RES-137: Test if dropout during CPPN inference increases order.

Hypothesis: Dropout during CPPN inference increases order by forcing
redundant representations.

Approach: Compare order distributions from CPPNs with and without
connection dropout during inference (not training).
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative, PRIOR_SIGMA


def create_cppn_with_hidden(n_hidden: int = 3) -> CPPN:
    """Create a CPPN with hidden nodes."""
    # Input nodes: x, y, r, bias (IDs 0-3)
    # Output node: ID 4
    # Hidden nodes: IDs 5+

    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),
    ]

    input_ids = [0, 1, 2, 3]
    hidden_ids = []
    activations = ['tanh', 'sin', 'gauss', 'sigmoid', 'relu']

    # Add hidden nodes
    for i in range(n_hidden):
        node_id = 5 + i
        hidden_ids.append(node_id)
        activation = np.random.choice(activations)
        nodes.append(Node(node_id, activation, np.random.randn() * PRIOR_SIGMA))

    connections = []

    # Connect inputs to hidden
    for inp_id in input_ids:
        for hidden_id in hidden_ids:
            connections.append(Connection(inp_id, hidden_id, np.random.randn() * PRIOR_SIGMA))

    # Connect hidden to output
    for hidden_id in hidden_ids:
        connections.append(Connection(hidden_id, 4, np.random.randn() * PRIOR_SIGMA))

    # Also connect inputs directly to output (skip connections)
    for inp_id in input_ids:
        connections.append(Connection(inp_id, 4, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=4)


def cppn_with_dropout(cppn: CPPN, dropout_rate: float = 0.3) -> CPPN:
    """Create a copy of CPPN with some connections disabled."""
    dropout_cppn = cppn.copy()
    for conn in dropout_cppn.connections:
        if np.random.random() < dropout_rate:
            conn.enabled = False
    return dropout_cppn


def main():
    np.random.seed(42)
    n_samples = 500
    n_hidden = 3  # Use 3 hidden nodes for complexity
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    results = {rate: [] for rate in dropout_rates}

    print(f"Testing dropout at inference on {n_samples} CPPNs with {n_hidden} hidden nodes...")

    for i in range(n_samples):
        # Generate a CPPN
        cppn = create_cppn_with_hidden(n_hidden)

        for rate in dropout_rates:
            if rate == 0.0:
                # No dropout
                img = cppn.render(32)
            else:
                # Apply dropout
                dropout_cppn = cppn_with_dropout(cppn, rate)
                img = dropout_cppn.render(32)

            order = order_multiplicative(img)
            results[rate].append(order)

    # Analysis
    print("\nResults by dropout rate:")
    print("=" * 50)

    baseline = np.array(results[0.0])
    print(f"Baseline (no dropout): mean={np.mean(baseline):.4f}, std={np.std(baseline):.4f}")

    for rate in dropout_rates[1:]:
        orders = np.array(results[rate])
        mean = np.mean(orders)
        std = np.std(orders)

        # Cohen's d
        pooled_std = np.sqrt((np.std(baseline)**2 + std**2) / 2)
        if pooled_std > 0:
            d = (mean - np.mean(baseline)) / pooled_std
        else:
            d = 0.0

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(orders, baseline, alternative='two-sided')

        print(f"Dropout {rate:.1f}: mean={mean:.4f}, std={std:.4f}, d={d:.3f}, p={p:.2e}")

    # Main comparison: 0.3 dropout vs no dropout
    dropout_30 = np.array(results[0.3])
    pooled_std = np.sqrt((np.std(baseline)**2 + np.std(dropout_30)**2) / 2)
    if pooled_std > 0:
        d_main = (np.mean(dropout_30) - np.mean(baseline)) / pooled_std
    else:
        d_main = 0.0
    stat, p_main = stats.mannwhitneyu(dropout_30, baseline, alternative='two-sided')

    print("\n" + "=" * 50)
    print("MAIN RESULT (30% dropout vs baseline):")
    print(f"  Effect size d = {d_main:.3f}")
    print(f"  p-value = {p_main:.2e}")
    print(f"  Baseline mean: {np.mean(baseline):.4f}")
    print(f"  Dropout mean: {np.mean(dropout_30):.4f}")

    # Check correlation between dropout rate and mean order
    rates = np.array(dropout_rates)
    means = np.array([np.mean(results[r]) for r in dropout_rates])
    r_corr, p_corr = stats.pearsonr(rates, means)
    print(f"\nCorrelation (rate vs order): r={r_corr:.3f}, p={p_corr:.3e}")

    # Verdict
    if abs(d_main) > 0.5 and p_main < 0.01:
        if d_main > 0:
            print("\nVERDICT: VALIDATED - Dropout increases order")
        else:
            print("\nVERDICT: REFUTED - Dropout decreases order (opposite effect)")
    else:
        print("\nVERDICT: REFUTED - Effect size too small or not significant")

    return d_main, p_main, r_corr

if __name__ == "__main__":
    main()
