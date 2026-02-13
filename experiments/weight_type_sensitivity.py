"""
RES-157: Input weight perturbations affect order more than output weight perturbations in CPPNs

Hypothesis: Input weights (connecting x,y,r,bias to hidden/output) have different sensitivity
profiles than output weights (connecting to output node). Tests if network architecture
creates asymmetric sensitivity to perturbations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats
import json

def categorize_weights(cppn):
    """Categorize weights by connection type."""
    input_ids = set(cppn.input_ids)
    output_id = cppn.output_id

    input_weight_indices = []
    output_weight_indices = []
    hidden_weight_indices = []

    idx = 0
    for c in cppn.connections:
        if not c.enabled:
            continue
        if c.from_id in input_ids:
            input_weight_indices.append(idx)
        elif c.to_id == output_id:
            output_weight_indices.append(idx)
        else:
            hidden_weight_indices.append(idx)
        idx += 1

    # Biases are after connection weights
    n_weights = idx
    bias_indices = []
    for n in cppn.nodes:
        if n.id not in input_ids:
            if n.id == output_id:
                output_weight_indices.append(idx)  # Output bias
            else:
                hidden_weight_indices.append(idx)  # Hidden bias
            idx += 1

    return {
        'input': input_weight_indices,
        'output': output_weight_indices,
        'hidden': hidden_weight_indices
    }

def measure_sensitivity_by_type(cppn, epsilon=0.01, n_trials=10, size=32):
    """Measure order sensitivity for each weight type."""
    base_img = cppn.render(size)
    base_order = order_multiplicative(base_img)

    categories = categorize_weights(cppn)
    weights = cppn.get_weights()

    results = {}
    for cat_name, indices in categories.items():
        if not indices:
            results[cat_name] = {'mean_delta': 0.0, 'std_delta': 0.0, 'n': 0}
            continue

        deltas = []
        for _ in range(n_trials):
            for idx in indices:
                # Perturb single weight
                perturbed = weights.copy()
                perturbed[idx] += np.random.randn() * epsilon

                test_cppn = cppn.copy()
                test_cppn.set_weights(perturbed)
                test_img = test_cppn.render(size)
                test_order = order_multiplicative(test_img)

                deltas.append(abs(test_order - base_order))

        results[cat_name] = {
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas)),
            'n': len(deltas)
        }

    return results, base_order

def create_cppn_with_hidden(n_hidden=2, seed=None):
    """Create a CPPN with hidden nodes for more complex weight structure."""
    if seed is not None:
        np.random.seed(seed)

    cppn = CPPN()

    # Add hidden nodes
    for i in range(n_hidden):
        hidden_id = 5 + i
        cppn.nodes.append(type(cppn.nodes[0])(hidden_id, 'tanh', np.random.randn()))

        # Connect inputs to hidden
        for inp_id in cppn.input_ids:
            cppn.connections.append(type(cppn.connections[0])(inp_id, hidden_id, np.random.randn()))

        # Connect hidden to output
        cppn.connections.append(type(cppn.connections[0])(hidden_id, cppn.output_id, np.random.randn()))

    return cppn

def main():
    set_global_seed(42)

    n_cppns = 200
    epsilon = 0.05
    n_trials = 5

    all_input_deltas = []
    all_output_deltas = []
    all_hidden_deltas = []
    all_base_orders = []

    print("Testing weight type sensitivity...")

    for i in range(n_cppns):
        # Mix of simple and complex CPPNs
        if i % 2 == 0:
            cppn = CPPN()  # Simple: 4 inputs -> 1 output
        else:
            cppn = create_cppn_with_hidden(n_hidden=2, seed=i)

        results, base_order = measure_sensitivity_by_type(cppn, epsilon, n_trials)

        all_base_orders.append(base_order)

        if results['input']['n'] > 0:
            all_input_deltas.append(results['input']['mean_delta'])
        if results['output']['n'] > 0:
            all_output_deltas.append(results['output']['mean_delta'])
        if results['hidden']['n'] > 0:
            all_hidden_deltas.append(results['hidden']['mean_delta'])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_cppns} CPPNs")

    # Statistical analysis
    input_mean = np.mean(all_input_deltas)
    output_mean = np.mean(all_output_deltas)

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(all_input_deltas, all_output_deltas, alternative='two-sided')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(all_input_deltas) + np.var(all_output_deltas)) / 2)
    cohens_d = (input_mean - output_mean) / (pooled_std + 1e-10)

    # Also compare with hidden weights if available
    if all_hidden_deltas:
        hidden_mean = np.mean(all_hidden_deltas)
        u_hidden, p_hidden = stats.mannwhitneyu(all_input_deltas, all_hidden_deltas, alternative='two-sided')
    else:
        hidden_mean = None
        p_hidden = None

    print("\n=== RESULTS ===")
    print(f"Input weight sensitivity: {input_mean:.6f} +/- {np.std(all_input_deltas):.6f}")
    print(f"Output weight sensitivity: {output_mean:.6f} +/- {np.std(all_output_deltas):.6f}")
    if hidden_mean is not None:
        print(f"Hidden weight sensitivity: {hidden_mean:.6f} +/- {np.std(all_hidden_deltas):.6f}")
    print(f"\nInput vs Output: U={u_stat:.1f}, p={p_value:.6f}, d={cohens_d:.3f}")
    if p_hidden is not None:
        print(f"Input vs Hidden: p={p_hidden:.6f}")

    # Determine outcome
    validated = p_value < 0.01 and abs(cohens_d) > 0.5

    if validated:
        if cohens_d > 0:
            status = "validated"
            summary = f"Input weights more sensitive than output (d={cohens_d:.2f}, p={p_value:.2e})"
        else:
            status = "refuted"
            summary = f"Output weights more sensitive than input (d={cohens_d:.2f}, p={p_value:.2e})"
    else:
        if p_value >= 0.01:
            status = "refuted"
            summary = f"No significant difference between weight types (p={p_value:.3f}, d={cohens_d:.2f})"
        else:
            status = "inconclusive"
            summary = f"Significant but small effect (p={p_value:.2e}, d={cohens_d:.2f})"

    print(f"\nSTATUS: {status.upper()}")
    print(f"SUMMARY: {summary}")

    # Save results
    os.makedirs('results/weight_type_sensitivity', exist_ok=True)
    results_data = {
        'input_mean': float(input_mean),
        'output_mean': float(output_mean),
        'hidden_mean': float(hidden_mean) if hidden_mean else None,
        'input_std': float(np.std(all_input_deltas)),
        'output_std': float(np.std(all_output_deltas)),
        'u_statistic': float(u_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'n_cppns': n_cppns,
        'epsilon': epsilon,
        'status': status,
        'summary': summary
    }

    with open('results/weight_type_sensitivity/results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    return status, cohens_d, p_value, summary

if __name__ == '__main__':
    main()
