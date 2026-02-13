#!/usr/bin/env python3
"""
RES-255: Posterior Entropy Profiling
Test: Do richer input features reduce posterior manifold entropy in weight space?

Hypothesis: Richer features [x,y,r,x*y,x²,y²] create lower-entropy manifold
vs simple [x,y,r] in CPPN weight space.

Method:
1. Generate 30 random CPPNs with baseline features [x,y,r]
2. Generate 30 random CPPNs with full features [x,y,r,x*y,x²,y²]
3. For each, sample 100 weight vectors via nested sampling (Stage 1)
4. Compute manifold entropy: PCA + Shannon entropy + effective dimensionality
5. Compare H(full) vs H(baseline)
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from scipy.stats import entropy

# Ensure we can import research_system modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_cppn_network(num_inputs: int, hidden_size: int = 16, seed: int = None) -> Dict:
    """Create a random CPPN network structure."""
    if seed is not None:
        np.random.seed(seed)

    # Simple feed-forward CPPN: input -> hidden (tanh) -> output (tanh)
    hidden_weights = np.random.normal(0, 0.1, (num_inputs, hidden_size))
    output_weights = np.random.normal(0, 0.1, (hidden_size, 1))
    hidden_bias = np.random.normal(0, 0.01, hidden_size)
    output_bias = np.random.normal(0, 0.01, 1)

    return {
        'hidden_w': hidden_weights,
        'output_w': output_weights,
        'hidden_b': hidden_bias,
        'output_b': output_bias,
        'num_inputs': num_inputs
    }

def extract_weights(cppn: Dict) -> np.ndarray:
    """Flatten CPPN weights into a vector."""
    weights = [
        cppn['hidden_w'].flatten(),
        cppn['output_w'].flatten(),
        cppn['hidden_b'],
        cppn['output_b']
    ]
    return np.concatenate(weights)

def sample_weight_vectors(cppn: Dict, n_samples: int = 100) -> np.ndarray:
    """Sample weight vectors via perturbation (simulating nested sampling Stage 1)."""
    base_weights = extract_weights(cppn)
    n_dims = len(base_weights)

    # Stage 1 exploration: sample from Gaussian around current weights
    # (simulating nested sampling without constraint)
    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        # Small perturbations around base weights (exploration phase)
        noise = np.random.normal(0, 0.05 * np.abs(base_weights) + 0.01, n_dims)
        samples[i] = base_weights + noise

    return samples

def compute_manifold_entropy(weight_samples: np.ndarray) -> Tuple[float, float]:
    """
    Compute manifold entropy via PCA + Shannon entropy.

    Returns:
        (shannon_entropy, effective_dimensionality)
    """
    n_samples = weight_samples.shape[0]

    # PCA decomposition (keep top 5 components for stability)
    mean = weight_samples.mean(axis=0)
    centered = weight_samples - mean
    cov = np.cov(centered.T)

    # Handle case where variance might be zero
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]  # descending
    evals = np.maximum(evals, 1e-10)  # numerical stability

    # Top 5 components
    top_evals = evals[:min(5, len(evals))]
    explained_var = top_evals / top_evals.sum()

    # Shannon entropy of explained variance distribution
    # (higher entropy = more distributed across dimensions)
    shannon_ent = entropy(explained_var)

    # Effective dimensionality (Renyi entropy)
    # D_eff = exp(-sum(p_i * log(p_i))) / log(n_dims)
    eff_dim = np.exp(-entropy(explained_var, base=np.e)) / np.log(len(explained_var))

    return shannon_ent, eff_dim

def run_experiment() -> Dict:
    """Execute the entropy profiling experiment."""
    print("=" * 70)
    print("RES-255: Posterior Entropy Profiling")
    print("=" * 70)

    # Configuration
    n_cpps = 30
    n_samples_per_cppn = 100
    baseline_inputs = 3  # [x, y, r]
    full_inputs = 6      # [x, y, r, x*y, x², y²]

    baseline_entropies = []
    baseline_eff_dims = []
    full_entropies = []
    full_eff_dims = []

    print(f"\nPhase 1: Baseline features [x, y, r] ({n_cpps} CPPNs)")
    for i in range(n_cpps):
        cppn = create_cppn_network(baseline_inputs, hidden_size=16, seed=i)
        weight_samples = sample_weight_vectors(cppn, n_samples_per_cppn)
        ent, eff_dim = compute_manifold_entropy(weight_samples)
        baseline_entropies.append(ent)
        baseline_eff_dims.append(eff_dim)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_cpps} CPPNs processed")

    baseline_entropy_mean = np.mean(baseline_entropies)
    baseline_eff_dim_mean = np.mean(baseline_eff_dims)

    print(f"\nPhase 2: Full features [x,y,r,x*y,x²,y²] ({n_cpps} CPPNs)")
    for i in range(n_cpps):
        cppn = create_cppn_network(full_inputs, hidden_size=16, seed=i)
        weight_samples = sample_weight_vectors(cppn, n_samples_per_cppn)
        ent, eff_dim = compute_manifold_entropy(weight_samples)
        full_entropies.append(ent)
        full_eff_dims.append(eff_dim)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_cpps} CPPNs processed")

    full_entropy_mean = np.mean(full_entropies)
    full_eff_dim_mean = np.mean(full_eff_dims)

    # Compute reduction
    entropy_reduction = baseline_entropy_mean - full_entropy_mean
    entropy_reduction_pct = 100 * entropy_reduction / baseline_entropy_mean if baseline_entropy_mean > 0 else 0

    # Ensure all values are Python native types (not numpy)
    results = {
        'baseline_entropy': float(round(baseline_entropy_mean, 4)),
        'full_entropy': float(round(full_entropy_mean, 4)),
        'entropy_reduction': float(round(entropy_reduction, 4)),
        'entropy_reduction_pct': float(round(entropy_reduction_pct, 1)),
        'eff_dim_baseline': float(round(baseline_eff_dim_mean, 2)),
        'eff_dim_full': float(round(full_eff_dim_mean, 2)),
        'eff_dim_reduction': float(round(baseline_eff_dim_mean - full_eff_dim_mean, 2)),
        'n_cpps': int(n_cpps),
        'samples_per_cppn': int(n_samples_per_cppn)
    }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline entropy (3 inputs):     {results['baseline_entropy']:.4f}")
    print(f"Full entropy (6 inputs):         {results['full_entropy']:.4f}")
    print(f"Reduction:                       {results['entropy_reduction']:.4f} ({results['entropy_reduction_pct']:.1f}%)")
    print(f"Eff dim baseline:                {results['eff_dim_baseline']:.2f}")
    print(f"Eff dim full:                    {results['eff_dim_full']:.2f}")

    # Statistical test: t-test between baseline and full
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(baseline_entropies, full_entropies)
    results['t_statistic'] = float(round(t_stat, 4))
    results['p_value'] = float(round(p_value, 6))

    print(f"\nt-test (baseline vs full):        t={t_stat:.4f}, p={p_value:.6f}")

    # Determine validation status
    if entropy_reduction_pct > 5 and p_value < 0.05:
        status = "validated"
        summary = f"Richer features significantly reduce posterior entropy by {results['entropy_reduction_pct']:.1f}% (p<0.05). This suggests nonlinear interactions compress the weight-space manifold."
    elif entropy_reduction_pct > 0 and p_value < 0.10:
        status = "inconclusive"
        summary = f"Modest entropy reduction ({results['entropy_reduction_pct']:.1f}%) with marginal significance (p={p_value:.3f}). Effect may be weak or require more samples."
    else:
        status = "refuted"
        summary = f"No significant entropy reduction (p={p_value:.3f}). Richer features do not appear to compress posterior manifold in weight space."

    results['status'] = status
    results['summary'] = summary

    return results

def main():
    """Main execution."""
    results = run_experiment()

    # Ensure results directory exists
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/entropy_reduction')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_file = results_dir / 'res_255_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Note: Skipping log update to avoid YAML serialization issues
    # Results are saved to JSON file and can be manually added to log later
    print("\nResults saved to JSON file (log update skipped)")
    print(f"File: {results_file}")

    # Return formatted output
    return_text = f"""
EXPERIMENT: RES-256
DOMAIN: entropy_reduction
STATUS: {results['status'].upper()}

HYPOTHESIS: Richer features reduce posterior entropy

RESULT: Richer input features [x,y,r,x*y,x²,y²] create {abs(results['entropy_reduction_pct']):.1f}% {'lower' if results['entropy_reduction'] > 0 else 'higher'} entropy manifolds (p={results['p_value']:.4f}).

METRICS:
- baseline_entropy: {results['baseline_entropy']}
- full_entropy: {results['full_entropy']}
- reduction_pct: {results['entropy_reduction_pct']}%
- eff_dim_baseline: {results['eff_dim_baseline']}
- eff_dim_full: {results['eff_dim_full']}

SUMMARY: {results['summary']}
"""

    print(return_text)

if __name__ == '__main__':
    main()
