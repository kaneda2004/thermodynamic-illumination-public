"""
RES-128: Compare order metric rankings across alternative metrics

Hypothesis: order_multiplicative and alternative metrics (Ising energy, spectral
coherence, compressibility) produce consistent rankings (Spearman rho > 0.8)
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, compute_compressibility,
    compute_spectral_coherence, compute_edge_density
)

def compute_ising_energy(img: np.ndarray) -> float:
    """Ising model energy: lower = more aligned neighbors (more structure)."""
    # Convert to -1/+1 for Ising model
    spins = 2 * img.astype(float) - 1
    h, w = img.shape
    energy = 0
    for i in range(h):
        for j in range(w):
            # Sum of interactions with right and down neighbors
            if j < w - 1:
                energy -= spins[i, j] * spins[i, j+1]
            if i < h - 1:
                energy -= spins[i, j] * spins[i+1, j]
    # Normalize by number of bonds (lower is more ordered)
    # Return negative so higher = more ordered (for consistent comparison)
    max_bonds = 2 * h * w - h - w
    return -energy / max_bonds  # Higher = more aligned = more structured


def main():
    np.random.seed(42)

    # Generate diverse set of CPPN images
    n_images = 100
    images = []
    for _ in range(n_images):
        cppn = CPPN()
        size = 32
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        continuous = cppn.activate(xx, yy)
        binary = (continuous > 0.5).astype(np.uint8)
        images.append(binary)

    # Also add some random images for diversity
    for _ in range(20):
        images.append(np.random.randint(0, 2, (32, 32), dtype=np.uint8))

    # Compute all metrics
    metrics = {
        'order_multiplicative': [],
        'compressibility': [],
        'spectral_coherence': [],
        'ising_energy': [],
        'edge_density': []
    }

    for img in images:
        metrics['order_multiplicative'].append(order_multiplicative(img))
        metrics['compressibility'].append(compute_compressibility(img))
        metrics['spectral_coherence'].append(compute_spectral_coherence(img))
        metrics['ising_energy'].append(compute_ising_energy(img))
        metrics['edge_density'].append(compute_edge_density(img))

    # Convert to arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    # Compute pairwise Spearman correlations with order_multiplicative
    print("="*60)
    print("SPEARMAN RANK CORRELATIONS WITH order_multiplicative")
    print("="*60)

    correlations = {}
    for metric_name in ['compressibility', 'spectral_coherence', 'ising_energy', 'edge_density']:
        rho, p_value = stats.spearmanr(metrics['order_multiplicative'], metrics[metric_name])
        correlations[metric_name] = (rho, p_value)
        print(f"{metric_name:25s}: rho = {rho:+.3f}, p = {p_value:.2e}")

    # Cross-correlations between alternative metrics
    print("\n" + "="*60)
    print("CROSS-CORRELATIONS BETWEEN ALTERNATIVE METRICS")
    print("="*60)

    alt_metrics = ['compressibility', 'spectral_coherence', 'ising_energy', 'edge_density']
    for i, m1 in enumerate(alt_metrics):
        for m2 in alt_metrics[i+1:]:
            rho, p = stats.spearmanr(metrics[m1], metrics[m2])
            print(f"{m1:20s} vs {m2:20s}: rho = {rho:+.3f}")

    # Test hypothesis: are correlations > 0.8?
    print("\n" + "="*60)
    print("HYPOTHESIS TEST: Spearman rho > 0.8")
    print("="*60)

    threshold = 0.8
    passed = []
    failed = []
    for metric_name, (rho, p) in correlations.items():
        if rho > threshold:
            passed.append(metric_name)
            print(f"PASS {metric_name}: {rho:.3f} > {threshold}")
        else:
            failed.append(metric_name)
            print(f"FAIL {metric_name}: {rho:.3f} <= {threshold}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Metrics tested: {len(correlations)}")
    print(f"Passed (rho > 0.8): {len(passed)}")
    print(f"Failed (rho <= 0.8): {len(failed)}")
    print(f"Mean correlation: {np.mean([r for r,p in correlations.values()]):.3f}")
    print(f"Min correlation: {min([r for r,p in correlations.values()]):.3f}")
    print(f"Max correlation: {max([r for r,p in correlations.values()]):.3f}")

    # Determine outcome
    if len(passed) == len(correlations):
        status = "VALIDATED"
        result = "All alternative metrics rank images consistently with order_multiplicative (rho > 0.8)"
    elif len(passed) == 0:
        status = "REFUTED"
        result = "No alternative metrics achieve rho > 0.8 with order_multiplicative"
    else:
        status = "PARTIAL"
        result = f"{len(passed)}/{len(correlations)} metrics achieve rho > 0.8"

    print(f"\nSTATUS: {status}")
    print(f"RESULT: {result}")

    # Effect size: average rho
    effect_size = np.mean([r for r,p in correlations.values()])
    min_p = min([p for r,p in correlations.values()])
    print(f"\nEFFECT SIZE (mean rho): {effect_size:.3f}")
    print(f"MIN P-VALUE: {min_p:.2e}")


if __name__ == "__main__":
    main()
