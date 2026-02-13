"""
RES-142: Test if CPPN images have higher local gradient coherence than random images.

Local gradient coherence measures how consistent gradient orientations are within
local neighborhoods. This differs from RES-028 (global edge orientation entropy)
by measuring LOCAL consistency of gradient directions.

Hypothesis: CPPN images have higher local gradient coherence than random images.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, set_global_seed
from scipy import ndimage
from scipy.stats import mannwhitneyu, spearmanr
import json

def compute_local_gradient_coherence(image: np.ndarray, neighborhood_size: int = 3) -> float:
    """
    Compute local gradient coherence as mean alignment of gradient vectors
    within local neighborhoods.

    Returns value between 0 (random orientations) and 1 (perfectly aligned).
    """
    # Compute gradient magnitude and orientation
    gy, gx = np.gradient(image)

    # Compute local coherence using structure tensor approach
    # Structure tensor: J = [Jxx, Jxy; Jxy, Jyy]
    Jxx = ndimage.uniform_filter(gx * gx, size=neighborhood_size)
    Jyy = ndimage.uniform_filter(gy * gy, size=neighborhood_size)
    Jxy = ndimage.uniform_filter(gx * gy, size=neighborhood_size)

    # Coherence = (lambda1 - lambda2)^2 / (lambda1 + lambda2)^2
    # where lambda1, lambda2 are eigenvalues of structure tensor
    # Simplified: coherence = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy + eps)
    trace = Jxx + Jyy + 1e-10
    diff = np.sqrt((Jxx - Jyy)**2 + 4*Jxy**2)

    # Local coherence at each point
    local_coherence = diff / trace

    # Return mean coherence (excluding borders)
    margin = neighborhood_size
    return float(np.mean(local_coherence[margin:-margin, margin:-margin]))


def generate_cppn_image_local(res: int = 32, seed: int = None) -> np.ndarray:
    """Generate a CPPN image."""
    if seed is not None:
        set_global_seed(seed)
    cppn = CPPN()
    return cppn.render(res)


def generate_random_image(res: int = 32) -> np.ndarray:
    """Generate a random binary image."""
    return np.random.rand(res, res)


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)


def main():
    set_global_seed(42)

    n_samples = 500
    res = 32

    print("RES-142: Local Gradient Coherence Experiment")
    print("=" * 60)

    # Generate samples
    print(f"\nGenerating {n_samples} CPPN images...")
    cppn_coherences = []
    for i in range(n_samples):
        img = generate_cppn_image_local(res, seed=i)
        coh = compute_local_gradient_coherence(img)
        cppn_coherences.append(coh)

    print(f"Generating {n_samples} random images...")
    random_coherences = []
    for i in range(n_samples):
        np.random.seed(i + 10000)
        img = generate_random_image(res)
        coh = compute_local_gradient_coherence(img)
        random_coherences.append(coh)

    cppn_coherences = np.array(cppn_coherences)
    random_coherences = np.array(random_coherences)

    # Statistical tests (two-sided to detect difference in either direction)
    u_stat, p_value = mannwhitneyu(cppn_coherences, random_coherences, alternative='two-sided')
    d = cohens_d(cppn_coherences, random_coherences)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nCPPN coherence: mean={np.mean(cppn_coherences):.4f}, std={np.std(cppn_coherences):.4f}")
    print(f"Random coherence: mean={np.mean(random_coherences):.4f}, std={np.std(random_coherences):.4f}")
    print(f"Ratio (CPPN/random): {np.mean(cppn_coherences)/np.mean(random_coherences):.2f}x")
    print(f"\nMann-Whitney U: {u_stat:.1f}")
    print(f"P-value (two-sided): {p_value:.2e}")
    print(f"Cohen's d: {d:.3f}")

    # Determine outcome
    if p_value < 0.01 and d > 0.5:
        status = "validated"
        summary = f"CPPN images have significantly higher local gradient coherence ({np.mean(cppn_coherences):.3f}) than random ({np.mean(random_coherences):.3f}). Effect size d={d:.2f}, p={p_value:.1e}."
    elif p_value < 0.01 and d < -0.5:
        status = "refuted"
        summary = f"CPPN images have LOWER local gradient coherence ({np.mean(cppn_coherences):.3f}) than random ({np.mean(random_coherences):.3f}). Effect size d={d:.2f}, p={p_value:.1e}. CPPN images have smoother regions with less local texture variation, leading to lower structure tensor coherence."
    else:
        status = "inconclusive"
        summary = f"Gradient coherence difference is small or not significant. d={d:.2f}, p={p_value:.1e}."

    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Save results
    results = {
        "experiment_id": "RES-142",
        "hypothesis": "CPPN images have higher local gradient coherence than random images",
        "domain": "image_statistics",
        "status": status,
        "metrics": {
            "cppn_mean": float(np.mean(cppn_coherences)),
            "cppn_std": float(np.std(cppn_coherences)),
            "random_mean": float(np.mean(random_coherences)),
            "random_std": float(np.std(random_coherences)),
            "cohens_d": float(d),
            "p_value": float(p_value),
            "n_samples": n_samples
        },
        "summary": summary
    }

    import os
    os.makedirs("/Users/matt/Development/monochrome_noise_converger/results", exist_ok=True)
    with open("/Users/matt/Development/monochrome_noise_converger/results/res_142_gradient_coherence.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/res_142_gradient_coherence.json")

    return status, summary, d, p_value


if __name__ == "__main__":
    main()
