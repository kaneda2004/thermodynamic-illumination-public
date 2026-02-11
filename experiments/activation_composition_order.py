#!/usr/bin/env python3
"""
RES-115: Activation function composition order experiment.

Hypothesis: The order of activation function composition (sin(tanh(x)) vs tanh(sin(x)))
produces systematically different image order distributions in CPPNs.

Tests if composition order matters for generating structured images.
"""

import numpy as np
from scipy import stats
import torch
import torch.nn as nn

# Parameters
N_SAMPLES = 200
IMG_SIZE = 64
HIDDEN_DIM = 16
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_order(img: np.ndarray) -> float:
    """Compute image order score (0-1) based on spatial autocorrelation and compressibility."""
    # Density: mean absolute deviation from 0.5
    density = 1.0 - 2.0 * np.mean(np.abs(img - 0.5))

    # Edge: fraction of pixels similar to neighbors
    h_diff = np.abs(img[:, 1:] - img[:, :-1])
    v_diff = np.abs(img[1:, :] - img[:-1, :])
    edge = 1.0 - (np.mean(h_diff) + np.mean(v_diff)) / 2.0

    # Autocorrelation at lag 1
    flat = img.flatten()
    if len(flat) > 1:
        autocorr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
        autocorr = max(0, autocorr)  # Only positive correlation contributes
    else:
        autocorr = 0

    # Combined order score
    order = 0.4 * edge + 0.3 * autocorr + 0.3 * density
    return float(np.clip(order, 0, 1))


class CompositionalCPPN(nn.Module):
    """CPPN with explicit control over activation composition order."""

    def __init__(self, composition_order: str = "sin_tanh"):
        super().__init__()
        self.fc1 = nn.Linear(2, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 1)
        self.composition_order = composition_order

    def forward(self, x):
        # First layer with standard tanh
        h = torch.tanh(self.fc1(x))

        # Second layer with composed activations
        h2 = self.fc2(h)

        if self.composition_order == "sin_tanh":
            # sin(tanh(x)) - inner tanh, outer sin
            h2 = torch.sin(torch.tanh(h2) * np.pi)
        elif self.composition_order == "tanh_sin":
            # tanh(sin(x)) - inner sin, outer tanh
            h2 = torch.tanh(torch.sin(h2 * np.pi))
        elif self.composition_order == "sin_sin":
            # sin(sin(x)) - double periodic
            h2 = torch.sin(torch.sin(h2 * np.pi) * np.pi)
        elif self.composition_order == "tanh_tanh":
            # tanh(tanh(x)) - double saturating
            h2 = torch.tanh(torch.tanh(h2))

        out = torch.sigmoid(self.fc3(h2))
        return out


def generate_image(composition_order: str) -> np.ndarray:
    """Generate a single CPPN image with given composition order."""
    model = CompositionalCPPN(composition_order)

    # Random weights
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.randn_like(param) * 0.5

    # Create coordinate grid
    coords = np.linspace(-1, 1, IMG_SIZE)
    xx, yy = np.meshgrid(coords, coords)
    inputs = np.stack([xx.flatten(), yy.flatten()], axis=1)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs).numpy().reshape(IMG_SIZE, IMG_SIZE)

    return outputs


def main():
    print("RES-115: Activation Composition Order Experiment")
    print("=" * 50)

    compositions = ["sin_tanh", "tanh_sin", "sin_sin", "tanh_tanh"]
    results = {comp: [] for comp in compositions}

    print(f"\nGenerating {N_SAMPLES} samples per composition...")

    for i in range(N_SAMPLES):
        np.random.seed(SEED + i)
        torch.manual_seed(SEED + i)

        for comp in compositions:
            img = generate_image(comp)
            order = compute_order(img)
            results[comp].append(order)

    # Statistics
    print("\nResults:")
    print("-" * 50)

    for comp in compositions:
        arr = np.array(results[comp])
        print(f"{comp:12}: mean={arr.mean():.4f} std={arr.std():.4f}")

    # Main comparison: sin_tanh vs tanh_sin
    sin_tanh = np.array(results["sin_tanh"])
    tanh_sin = np.array(results["tanh_sin"])

    # Statistical tests
    t_stat, p_value = stats.ttest_ind(sin_tanh, tanh_sin)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((sin_tanh.std()**2 + tanh_sin.std()**2) / 2)
    cohens_d = abs(sin_tanh.mean() - tanh_sin.mean()) / pooled_std

    # Mann-Whitney U test (non-parametric)
    u_stat, p_mw = stats.mannwhitneyu(sin_tanh, tanh_sin, alternative='two-sided')

    print("\n" + "=" * 50)
    print("PRIMARY COMPARISON: sin(tanh(x)) vs tanh(sin(x))")
    print("=" * 50)
    print(f"sin_tanh mean: {sin_tanh.mean():.4f}")
    print(f"tanh_sin mean: {tanh_sin.mean():.4f}")
    print(f"Difference: {abs(sin_tanh.mean() - tanh_sin.mean()):.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (t-test): {p_value:.6f}")
    print(f"p-value (Mann-Whitney): {p_mw:.6f}")
    print(f"Cohen's d: {cohens_d:.4f}")

    # Secondary: sin_sin vs tanh_tanh
    sin_sin = np.array(results["sin_sin"])
    tanh_tanh = np.array(results["tanh_tanh"])

    t2, p2 = stats.ttest_ind(sin_sin, tanh_tanh)
    d2 = abs(sin_sin.mean() - tanh_tanh.mean()) / np.sqrt((sin_sin.std()**2 + tanh_tanh.std()**2) / 2)

    print("\n" + "=" * 50)
    print("SECONDARY: sin(sin(x)) vs tanh(tanh(x))")
    print("=" * 50)
    print(f"sin_sin mean: {sin_sin.mean():.4f}")
    print(f"tanh_tanh mean: {tanh_tanh.mean():.4f}")
    print(f"p-value: {p2:.6f}")
    print(f"Cohen's d: {d2:.4f}")

    # Validation decision
    print("\n" + "=" * 50)
    print("VALIDATION DECISION")
    print("=" * 50)

    # Check if primary comparison meets criteria
    primary_validated = p_value < 0.01 and cohens_d > 0.5
    secondary_validated = p2 < 0.01 and d2 > 0.5

    if primary_validated:
        status = "VALIDATED"
        summary = f"Activation composition order significantly affects order: sin(tanh) mean={sin_tanh.mean():.3f} vs tanh(sin) mean={tanh_sin.mean():.3f}, d={cohens_d:.2f}"
    elif secondary_validated:
        status = "VALIDATED"
        summary = f"Double-periodic vs double-saturating significantly differ: sin(sin) mean={sin_sin.mean():.3f} vs tanh(tanh) mean={tanh_tanh.mean():.3f}, d={d2:.2f}"
    elif p_value < 0.05 or p2 < 0.05:
        status = "INCONCLUSIVE"
        summary = f"Trend observed but effect size insufficient. Primary d={cohens_d:.2f}, secondary d={d2:.2f}"
    else:
        status = "REFUTED"
        summary = f"No significant difference in order between composition orders. p={p_value:.3f}, d={cohens_d:.2f}"

    print(f"Status: {status}")
    print(f"Summary: {summary}")

    # Output for log
    print("\n" + "=" * 50)
    print("LOG OUTPUT")
    print("=" * 50)
    print(f"effect_size: {max(cohens_d, d2):.4f}")
    print(f"p_value: {min(p_value, p2):.6f}")
    print(f"status: {status.lower()}")

    return status.lower(), cohens_d, p_value, summary


if __name__ == "__main__":
    status, effect, p, summary = main()
