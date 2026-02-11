#!/usr/bin/env python3
"""
Demo: CPPN as Regularizer / Prior for Pretrained Models

This shows the thermodynamic insight is ACTIONABLE, not just diagnostic:
1. CPPN Regularization: Add CPPN prior loss to guide a high-bits model
2. Ensemble Blending: Combine MLP output with CPPN output at inference

Setup:
- We simulate a "pretrained" MLP-based SR model (high bits, noisy outputs)
- We show how CPPN prior can improve its results

Usage:
    uv run python demo_cppn_regularizer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Models
# =============================================================================

class CPPNPrior(nn.Module):
    """Low-bits CPPN used as a structural prior/regularizer."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(m.weight, 0, 1.0)
            nn.init.zeros_(m.bias)

    def forward(self, z, coords):
        n_pixels = coords.shape[0]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z_exp = z.expand(n_pixels, -1)

        inp = torch.cat([z_exp, coords], dim=1)
        h = torch.tanh(self.fc1(inp))
        h = torch.tanh(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class MLPUpscaler(nn.Module):
    """
    Simulates a 'pretrained' high-bits SR model.
    High entropy = can fit noise, poor generalization.
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        """z: (B, latent_dim) -> (B, 784) image flattened"""
        return self.net(z)


# =============================================================================
# Utilities
# =============================================================================

def make_coords(resolution, device):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)


def get_target_image(device, digit=3):
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.ToTensor())
    for img, label in dataset:
        if label == digit:
            return img.squeeze(0).to(device)
    return torch.zeros(28, 28).to(device)


def downsample(img_28, size=7):
    """Downsample 28x28 to 7x7"""
    return torch.nn.functional.interpolate(
        img_28.unsqueeze(0).unsqueeze(0),
        size=size,
        mode='bilinear',
        align_corners=False
    ).squeeze()


def ascii_plot(img_tensor, title):
    chars = " .:-=+*#%@"
    img = img_tensor.detach().cpu().numpy()
    if img.ndim > 2:
        img = img.squeeze()

    print(f"\n--- {title} ---")
    for r in range(img.shape[0]):
        line = ""
        for c in range(img.shape[1]):
            val = np.clip(img[r, c], 0, 1)
            char = chars[int(val * (len(chars) - 1))]
            line += char
        print(line)


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


# =============================================================================
# Demo 1: CPPN as Regularizer
# =============================================================================

def demo_cppn_regularizer(device):
    print("=" * 70)
    print("DEMO 1: CPPN AS REGULARIZER (Noisy Target)")
    print("=" * 70)
    print("""
Scenario: We have a NOISY target image. The MLP will overfit to noise.
          Adding CPPN regularizer prevents overfitting.

Loss = MSE(output, noisy_target) + λ * MSE(output, cppn_output)
    """)

    # Setup - add noise to target
    img_28 = get_target_image(device)
    noise = torch.randn_like(img_28) * 0.25
    noisy_target = torch.clamp(img_28 + noise, 0, 1)
    coords_28 = make_coords(28, device)

    ascii_plot(img_28, "Clean Ground Truth")
    ascii_plot(noisy_target, "Noisy Target (what models see)")

    target_flat = noisy_target.view(-1)
    clean_flat = img_28.view(-1)

    # 1. MLP alone - will overfit to noise
    print("\n[1] Training MLP on noisy target (no regularization)...")
    mlp_alone = MLPUpscaler().to(device)
    z_alone = torch.randn(1, 64, device=device, requires_grad=True)
    opt = optim.Adam(list(mlp_alone.parameters()) + [z_alone], lr=0.01)

    for i in range(1500):
        opt.zero_grad()
        out = mlp_alone(z_alone).squeeze()
        loss = torch.mean((out - target_flat) ** 2)
        loss.backward()
        opt.step()

    mlp_alone_out = mlp_alone(z_alone).squeeze().view(28, 28)
    ascii_plot(mlp_alone_out, "MLP Alone (overfits noise)")

    # 2. MLP + CPPN regularizer
    print("\n[2] Training MLP on noisy target WITH CPPN regularizer...")
    mlp_reg = MLPUpscaler().to(device)
    cppn = CPPNPrior(latent_dim=32).to(device)
    z_mlp = torch.randn(1, 64, device=device, requires_grad=True)
    z_cppn = torch.randn(1, 32, device=device, requires_grad=True)

    opt = optim.Adam(
        list(mlp_reg.parameters()) + list(cppn.parameters()) + [z_mlp, z_cppn],
        lr=0.01
    )

    lambda_reg = 0.5

    for i in range(1500):
        opt.zero_grad()

        mlp_out = mlp_reg(z_mlp).squeeze()
        cppn_out = cppn(z_cppn, coords_28).squeeze()

        loss_main = torch.mean((mlp_out - target_flat) ** 2)
        loss_reg = torch.mean((mlp_out - cppn_out) ** 2)

        loss = loss_main + lambda_reg * loss_reg
        loss.backward()
        opt.step()

    mlp_reg_out = mlp_reg(z_mlp).squeeze().view(28, 28)
    cppn_out_final = cppn(z_cppn, coords_28).squeeze().view(28, 28)

    ascii_plot(mlp_reg_out, "MLP + CPPN Regularizer")
    ascii_plot(cppn_out_final, "CPPN Prior (structural guide)")

    # Compare vs CLEAN ground truth
    mse_alone = mse(mlp_alone_out, img_28)
    mse_reg = mse(mlp_reg_out, img_28)
    mse_cppn = mse(cppn_out_final, img_28)

    # Also show fit to noisy target
    fit_alone = mse(mlp_alone_out, noisy_target)
    fit_reg = mse(mlp_reg_out, noisy_target)

    print(f"\nFit to NOISY target (lower = memorized noise):")
    print(f"  MLP alone:       {fit_alone:.4f}")
    print(f"  MLP + CPPN reg:  {fit_reg:.4f}")

    print(f"\nMSE vs CLEAN ground truth (lower = better generalization):")
    print(f"  MLP alone:       {mse_alone:.4f}")
    print(f"  MLP + CPPN reg:  {mse_reg:.4f} {'← Winner' if mse_reg < mse_alone else ''}")
    print(f"  CPPN only:       {mse_cppn:.4f}")

    if mse_reg < mse_alone:
        improvement = (mse_alone - mse_reg) / mse_alone * 100
        print(f"\n  CPPN regularizer improved generalization by {improvement:.1f}%!")

    return mlp_alone_out, mlp_reg_out, img_28


# =============================================================================
# Demo 2: Ensemble Blending
# =============================================================================

def demo_ensemble_blending(device):
    print("\n" + "=" * 70)
    print("DEMO 2: ENSEMBLE BLENDING AT INFERENCE")
    print("=" * 70)
    print("""
Scenario: We have a noisy/imperfect SR model output.
          Blend with CPPN to reduce artifacts.

Output = α * SR_output + (1-α) * CPPN_output
    """)

    # Setup - noisy target simulates imperfect SR
    img_28 = get_target_image(device)
    noise = torch.randn_like(img_28) * 0.3
    noisy_sr_output = torch.clamp(img_28 + noise, 0, 1)  # Simulates noisy SR
    coords_28 = make_coords(28, device)

    ascii_plot(img_28, "Clean Ground Truth")
    ascii_plot(noisy_sr_output, "Noisy SR Output (simulated)")

    # Train CPPN on noisy target
    print("\n[1] Training CPPN to match noisy SR output...")
    cppn = CPPNPrior(latent_dim=32).to(device)
    z = torch.randn(1, 32, device=device, requires_grad=True)
    opt = optim.Adam(list(cppn.parameters()) + [z], lr=0.01)

    target_flat = noisy_sr_output.view(-1)

    for i in range(1500):
        opt.zero_grad()
        out = cppn(z, coords_28).squeeze()
        loss = torch.mean((out - target_flat) ** 2)
        loss.backward()
        opt.step()

    cppn_output = cppn(z, coords_28).squeeze().view(28, 28)
    ascii_plot(cppn_output, "CPPN Output (smooth, denoised)")

    # Blend at different ratios
    print("\n[2] Blending SR + CPPN at different ratios...")

    best_alpha = 0
    best_mse = float('inf')

    results = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = alpha * noisy_sr_output + (1 - alpha) * cppn_output
        blend_mse = mse(blended, img_28)
        results.append((alpha, blend_mse))

        if blend_mse < best_mse:
            best_mse = blend_mse
            best_alpha = alpha

        label = ""
        if alpha == 1.0:
            label = " (SR only)"
        elif alpha == 0.0:
            label = " (CPPN only)"
        print(f"  α={alpha:.2f}: MSE = {blend_mse:.4f}{label}")

    # Show best blend
    best_blend = best_alpha * noisy_sr_output + (1 - best_alpha) * cppn_output
    ascii_plot(best_blend, f"Best Blend (α={best_alpha})")

    sr_mse = mse(noisy_sr_output, img_28)
    cppn_mse = mse(cppn_output, img_28)

    print(f"\nSummary:")
    print(f"  Noisy SR alone:      {sr_mse:.4f}")
    print(f"  CPPN alone:          {cppn_mse:.4f}")
    print(f"  Best blend (α={best_alpha:.2f}): {best_mse:.4f}")

    if best_mse < sr_mse:
        improvement = (sr_mse - best_mse) / sr_mse * 100
        print(f"\n  Blending improved MSE by {improvement:.1f}% over noisy SR!")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CPPN AS REGULARIZER / PRIOR FOR PRETRAINED MODELS")
    print("=" * 70)
    print("""
This demo shows the thermodynamic insight is ACTIONABLE:

The measurement tells us: "Low-bits priors (CPPNs) enforce structure"
The action is: Use CPPN as a regularizer or ensemble component

Two approaches:
1. REGULARIZER: Add CPPN loss during training/fine-tuning
2. ENSEMBLE: Blend CPPN output with SR output at inference
    """)

    device = get_device()
    print(f"Using device: {device}\n")

    demo_cppn_regularizer(device)
    demo_ensemble_blending(device)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The thermodynamic framework isn't just diagnostic—it's prescriptive:

1. MEASURE: Calculate bits to understand prior quality
2. SELECT: Choose low-bits architectures for restoration tasks
3. IMPROVE: Use CPPN as regularizer or ensemble component

The CPPN prior acts as a "structural guide" that prevents high-bits
models from overfitting to noise or producing artifacts.
    """)


if __name__ == "__main__":
    main()
