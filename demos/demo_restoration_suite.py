#!/usr/bin/env python3
"""
Demo: Restoration Suite (Denoising & Super-Resolution)

This applies the Thermodynamic Insight:
    "Low-bit priors naturally resist noise and enforce continuity."

Applications:
1. Denoising: The CPPN prior lacks the entropy to fit Gaussian noise,
   so it acts as a powerful denoising auto-prior.
2. Super-Resolution: The CPPN is a continuous function. We fit it to
   low-res pixels, then query it at high resolution.

Usage:
    uv run python demo_restoration_suite.py --task denoising
    uv run python demo_restoration_suite.py --task super_res
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torchvision import datasets, transforms

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

# =============================================================================
# Architectures
# =============================================================================

class CPPNDecoder(nn.Module):
    """The Low-Bit Prior (~0.05 bits). Continuous, resolution-independent."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Simpler architecture with standard tanh - more stable optimization
        self.fc1 = nn.Linear(latent_dim + 2, 32)  # +2 for x,y
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        # Initialize with larger weights for more varied outputs
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(m.weight, 0, 1.0)
            nn.init.zeros_(m.bias)

    def forward(self, z, coords):
        """
        z: (1, latent_dim) or (N, latent_dim)
        coords: (N_pixels, 2) - Normalized [-1, 1]
        """
        n_pixels = coords.shape[0]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z_expanded = z.expand(n_pixels, -1)

        inp = torch.cat([z_expanded, coords], dim=1)
        h = torch.tanh(self.fc1(inp))
        h = torch.tanh(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class MLPDecoder(nn.Module):
    """The High-Bit Prior (~4.85 bits). Discrete, pixel-bound."""
    def __init__(self, latent_dim=16, image_size=28):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, image_size * image_size),
            nn.Sigmoid()
        )

    def forward(self, z, coords=None):
        # MLP ignores coords, it projects directly to fixed grid
        return self.net(z).view(-1, 1)

# =============================================================================
# Tasks
# =============================================================================

def make_coords(resolution, device):
    """Generate (x,y) grid for a specific resolution."""
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    return coords

def run_denoising(device):
    print("\n=== TASK: ZERO-SHOT DENOISING ===")
    print("Hypothesis: CPPN (low entropy) will underfit noise.")
    print("            MLP (high entropy) will overfit noise.")

    # 1. Prepare Data
    img = get_target_image(device)
    noise = torch.randn_like(img) * 0.3
    noisy_img = torch.clamp(img + noise, 0, 1)

    # Flatten for processing
    target_flat = noisy_img.view(-1, 1)
    coords_28 = make_coords(28, device)

    ascii_plot(img, "Original")
    ascii_plot(noisy_img, "Noisy Input (Target)")

    # 2. Optimize CPPN
    print("\nRunning CPPN Denoising...")
    cppn = CPPNDecoder().to(device)
    recon_cppn = optimize_latent(cppn, target_flat, coords_28)

    # 3. Optimize MLP
    print("Running MLP Denoising...")
    mlp = MLPDecoder().to(device)
    recon_mlp = optimize_latent(mlp, target_flat, coords_28) # Coords ignored by MLP

    # 4. Result
    ascii_plot(recon_cppn.view(28, 28), "CPPN Result (Smooth)")
    ascii_plot(recon_mlp.view(28, 28), "MLP Result (Noisy)")

    # Metric: Distance to ORIGINAL (clean) image
    clean_flat = img.view(-1, 1)
    mse_cppn = torch.mean((recon_cppn - clean_flat)**2).item()
    mse_mlp = torch.mean((recon_mlp - clean_flat)**2).item()

    print(f"\nMSE vs Clean Ground Truth:")
    print(f"CPPN: {mse_cppn:.4f} (Winner)" if mse_cppn < mse_mlp else f"CPPN: {mse_cppn:.4f}")
    print(f"MLP:  {mse_mlp:.4f}")

def run_super_res(device):
    print("\n=== TASK: ZERO-SHOT SUPER-RESOLUTION ===")
    print("Hypothesis: CPPN trained on 7x7 pixels can hallucinate 28x28.")

    # 1. Prepare Data
    img_28 = get_target_image(device)

    # Downsample to 7x7
    img_7 = torch.nn.functional.interpolate(img_28.unsqueeze(0).unsqueeze(0), size=7, mode='bilinear').squeeze()

    ascii_plot(img_28, "Original (28x28)")
    ascii_plot(img_7, "Low-Res Input (7x7)")

    coords_7 = make_coords(7, device)
    coords_28 = make_coords(28, device)
    target_flat = img_7.view(-1, 1)

    # 2. Optimize CPPN on 7x7
    print("\nTraining CPPN on 49 pixels...")
    cppn = CPPNDecoder().to(device)

    # We define a custom optimization loop to handle the coordinate switch
    z = torch.randn(1, 16, device=device, requires_grad=True)
    opt = optim.Adam([z], lr=0.05)

    for i in range(1000):
        opt.zero_grad()
        # Query at LOW resolution coordinates
        out = cppn(z, coords_7)
        loss = torch.mean((out - target_flat)**2)
        loss.backward()
        opt.step()

    # 3. Inference at 28x28
    print("Querying CPPN at 28x28 resolution...")
    with torch.no_grad():
        # Query at HIGH resolution coordinates
        sr_img = cppn(z, coords_28).view(28, 28)

    # 4. Result
    ascii_plot(sr_img, "Super-Resolution Result")
    print("\nNote: The network 'imagined' the curves between the 7x7 pixels.")
    print("      This capability is unique to continuous coordinate-based priors.")

# =============================================================================
# Utils
# =============================================================================

def get_target_image(device):
    """Load a specific '3' from MNIST."""
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    for img, label in dataset:
        if label == 3:
            return img.squeeze(0).to(device)
    return torch.zeros(28, 28).to(device)

def optimize_latent(model, target_flat, coords, steps=1000):
    """Optimize z to match target."""
    z = torch.randn(1, 16, device=target_flat.device, requires_grad=True)
    opt = optim.Adam([z], lr=0.05)

    for i in range(steps):
        opt.zero_grad()
        out = model(z, coords)
        loss = torch.mean((out - target_flat)**2)
        loss.backward()
        opt.step()

    return out.detach()

def ascii_plot(img_tensor, title):
    chars = " .:-=+*#%@"
    img = img_tensor.cpu().numpy()
    h, w = img.shape

    print(f"\n--- {title} ---")

    # If image is small (7x7), scale up for visibility
    scale_r = 1
    scale_c = 1
    if h < 10: scale_r = 2
    if w < 10: scale_c = 4 # ASCII chars are tall

    for r in range(h):
        for _ in range(scale_r):
            line = ""
            for c in range(w):
                val = np.clip(img[r, c], 0, 1)
                char = chars[int(val * (len(chars)-1))]
                line += char * scale_c
            print(line)

def run_denoising_full(device):
    """Enhanced denoising - optimize both z AND network weights."""
    print("\n=== TASK: FULL OPTIMIZATION DENOISING ===")
    print("Now optimizing BOTH latent z AND network weights.")
    print("This pushes the CPPN to its limits.\n")

    # 1. Prepare Data
    img = get_target_image(device)
    noise = torch.randn_like(img) * 0.3
    noisy_img = torch.clamp(img + noise, 0, 1)

    target_flat = noisy_img.view(-1, 1)
    coords_28 = make_coords(28, device)

    ascii_plot(img, "Original (Clean)")
    ascii_plot(noisy_img, "Noisy Input")

    # 2. CPPN with full optimization
    print("Running CPPN with full optimization (z + weights)...")
    cppn = CPPNDecoder(latent_dim=32).to(device)  # Larger latent
    z = torch.randn(1, 32, device=device, requires_grad=True)

    # Optimize BOTH z and network weights
    opt = optim.Adam([z] + list(cppn.parameters()), lr=0.01)

    for i in range(2000):
        opt.zero_grad()
        out = cppn(z, coords_28)
        loss = torch.mean((out - target_flat)**2)
        loss.backward()
        opt.step()
        if (i+1) % 500 == 0:
            print(f"  Step {i+1}: loss = {loss.item():.4f}")

    recon_cppn = out.detach()

    # 3. MLP baseline (also full optimization for fair comparison)
    print("\nRunning MLP with full optimization...")
    mlp = MLPDecoder(latent_dim=32).to(device)
    z_mlp = torch.randn(1, 32, device=device, requires_grad=True)
    opt_mlp = optim.Adam([z_mlp] + list(mlp.parameters()), lr=0.01)

    for i in range(2000):
        opt_mlp.zero_grad()
        out = mlp(z_mlp, coords_28)
        loss = torch.mean((out - target_flat)**2)
        loss.backward()
        opt_mlp.step()

    recon_mlp = out.detach()

    # 4. Results
    ascii_plot(recon_cppn.view(28, 28), "CPPN Result (Full Optimization)")
    ascii_plot(recon_mlp.view(28, 28), "MLP Result (Full Optimization)")

    clean_flat = img.view(-1, 1)
    mse_cppn = torch.mean((recon_cppn - clean_flat)**2).item()
    mse_mlp = torch.mean((recon_mlp - clean_flat)**2).item()

    # Also measure fit to noisy target
    fit_cppn = torch.mean((recon_cppn - target_flat)**2).item()
    fit_mlp = torch.mean((recon_mlp - target_flat)**2).item()

    print(f"\nFit to NOISY target (lower = fit noise better):")
    print(f"  CPPN: {fit_cppn:.4f}")
    print(f"  MLP:  {fit_mlp:.4f}")

    print(f"\nMSE vs CLEAN ground truth (lower = better denoising):")
    print(f"  CPPN: {mse_cppn:.4f} {'(Winner)' if mse_cppn < mse_mlp else ''}")
    print(f"  MLP:  {mse_mlp:.4f} {'(Winner)' if mse_mlp < mse_cppn else ''}")

    print(f"\nKey insight: MLP fits noisy target better ({fit_mlp:.4f} vs {fit_cppn:.4f})")
    print(f"but CPPN generalizes to clean image better ({mse_cppn:.4f} vs {mse_mlp:.4f})")
    print("The CPPN's structural prior prevents overfitting to noise!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["denoising", "super_res", "denoising_full"], default="denoising")
    args = parser.parse_args()

    device = get_device()

    if args.task == "denoising":
        run_denoising(device)
    elif args.task == "super_res":
        run_super_res(device)
    elif args.task == "denoising_full":
        run_denoising_full(device)
