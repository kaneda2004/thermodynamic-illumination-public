#!/usr/bin/env python3
"""
Thermodynamic Illumination: The Shield of Structure
Experiment: Deep Image Prior (DIP) Dynamics

Hypothesis:
- Low-Bit Architectures (ResNet) act as an implicit regularizer, fitting signal before noise.
- High-Bit Architectures (ViT) lack this shield, fitting noise immediately.

This validates that 'Thermodynamic Volume' predicts 'Generalization Gap'.

Usage:
    uv run python experiments/dip_dynamics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import io
from PIL import Image, ImageDraw

# ==========================================
# 1. SETUP DATA (Synthetic Ground Truth)
# ==========================================

def get_data(size=64, noise_level=0.15):
    """Create a structured test image with geometric shapes."""
    img = Image.new('RGB', (size, size), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)

    # Red Rectangle
    draw.rectangle([10, 10, 30, 50], fill=(255, 50, 50))
    # Blue Circle
    draw.ellipse([30, 20, 55, 45], fill=(50, 100, 255))
    # Yellow diagonal line
    draw.line([0, 60, 64, 0], fill=(255, 255, 0), width=3)
    # Green small square
    draw.rectangle([45, 45, 58, 58], fill=(50, 200, 50))

    # Convert to Tensor [1, 3, H, W]
    x_clean = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    x_clean = x_clean.unsqueeze(0)

    # Add Gaussian noise
    noise = torch.randn_like(x_clean) * noise_level
    x_noisy = torch.clamp(x_clean + noise, 0, 1)

    return x_clean, x_noisy


# ==========================================
# 2. ARCHITECTURES
# ==========================================

class ResNetGen(nn.Module):
    """Convolutional Generator (Low-Bit Prior)"""
    def __init__(self, channels=3):
        super().__init__()
        self.input_shape = (1, 128, 4, 4)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(128, 128),  # 8x8
            block(128, 64),   # 16x16
            block(64, 32),    # 32x32
            block(32, 16),    # 64x64
            nn.Conv2d(16, channels, 3, padding=1),
            nn.Sigmoid()
        )
        # Learnable seed (the "latent code" being optimized)
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class ViTGen(nn.Module):
    """Vision Transformer Generator (High-Bit Prior)"""
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2

        # Learnable positional embeddings (the "latent code")
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_pixels = nn.Linear(dim, 3 * patch_size * patch_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        x = self.transformer(self.pos_embed)
        x = self.to_pixels(x)

        b, n, p = x.shape
        h = w = int(math.sqrt(n))
        x = x.view(b, h, w, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return self.sigmoid(x)


class MLPGen(nn.Module):
    """MLP Generator (Maximum Entropy Prior)"""
    def __init__(self, img_size=64):
        super().__init__()
        self.img_size = img_size
        # Learnable latent code
        self.latent = nn.Parameter(torch.randn(1, 256))

        self.net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self):
        out = self.net(self.latent)
        return out.view(1, 3, self.img_size, self.img_size)


# ==========================================
# 3. METRICS
# ==========================================

def get_structure_score(img_tensor):
    """Combined JPEG compressibility + TV smoothness score."""
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Compressibility
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    # TV smoothness
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_score = math.exp(-10 * (tv_h + tv_w).item())

    return comp_score * tv_score


def psnr(pred, target):
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


# ==========================================
# 4. TRAINING LOOP
# ==========================================

def optimize_prior(name, model, target_noisy, target_clean, steps=2000, lr=0.01):
    """
    Train the network to reconstruct the NOISY image.
    Track performance against CLEAN image (generalization).
    """
    print(f"\n{'='*50}")
    print(f"Optimizing: {name}")
    print(f"{'='*50}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'loss': [],           # Training loss (vs noisy)
        'mse_clean': [],      # Generalization (vs clean)
        'psnr_clean': [],     # PSNR vs clean
        'structure': []       # Structure score
    }

    best_img = None
    best_psnr = -float('inf')

    for i in range(steps):
        optimizer.zero_grad()
        out = model()

        # Loss is ONLY against noisy target (simulating overfitting scenario)
        loss = nn.MSELoss()(out, target_noisy)
        loss.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            mse_clean = nn.MSELoss()(out, target_clean).item()
            psnr_clean = psnr(out, target_clean)
            structure = get_structure_score(out)

            history['loss'].append(loss.item())
            history['mse_clean'].append(mse_clean)
            history['psnr_clean'].append(psnr_clean)
            history['structure'].append(structure)

            if psnr_clean > best_psnr:
                best_psnr = psnr_clean
                best_img = out.detach().clone()
                best_iter = i

        if i % 400 == 0:
            print(f"  Step {i:4d}: Loss={loss.item():.5f} | "
                  f"Clean MSE={mse_clean:.5f} | PSNR={psnr_clean:.2f}dB | "
                  f"Structure={structure:.3f}")

    print(f"  Best PSNR: {best_psnr:.2f}dB at iteration {best_iter}")

    return history, best_img


# ==========================================
# 5. VISUALIZATION
# ==========================================

def plot_results(histories, images, clean, noisy, save_dir):
    """Create comprehensive visualization of DIP dynamics."""

    # Figure 1: Training Dynamics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'ResNet': '#2ecc71', 'ViT': '#9b59b6', 'MLP': '#e74c3c'}

    # A. Training Loss (vs Noisy)
    ax = axes[0, 0]
    for name, hist in histories.items():
        ax.plot(hist['loss'], label=name, color=colors[name], linewidth=2)
    ax.set_title('Training Loss (vs Noisy Target)', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # B. Generalization (vs Clean) - THE KEY METRIC
    ax = axes[0, 1]
    for name, hist in histories.items():
        ax.plot(hist['mse_clean'], label=name, color=colors[name], linewidth=2)
    ax.set_title('Generalization: MSE vs Clean Image', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE (Lower = Better Denoising)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add annotation showing the "overfitting point"
    ax.axhline(y=0.15**2, color='gray', linestyle='--', alpha=0.5, label='Noise Floor')

    # C. PSNR vs Clean
    ax = axes[1, 0]
    for name, hist in histories.items():
        ax.plot(hist['psnr_clean'], label=name, color=colors[name], linewidth=2)
    ax.set_title('Denoising Quality (PSNR vs Clean)', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB, Higher = Better)')
    ax.legend()
    ax.grid(alpha=0.3)

    # D. Structure Score
    ax = axes[1, 1]
    for name, hist in histories.items():
        ax.plot(hist['structure'], label=name, color=colors[name], linewidth=2)
    ax.set_title('Structure Score (Thermodynamic Order)', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Structure (Higher = More Ordered)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'dip_dynamics.png', dpi=150)
    plt.savefig(save_dir / 'dip_dynamics.pdf')
    print(f"Saved: {save_dir / 'dip_dynamics.png'}")
    plt.close()

    # Figure 2: Visual Comparison
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Top row: targets and results
    axes[0, 0].imshow(clean.squeeze().permute(1, 2, 0).numpy())
    axes[0, 0].set_title('Ground Truth (Clean)', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy.squeeze().permute(1, 2, 0).numpy())
    axes[0, 1].set_title('Input (Noisy)', fontsize=11)
    axes[0, 1].axis('off')

    # Placeholder for difference
    diff = torch.abs(noisy - clean)
    axes[0, 2].imshow(diff.squeeze().permute(1, 2, 0).numpy() * 3)  # Amplify for visibility
    axes[0, 2].set_title('Noise (3x amplified)', fontsize=11)
    axes[0, 2].axis('off')

    # Bottom row: reconstructions
    for idx, (name, img) in enumerate(images.items()):
        ax = axes[1, idx]
        ax.imshow(img.squeeze().permute(1, 2, 0).numpy())
        psnr_val = psnr(img, clean)
        ax.set_title(f'{name}\nPSNR: {psnr_val:.1f}dB', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'dip_visual_comparison.png', dpi=150)
    print(f"Saved: {save_dir / 'dip_visual_comparison.png'}")
    plt.close()


def plot_summary_figure(histories, images, clean, noisy, save_dir):
    """Create a single summary figure for the paper."""
    fig = plt.figure(figsize=(10, 5))

    # Left: Generalization curve (the key result)
    ax1 = fig.add_subplot(1, 2, 1)
    colors = {'ResNet': '#2ecc71', 'ViT': '#9b59b6', 'MLP': '#e74c3c'}

    for name, hist in histories.items():
        ax1.plot(hist['mse_clean'], label=name, color=colors[name], linewidth=2.5)

    ax1.set_title('The Shield of Structure:\nGeneralization During Optimization', fontsize=11)
    ax1.set_xlabel('Optimization Step', fontsize=10)
    ax1.set_ylabel('MSE vs Clean Image (Lower = Better)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Add annotations
    resnet_min_idx = np.argmin(histories['ResNet']['mse_clean'])
    resnet_min_val = histories['ResNet']['mse_clean'][resnet_min_idx]
    ax1.annotate(f'ResNet best\n(early stop)',
                xy=(resnet_min_idx, resnet_min_val),
                xytext=(resnet_min_idx + 300, resnet_min_val + 0.005),
                arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                fontsize=9, color='#2ecc71')

    # Right: Visual comparison
    ax2 = fig.add_subplot(1, 2, 2)
    # Create 2x2 grid: clean, noisy, resnet, vit
    grid = np.zeros((128, 128, 3))
    grid[:64, :64] = clean.squeeze().permute(1, 2, 0).numpy()
    grid[:64, 64:] = noisy.squeeze().permute(1, 2, 0).numpy()
    grid[64:, :64] = images['ResNet'].squeeze().permute(1, 2, 0).numpy()
    grid[64:, 64:] = images['ViT'].squeeze().permute(1, 2, 0).numpy()

    ax2.imshow(grid)
    ax2.axhline(y=64, color='white', linewidth=2)
    ax2.axvline(x=64, color='white', linewidth=2)
    ax2.text(32, 8, 'Clean', ha='center', fontsize=9, color='white')
    ax2.text(96, 8, 'Noisy', ha='center', fontsize=9, color='white')
    ax2.text(32, 72, f'ResNet\n{psnr(images["ResNet"], clean):.1f}dB',
             ha='center', fontsize=9, color='white')
    ax2.text(96, 72, f'ViT\n{psnr(images["ViT"], clean):.1f}dB',
             ha='center', fontsize=9, color='white')
    ax2.set_title('Visual Comparison (clean/noisy/ResNet/ViT)', fontsize=11)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'dip_summary.png', dpi=150)
    plt.savefig(save_dir / 'dip_summary.pdf')
    print(f"Saved: {save_dir / 'dip_summary.png'}")
    plt.close()


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def run_dip_experiment(steps=2000, noise_level=0.15):
    """Run the complete DIP dynamics experiment."""
    print("=" * 60)
    print("Thermodynamic Illumination: The Shield of Structure")
    print("Deep Image Prior Dynamics Experiment")
    print("=" * 60)
    print(f"\nHypothesis:")
    print("  Low-Bit (ResNet): Implicit regularizer, fits signal before noise")
    print("  High-Bit (ViT/MLP): No shield, fits noise immediately")
    print(f"\nNoise level: {noise_level}")
    print(f"Optimization steps: {steps}")

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Generate data
    clean, noisy = get_data(noise_level=noise_level)
    print(f"\nTarget images generated (64x64 RGB)")
    print(f"  Clean PSNR vs Noisy: {psnr(noisy, clean):.2f}dB")

    # Run optimization for each architecture
    histories = {}
    images = {}

    # ResNet
    model = ResNetGen()
    hist, img = optimize_prior("ResNet", model, noisy, clean, steps=steps)
    histories['ResNet'] = hist
    images['ResNet'] = img

    # ViT
    model = ViTGen()
    hist, img = optimize_prior("ViT", model, noisy, clean, steps=steps)
    histories['ViT'] = hist
    images['ViT'] = img

    # MLP
    model = MLPGen()
    hist, img = optimize_prior("MLP", model, noisy, clean, steps=steps)
    histories['MLP'] = hist
    images['MLP'] = img

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    plot_results(histories, images, clean, noisy, out_dir)
    plot_summary_figure(histories, images, clean, noisy, out_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for name in ['ResNet', 'ViT', 'MLP']:
        hist = histories[name]
        best_psnr = max(hist['psnr_clean'])
        best_iter = hist['psnr_clean'].index(best_psnr)
        final_psnr = hist['psnr_clean'][-1]

        print(f"\n{name}:")
        print(f"  Best PSNR: {best_psnr:.2f}dB at iteration {best_iter}")
        print(f"  Final PSNR: {final_psnr:.2f}dB")
        print(f"  Degradation: {best_psnr - final_psnr:.2f}dB (overfitting)")

    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)

    resnet_best = max(histories['ResNet']['psnr_clean'])
    vit_best = max(histories['ViT']['psnr_clean'])

    if resnet_best > vit_best + 1.0:
        print("CONFIRMED: ResNet achieves better denoising than ViT")
        print(f"  ResNet: {resnet_best:.2f}dB")
        print(f"  ViT: {vit_best:.2f}dB")
        print(f"  Gap: {resnet_best - vit_best:.2f}dB")
        print("\n  -> Thermodynamic Volume PREDICTS Generalization!")
    else:
        print("UNEXPECTED: ViT performs similarly to ResNet")
        print("  This may indicate the noise level is too low")

    return histories, images


if __name__ == "__main__":
    histories, images = run_dip_experiment(steps=2000, noise_level=0.15)
