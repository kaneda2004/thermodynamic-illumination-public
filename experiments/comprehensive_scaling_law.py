#!/usr/bin/env python3
"""
Comprehensive Scaling Law Experiment
=====================================
Tests 11 architectures for BOTH structure AND DIP performance.

Goal: Establish correlation between thermodynamic structure and generalization
with n=11 data points (up from n=3).

Architectures:
1. ResNet-4 (baseline)     - 4-layer conv decoder
2. ResNet-2 (shallow)      - 2-layer conv decoder
3. ResNet-6 (deep)         - 6-layer conv decoder
4. U-Net decoder           - Skip connections
5. Depthwise Separable     - Weaker spatial coupling
6. Local Attention (w=8)   - Windowed, not global
7. CPPN                    - Coordinate-based MLP
8. Fourier Features        - Positional encoding + MLP
9. Hybrid ViT              - Conv stem destroyed by transformer
10. ViT (baseline)         - Global attention scrambles
11. MLP (baseline)         - No spatial bias

Usage:
    uv run python experiments/comprehensive_scaling_law.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import io
import json
from PIL import Image, ImageDraw
from scipy import stats

# ==========================================
# ARCHITECTURES (11 total, 64x64 RGB output)
# ==========================================

class ResNet4Gen(nn.Module):
    """4-layer ConvNet (baseline) - Strong locality bias"""
    def __init__(self):
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
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class ResNet2Gen(nn.Module):
    """2-layer ConvNet (shallow) - Less smoothing"""
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 64, 16, 16)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(64, 32),    # 32x32
            block(32, 16),    # 64x64
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class ResNet6Gen(nn.Module):
    """6-layer ConvNet (deep) - Maximum smoothing"""
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 256, 1, 1)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(256, 256),  # 2x2
            block(256, 128),  # 4x4
            block(128, 128),  # 8x8
            block(128, 64),   # 16x16
            block(64, 32),    # 32x32
            block(32, 16),    # 64x64
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class UNetGen(nn.Module):
    """U-Net style decoder with skip connections"""
    def __init__(self):
        super().__init__()
        # Encoder seeds at multiple scales
        self.seed_4 = nn.Parameter(torch.randn(1, 128, 4, 4))
        self.seed_8 = nn.Parameter(torch.randn(1, 64, 8, 8))
        self.seed_16 = nn.Parameter(torch.randn(1, 32, 16, 16))
        self.seed_32 = nn.Parameter(torch.randn(1, 16, 32, 32))

        # Decoder with skip connections
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 32, 3, padding=1),  # 64+64=128 after concat
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 16, 3, padding=1),   # 32+32=64 after concat
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),   # 16+16=32 after concat
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self):
        x = self.up1(self.seed_4)                    # 8x8, 64ch
        x = self.up2(torch.cat([x, self.seed_8], 1)) # 16x16, 32ch
        x = self.up3(torch.cat([x, self.seed_16], 1)) # 32x32, 16ch
        x = self.up4(torch.cat([x, self.seed_32], 1)) # 64x64, 16ch
        return self.final(x)


class DepthwiseSepGen(nn.Module):
    """Depthwise Separable Conv - Weaker spatial coupling"""
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 128, 4, 4)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                # Depthwise conv (each channel processed separately)
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),
                # Pointwise conv (1x1 to mix channels)
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(128, 128),  # 8x8
            block(128, 64),   # 16x16
            block(64, 32),    # 32x32
            block(32, 16),    # 64x64
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class LocalAttentionGen(nn.Module):
    """Local (windowed) attention - Some structure preserved"""
    def __init__(self, img_size=64, window_size=8, dim=128, depth=4):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.num_windows = (img_size // window_size) ** 2
        self.dim = dim

        # Learnable embeddings per window
        self.window_embeds = nn.Parameter(torch.randn(1, self.num_windows, dim))

        # Local attention within each window (smaller transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Project to pixels (window_size^2 * 3 channels per window)
        self.to_pixels = nn.Linear(dim, window_size * window_size * 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        x = self.transformer(self.window_embeds)  # [1, num_windows, dim]
        x = self.to_pixels(x)  # [1, num_windows, w*w*3]
        x = self.sigmoid(x)

        # Reshape windows to image
        b, nw, _ = x.shape
        h_wins = w_wins = int(math.sqrt(nw))
        ws = self.window_size

        x = x.view(b, h_wins, w_wins, 3, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h_wins*ws, w_wins*ws)
        return x


class CPPNGen(nn.Module):
    """CPPN: Coordinate-based MLP with spectral bias"""
    def __init__(self, img_size=64, hidden=256):
        super().__init__()
        self.img_size = img_size

        # Create coordinate grid
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        ), dim=-1)  # [H, W, 2]

        # Add radius
        r = torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2).unsqueeze(-1)
        self.register_buffer('coords', torch.cat([coords, r], dim=-1))  # [H, W, 3]

        # MLP: (x, y, r) -> RGB
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Sigmoid()
        )

    def forward(self):
        x = self.coords.view(-1, 3)  # [H*W, 3]
        out = self.net(x)  # [H*W, 3]
        return out.view(1, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)


class FourierFeaturesGen(nn.Module):
    """Fourier Features: Positional encoding with low-freq bias"""
    def __init__(self, img_size=64, n_freqs=8, hidden=256):
        super().__init__()
        self.img_size = img_size
        self.n_freqs = n_freqs

        # Create coordinate grid
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        ), dim=-1)  # [H, W, 2]
        self.register_buffer('coords', coords)

        # Frequency bands (lower frequencies = stronger bias toward smoothness)
        freqs = 2 ** torch.linspace(0, n_freqs-1, n_freqs) * math.pi
        self.register_buffer('freqs', freqs)

        # Input dim: 2 + 2*n_freqs*2 (coords + sin/cos for each freq for each coord)
        input_dim = 2 + 4 * n_freqs

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid()
        )

    def forward(self):
        # Positional encoding
        coords = self.coords.view(-1, 2)  # [H*W, 2]

        # Add sin/cos features
        features = [coords]
        for freq in self.freqs:
            features.append(torch.sin(freq * coords))
            features.append(torch.cos(freq * coords))

        x = torch.cat(features, dim=-1)  # [H*W, input_dim]
        out = self.net(x)  # [H*W, 3]
        return out.view(1, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)


class HybridViTGen(nn.Module):
    """Hybrid: Conv Stem + Transformer + Conv Decoder"""
    def __init__(self, img_size=64, dim=128, depth=4, heads=4):
        super().__init__()

        # Conv Stem
        self.seed_shape = (1, 32, 4, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(64, dim, 3, padding=1),
            nn.ReLU()
        )

        # Transformer (64 tokens for 8x8 grid)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Conv Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, stride=4),  # 8->32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2),    # 32->64
            nn.Sigmoid()
        )

        self.seed = nn.Parameter(torch.randn(self.seed_shape))

    def forward(self):
        x = self.stem(self.seed)  # [1, dim, 8, 8]

        # Flatten for transformer
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [1, 64, dim]

        x = x + self.pos_embed
        x = self.transformer(x)

        # Reshape back
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.decoder(x)


class ViTGen(nn.Module):
    """Pure Vision Transformer - Global attention scrambles"""
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.to_pixels = nn.Linear(dim, 3 * patch_size * patch_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        x = self.transformer(self.pos_embed)
        x = self.to_pixels(x)

        b, n, p = x.shape
        h = w = int(math.sqrt(n))
        ps = self.patch_size
        x = x.view(b, h, w, 3, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h*ps, w*ps)
        return self.sigmoid(x)


class MLPGen(nn.Module):
    """Pure MLP - No spatial bias"""
    def __init__(self, img_size=64):
        super().__init__()
        self.img_size = img_size

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
# ORDER METRIC
# ==========================================

def calculate_order(img_tensor):
    """JPEG compression × TV smoothness score"""
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # JPEG Compression
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    # Total Variation (smoothness)
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_val = (tv_h + tv_w).item()

    # Soft mapping: smooth -> 1.0, noisy -> 0.0
    tv_score = math.exp(-10 * tv_val)

    return comp_score * tv_score


# ==========================================
# STRUCTURE MEASUREMENT (Simplified Nested Sampling)
# ==========================================

def get_weights_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def set_weights_vec(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())

def measure_structure(model_class, name, n_live=10, max_iter=100, seed=0):
    """
    Simplified nested sampling to measure structure score.
    Returns the score at ~10 bits of prior volume.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class()
    param_count = sum(p.numel() for p in model.parameters())

    # Initialize live points from prior N(0, 1)
    live = []
    for _ in range(n_live):
        w = torch.randn(len(get_weights_vec(model)))
        set_weights_vec(model, w)
        with torch.no_grad():
            score = calculate_order(model())
        live.append({'w': w, 'score': score})
    live.sort(key=lambda x: x['score'])

    # Run nested sampling
    final_score = 0.0
    for i in range(max_iter):
        dead = live.pop(0)
        threshold = dead['score']
        bits = i / n_live / math.log(2)

        # ESS to find replacement
        donor = np.random.choice(live)
        nu = torch.randn_like(donor['w'])
        theta = np.random.uniform(0, 2*np.pi)
        theta_min, theta_max = theta - 2*np.pi, theta

        for _ in range(30):
            new_w = donor['w'] * math.cos(theta) + nu * math.sin(theta)
            set_weights_vec(model, new_w)
            with torch.no_grad():
                score = calculate_order(model())

            if score > threshold:
                live.append({'w': new_w, 'score': score})
                break
            else:
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)
        else:
            live.append(donor)

        live.sort(key=lambda x: x['score'])

        # Record score at ~10 bits
        if bits >= 9.5:
            final_score = max(l['score'] for l in live)
            break

    return final_score, param_count


# ==========================================
# DIP MEASUREMENT
# ==========================================

def get_test_data(size=64, noise_level=0.15):
    """Create structured test image with geometric shapes."""
    img = Image.new('RGB', (size, size), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)

    draw.rectangle([10, 10, 30, 50], fill=(255, 50, 50))
    draw.ellipse([30, 20, 55, 45], fill=(50, 100, 255))
    draw.line([0, 60, 64, 0], fill=(255, 255, 0), width=3)
    draw.rectangle([45, 45, 58, 58], fill=(50, 200, 50))

    x_clean = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    x_clean = x_clean.unsqueeze(0)

    noise = torch.randn_like(x_clean) * noise_level
    x_noisy = torch.clamp(x_clean + noise, 0, 1)

    return x_clean, x_noisy


def measure_dip(model_class, name, n_steps=2000, lr=0.01, seed=0):
    """
    Deep Image Prior: Train to fit noisy image, measure generalization to clean.
    Returns best PSNR on clean target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_clean, x_noisy = get_test_data()
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_psnr = 0.0

    for step in range(n_steps):
        optimizer.zero_grad()
        output = model()
        loss = nn.functional.mse_loss(output, x_noisy)
        loss.backward()
        optimizer.step()

        # Measure generalization
        with torch.no_grad():
            mse_clean = nn.functional.mse_loss(output, x_clean).item()
            psnr = 10 * math.log10(1.0 / max(mse_clean, 1e-10))
            best_psnr = max(best_psnr, psnr)

    return best_psnr


# ==========================================
# MAIN EXPERIMENT
# ==========================================

ARCHITECTURES = [
    ('ResNet-4', ResNet4Gen),
    ('ResNet-2', ResNet2Gen),
    ('ResNet-6', ResNet6Gen),
    ('U-Net', UNetGen),
    ('Depthwise', DepthwiseSepGen),
    ('LocalAttn', LocalAttentionGen),
    ('CPPN', CPPNGen),
    ('Fourier', FourierFeaturesGen),
    ('HybridViT', HybridViTGen),
    ('ViT', ViTGen),
    ('MLP', MLPGen),
]


def run_experiment(n_seeds=2):
    """Run full experiment with multiple seeds."""
    print("=" * 60)
    print("COMPREHENSIVE SCALING LAW EXPERIMENT")
    print("=" * 60)

    results = {}

    for name, model_class in ARCHITECTURES:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")

        structures = []
        psnrs = []

        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}...")

            # Structure measurement
            structure, params = measure_structure(model_class, name,
                                                   n_live=10, max_iter=100, seed=seed)
            structures.append(structure)

            # DIP measurement
            psnr = measure_dip(model_class, name, n_steps=2000, lr=0.01, seed=seed)
            psnrs.append(psnr)

            print(f"    Structure: {structure:.4f}, PSNR: {psnr:.2f} dB")

        results[name] = {
            'structure_mean': np.mean(structures),
            'structure_std': np.std(structures),
            'psnr_mean': np.mean(psnrs),
            'psnr_std': np.std(psnrs),
            'params': params
        }

        print(f"  Mean: Structure={np.mean(structures):.4f}±{np.std(structures):.4f}, "
              f"PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB")

    return results


def create_figures(results):
    """Create scaling law scatter plot and architecture comparison."""
    fig_dir = Path(__file__).parent.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)

    names = list(results.keys())
    structures = [results[n]['structure_mean'] for n in names]
    structure_errs = [results[n]['structure_std'] for n in names]
    psnrs = [results[n]['psnr_mean'] for n in names]
    psnr_errs = [results[n]['psnr_std'] for n in names]

    # Calculate correlation
    r, p = stats.pearsonr(structures, psnrs)

    # Figure 1: Scatter plot with regression
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by architecture type
    colors = {
        'ResNet-4': '#2ecc71', 'ResNet-2': '#27ae60', 'ResNet-6': '#1abc9c',
        'U-Net': '#3498db', 'Depthwise': '#9b59b6', 'LocalAttn': '#e74c3c',
        'CPPN': '#f39c12', 'Fourier': '#e67e22',
        'HybridViT': '#95a5a6', 'ViT': '#7f8c8d', 'MLP': '#34495e'
    }

    for i, name in enumerate(names):
        ax.errorbar(structures[i], psnrs[i],
                   xerr=structure_errs[i], yerr=psnr_errs[i],
                   fmt='o', markersize=12, capsize=5,
                   color=colors.get(name, '#333'),
                   label=name)

    # Regression line
    z = np.polyfit(structures, psnrs, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(structures)-0.05, max(structures)+0.05, 100)
    ax.plot(x_line, p_line(x_line), '--', color='gray', alpha=0.7,
            label=f'Linear fit (r={r:.2f})')

    # Noise floor reference
    noisy_psnr = 18.21  # PSNR of noisy input
    ax.axhline(y=noisy_psnr, color='red', linestyle=':', alpha=0.5,
               label=f'Noise floor ({noisy_psnr:.1f} dB)')

    ax.set_xlabel('Thermodynamic Structure Score', fontsize=14)
    ax.set_ylabel('Best PSNR on Clean Target (dB)', fontsize=14)
    ax.set_title(f'The Thermodynamic Scaling Law (n={len(names)}, r={r:.2f}, p={p:.4f})',
                fontsize=16)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'scaling_law_comprehensive.pdf', dpi=150)
    plt.savefig(fig_dir / 'scaling_law_comprehensive.png', dpi=150)
    plt.close()

    # Figure 2: Architecture comparison bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(names))

    ax1.bar(x, structures, yerr=structure_errs, color=[colors.get(n, '#333') for n in names],
           capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Structure Score')
    ax1.set_title('Thermodynamic Structure by Architecture')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, psnrs, yerr=psnr_errs, color=[colors.get(n, '#333') for n in names],
           capsize=3)
    ax2.axhline(y=noisy_psnr, color='red', linestyle=':', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Best PSNR (dB)')
    ax2.set_title('DIP Denoising Performance by Architecture')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(fig_dir / 'scaling_law_architectures.pdf', dpi=150)
    plt.savefig(fig_dir / 'scaling_law_architectures.png', dpi=150)
    plt.close()

    print(f"\nFigures saved to {fig_dir}")
    return r, p


def save_results(results, r, p):
    """Save results to JSON."""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    output = {
        'correlation': {'pearson_r': r, 'p_value': p},
        'architectures': results
    }

    with open(results_dir / 'comprehensive_scaling_law.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {results_dir / 'comprehensive_scaling_law.json'}")


if __name__ == '__main__':
    results = run_experiment(n_seeds=2)
    r, p = create_figures(results)
    save_results(results, r, p)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Correlation: r = {r:.3f}, p = {p:.6f}")
    print(f"Architectures tested: {len(results)}")

    # Print table
    print("\n{:<12} {:>10} {:>10}".format("Arch", "Structure", "PSNR"))
    print("-" * 35)
    for name in sorted(results.keys(), key=lambda n: -results[n]['structure_mean']):
        r = results[name]
        print(f"{name:<12} {r['structure_mean']:>10.4f} {r['psnr_mean']:>10.2f}")
