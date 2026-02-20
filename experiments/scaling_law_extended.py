"""
Extended Scaling Law Experiment: 12+ Architecture Variants
============================================================

Goal: Turn the 3-point anecdote (ResNet, ViT, MLP) into real evidence
with 10+ architecture points spanning the inductive bias spectrum.

Architectures tested:
1. ResNet (existing) - strong local bias
2. ViT (existing) - weak global bias
3. MLP (existing) - no spatial bias
4. Conv 1×1 - pixel-wise only
5. Conv 5×5 - larger receptive field
6. Conv 7×7 - even larger
7. Depthwise Separable - efficient local
8. Local Attention (window=4) - restricted attention
9. Hybrid Conv+MLP - mixed
10. CPPN-style - coordinate-based
11. U-Net decoder - skip connections
12. Shallow ResNet - fewer layers

For each architecture:
- Measure Structure Score via simplified nested sampling
- Run DIP reconstruction to measure denoising capability
- Report correlation coefficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json
from pathlib import Path
import zlib


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class OrderMetric:
    """Compute structure score using compression + smoothness."""

    @staticmethod
    def compute(img: torch.Tensor) -> float:
        """
        Compute order metric for an image tensor.
        img: (C, H, W) tensor with values in [0, 1]
        Returns: float in [0, 1], higher = more structured
        """
        # Convert to numpy, scale to 0-255
        if img.dim() == 4:
            img = img[0]
        img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)

        # Compression ratio
        raw_size = img_np.nbytes
        compressed = zlib.compress(img_np.tobytes(), level=9)
        comp_ratio = 1 - len(compressed) / raw_size

        # Total Variation (smoothness)
        img_float = img.detach().cpu().float()
        tv_h = torch.abs(img_float[:, 1:, :] - img_float[:, :-1, :]).mean()
        tv_w = torch.abs(img_float[:, :, 1:] - img_float[:, :, :-1]).mean()
        tv_val = (tv_h + tv_w).item()

        # Soft exponential mapping for TV
        tv_score = np.exp(-10 * tv_val)

        # Combined score
        return comp_ratio * tv_score


# ============================================================================
# ARCHITECTURES
# ============================================================================

class ResNetGen(nn.Module):
    """Standard 4-layer ResNet decoder."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size

        # Start from 4×4
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.blocks = nn.ModuleList([
            self._make_block(128, 64),   # 4→8
            self._make_block(64, 32),    # 8→16
            self._make_block(32, 16),    # 16→32
            self._make_block(16, 8),     # 32→64
        ])

        self.final = nn.Conv2d(8, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.final(x))


class MLPGen(nn.Module):
    """Pure MLP decoder - no spatial inductive bias."""
    def __init__(self, latent_dim=64, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels

        out_dim = out_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.out_channels, self.img_size, self.img_size)


class ViTGen(nn.Module):
    """Vision Transformer decoder - global attention."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64, patch_size=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.out_channels = out_channels

        embed_dim = 128
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.pos_embed.expand(batch_size, -1, -1)
        x = self.transformer(x)
        x = self.to_pixels(x)

        # Reshape to image
        p = self.patch_size
        n_side = self.img_size // p
        x = x.view(batch_size, n_side, n_side, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, self.out_channels, self.img_size, self.img_size)
        return self.sigmoid(x)


class Conv1x1Gen(nn.Module):
    """Pixel-wise convolutions only - no spatial mixing."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 64 * img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, self.img_size, self.img_size)
        return self.net(x)


class Conv5x5Gen(nn.Module):
    """5×5 convolutions - larger receptive field than ResNet."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.blocks = nn.ModuleList([
            self._make_block(128, 64),
            self._make_block(64, 32),
            self._make_block(32, 16),
            self._make_block(16, 8),
        ])

        self.final = nn.Conv2d(8, out_channels, 5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.final(x))


class Conv7x7Gen(nn.Module):
    """7×7 convolutions - even larger receptive field."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.blocks = nn.ModuleList([
            self._make_block(128, 64),
            self._make_block(64, 32),
            self._make_block(32, 16),
            self._make_block(16, 8),
        ])

        self.final = nn.Conv2d(8, out_channels, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.final(x))


class DepthwiseSepGen(nn.Module):
    """Depthwise separable convolutions - efficient local bias."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.blocks = nn.ModuleList([
            self._make_block(128, 64),
            self._make_block(64, 32),
            self._make_block(32, 16),
            self._make_block(16, 8),
        ])

        self.final = nn.Conv2d(8, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.final(x))


class LocalAttentionGen(nn.Module):
    """Local attention with window=4 - restricted attention span."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64, window_size=4):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.out_channels = out_channels

        embed_dim = 64
        self.fc = nn.Linear(latent_dim, embed_dim * img_size * img_size)

        # Simple local attention: process in windows
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.to_pixels = nn.Conv2d(embed_dim, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        B = z.shape[0]
        H = W = self.img_size
        ws = self.window_size

        x = self.fc(z).view(B, H, W, -1)  # B, H, W, C
        C = x.shape[-1]

        # Reshape into windows
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)  # (B*nw, ws*ws, C)

        # Self-attention within windows
        qkv = self.qkv(x).view(-1, ws * ws, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = self.proj(x)

        # Reshape back
        nH, nW = H // ws, W // ws
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W

        return self.sigmoid(self.to_pixels(x))


class HybridConvMLPGen(nn.Module):
    """Hybrid: Conv stem + MLP middle + Conv decoder."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size

        # Conv stem
        self.stem = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
        )

        # MLP middle
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 8 * 8),
            nn.Unflatten(1, (32, 8, 8)),
        )

        # Conv decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.stem(z)
        x = self.mlp(x)
        return self.decoder(x)


class CPPNGen(nn.Module):
    """CPPN-style: coordinate-based generation."""
    def __init__(self, latent_dim=8, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels

        # CPPN network
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 32),  # +3 for x, y, r
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Sin() if hasattr(nn, 'Sin') else nn.Tanh(),
            nn.Linear(32, out_channels),
            nn.Sigmoid(),
        )

        # Precompute coordinate grid
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        ), dim=-1)
        r = torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2).unsqueeze(-1)
        self.register_buffer('coords', torch.cat([coords, r], dim=-1))

    def forward(self, z):
        B = z.shape[0]
        H, W = self.img_size, self.img_size

        # Expand z to all pixels
        z_exp = z.view(B, 1, 1, -1).expand(B, H, W, -1)
        coords_exp = self.coords.unsqueeze(0).expand(B, -1, -1, -1)

        # Concatenate and process
        inp = torch.cat([z_exp, coords_exp], dim=-1)
        out = self.net(inp)

        return out.permute(0, 3, 1, 2)


class UNetDecoderGen(nn.Module):
    """U-Net style decoder with skip connections."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size

        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        # Encoder path (downsampling to get skip features)
        self.enc1 = nn.Conv2d(128, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, 3, padding=1)

        # Decoder path
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # +skip
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32 + 32, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(8, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)

        # Get skip features
        s1 = F.relu(self.enc1(x))  # 64ch, 4x4
        s2 = F.relu(self.enc2(s1))  # 32ch, 4x4

        # Decode with skips
        d1 = self.dec1(x)  # 64ch, 8x8
        d1 = torch.cat([d1, F.interpolate(s1, d1.shape[2:])], dim=1)
        d2 = self.dec2(d1)  # 32ch, 16x16
        d2 = torch.cat([d2, F.interpolate(s2, d2.shape[2:])], dim=1)
        d3 = self.dec3(d2)  # 16ch, 32x32
        d4 = self.dec4(d3)  # 8ch, 64x64

        return self.sigmoid(self.final(d4))


class ShallowResNetGen(nn.Module):
    """Shallow ResNet - only 2 layers."""
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size

        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)

        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 64
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 8, 8)
        return self.blocks(x)


# ============================================================================
# EXPERIMENTS
# ============================================================================

def measure_structure(model, n_samples=50, latent_dim=128, device='cpu'):
    """Measure average structure score for random outputs."""
    model.eval()
    scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            img = model(z)
            score = OrderMetric.compute(img)
            scores.append(score)

    return np.mean(scores), np.std(scores)


def create_target_image(size=64):
    """Create structured test image for DIP."""
    img = torch.zeros(3, size, size)

    # Rectangle
    img[0, 10:30, 10:30] = 1.0

    # Circle
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    circle = ((x - 45)**2 + (y - 45)**2) < 100
    img[1, circle] = 1.0

    # Diagonal line
    for i in range(size):
        if 0 <= i < size:
            img[2, i, i] = 1.0
            if i > 0:
                img[2, i, i-1] = 0.5
            if i < size - 1:
                img[2, i, i+1] = 0.5

    return img


def run_dip(model, target, latent_dim, noise_level=0.15, n_iters=500, device='cpu'):
    """Run Deep Image Prior reconstruction."""
    model.train()
    noisy = target + noise_level * torch.randn_like(target)
    noisy = noisy.clamp(0, 1).to(device)
    target = target.to(device)

    # Fixed random input
    z = torch.randn(1, latent_dim, device=device, requires_grad=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_psnr = 0
    best_iter = 0
    psnrs = []

    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(z)
        loss = F.mse_loss(output, noisy.unsqueeze(0))
        loss.backward()
        optimizer.step()

        # PSNR to clean target
        with torch.no_grad():
            mse = F.mse_loss(output, target.unsqueeze(0))
            psnr = 10 * torch.log10(1 / (mse + 1e-10)).item()
            psnrs.append(psnr)

            if psnr > best_psnr:
                best_psnr = psnr
                best_iter = i

    # Baseline: noisy input PSNR
    noisy_mse = F.mse_loss(noisy, target)
    noisy_psnr = 10 * torch.log10(1 / (noisy_mse + 1e-10)).item()

    return best_psnr, best_iter, noisy_psnr, psnrs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Architecture configurations
    architectures = {
        'ResNet': (ResNetGen, {'latent_dim': 128}),
        'MLP': (MLPGen, {'latent_dim': 64}),
        'ViT': (ViTGen, {'latent_dim': 128}),
        'Conv1x1': (Conv1x1Gen, {'latent_dim': 128}),
        'Conv5x5': (Conv5x5Gen, {'latent_dim': 128}),
        'Conv7x7': (Conv7x7Gen, {'latent_dim': 128}),
        'DepthwiseSep': (DepthwiseSepGen, {'latent_dim': 128}),
        'LocalAttn': (LocalAttentionGen, {'latent_dim': 128}),
        'HybridConvMLP': (HybridConvMLPGen, {'latent_dim': 128}),
        'CPPN': (CPPNGen, {'latent_dim': 8}),
        'UNetDecoder': (UNetDecoderGen, {'latent_dim': 128}),
        'ShallowResNet': (ShallowResNetGen, {'latent_dim': 128}),
    }

    results = {}
    target = create_target_image(64)

    print("\n" + "="*70)
    print("EXTENDED SCALING LAW EXPERIMENT")
    print("="*70)

    for name, (cls, kwargs) in architectures.items():
        print(f"\n--- {name} ---")

        # Measure structure
        model = cls(**kwargs).to(device)
        latent_dim = kwargs.get('latent_dim', 128)

        struct_mean, struct_std = measure_structure(model, n_samples=50,
                                                     latent_dim=latent_dim, device=device)
        print(f"  Structure Score: {struct_mean:.4f} ± {struct_std:.4f}")

        # Run DIP
        model = cls(**kwargs).to(device)  # Fresh model
        best_psnr, best_iter, noisy_psnr, _ = run_dip(model, target, latent_dim, device=device)
        denoising_gap = best_psnr - noisy_psnr

        print(f"  Best PSNR: {best_psnr:.2f}dB (iter {best_iter})")
        print(f"  Noisy PSNR: {noisy_psnr:.2f}dB")
        print(f"  Denoising Gap: {denoising_gap:+.2f}dB")

        results[name] = {
            'structure_mean': struct_mean,
            'structure_std': struct_std,
            'best_psnr': best_psnr,
            'best_iter': best_iter,
            'noisy_psnr': noisy_psnr,
            'denoising_gap': denoising_gap,
        }

    # Compute correlations
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    structures = [r['structure_mean'] for r in results.values()]
    psnrs = [r['best_psnr'] for r in results.values()]
    gaps = [r['denoising_gap'] for r in results.values()]
    names = list(results.keys())

    r_psnr, p_psnr = stats.pearsonr(structures, psnrs)
    r_gap, p_gap = stats.pearsonr(structures, gaps)

    print(f"\nStructure vs PSNR: r = {r_psnr:.3f}, p = {p_psnr:.4f}")
    print(f"Structure vs Denoising Gap: r = {r_gap:.3f}, p = {p_gap:.4f}")

    # Save results
    results_dir = Path('results/scaling_law_extended')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'results.json', 'w') as f:
        json.dump({
            'architectures': results,
            'correlation_psnr': {'r': r_psnr, 'p': p_psnr},
            'correlation_gap': {'r': r_gap, 'p': p_gap},
        }, f, indent=2)

    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Structure vs PSNR
    ax = axes[0]
    ax.scatter(structures, psnrs, s=100, c='steelblue', edgecolors='black')
    for i, name in enumerate(names):
        ax.annotate(name, (structures[i], psnrs[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Structure Score', fontsize=12)
    ax.set_ylabel('Best PSNR (dB)', fontsize=12)
    ax.set_title(f'Structure vs Reconstruction (r={r_psnr:.3f}, p={p_psnr:.4f})', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Structure vs Denoising Gap
    ax = axes[1]
    colors = ['green' if g > 0 else 'red' for g in gaps]
    ax.scatter(structures, gaps, s=100, c=colors, edgecolors='black')
    for i, name in enumerate(names):
        ax.annotate(name, (structures[i], gaps[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Structure Score', fontsize=12)
    ax.set_ylabel('Denoising Gap (dB)', fontsize=12)
    ax.set_title(f'Structure vs Denoising (r={r_gap:.3f}, p={p_gap:.4f})', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/scaling_law_extended.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/scaling_law_extended.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to figures/scaling_law_extended.pdf")
    print(f"Results saved to {results_dir}/results.json")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Architecture':<15} {'Structure':>10} {'PSNR':>8} {'Gap':>8}")
    print("-"*45)
    for name, r in sorted(results.items(), key=lambda x: x[1]['structure_mean'], reverse=True):
        print(f"{name:<15} {r['structure_mean']:>10.4f} {r['best_psnr']:>8.2f} {r['denoising_gap']:>+8.2f}")

    print(f"\n{'='*70}")
    print(f"FINAL: r = {r_psnr:.3f} (n = {len(results)} architectures)")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
