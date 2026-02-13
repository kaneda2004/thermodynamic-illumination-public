"""
Order Metric Robustness Experiment v2
======================================

Instead of measuring bits (which saturates), we directly compare:
1. Average structure score for each architecture
2. Rankings across different metrics

This addresses reviewer concern #5 about "bits to YOUR metric".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.fftpack import dct
from scipy import ndimage
import matplotlib.pyplot as plt
import json
from pathlib import Path
import zlib


torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# ARCHITECTURES
# ============================================================================

class ResNetGen(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, out_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.blocks(self.fc(z).view(-1, 128, 4, 4))


class MLPGen(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3, img_size=64):
        super().__init__()
        self.img_size, self.out_channels = img_size, out_channels
        out_dim = out_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.out_channels, self.img_size, self.img_size)


class ViTGen(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, img_size=64, patch_size=8):
        super().__init__()
        self.img_size, self.patch_size = img_size, patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.out_channels = out_channels
        embed_dim = 128

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                    dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, z):
        B = z.shape[0]
        x = self.transformer(self.pos_embed.expand(B, -1, -1))
        x = torch.sigmoid(self.to_pixels(x))
        p, n = self.patch_size, self.img_size // self.patch_size
        x = x.view(B, n, n, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, self.out_channels, self.img_size, self.img_size)


class CPPNGen(nn.Module):
    def __init__(self, latent_dim=8, out_channels=3, img_size=64):
        super().__init__()
        self.img_size, self.out_channels = img_size, out_channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, out_channels), nn.Sigmoid(),
        )
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size), indexing='ij'
        ), dim=-1)
        r = torch.sqrt(coords[..., 0]**2 + coords[..., 1]**2).unsqueeze(-1)
        self.register_buffer('coords', torch.cat([coords, r], dim=-1))

    def forward(self, z):
        B, H, W = z.shape[0], self.img_size, self.img_size
        z_exp = z.view(B, 1, 1, -1).expand(B, H, W, -1)
        inp = torch.cat([z_exp, self.coords.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1)
        return self.net(inp).permute(0, 3, 1, 2)


class UniformGen(nn.Module):
    """Uniform random baseline - no structure at all."""
    def __init__(self, out_channels=3, img_size=64):
        super().__init__()
        self.out_channels, self.img_size = out_channels, img_size

    def forward(self, z):
        return torch.rand(z.shape[0], self.out_channels, self.img_size, self.img_size,
                         device=z.device)


# ============================================================================
# ORDER METRICS
# ============================================================================

def metric_original(img: torch.Tensor) -> float:
    """Original: Compression × TV_soft"""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    comp_ratio = 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return comp_ratio * np.exp(-10 * tv)


def metric_additive(img: torch.Tensor) -> float:
    """Additive: 0.5 * Compression + 0.5 * TV_soft"""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    comp_ratio = 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return 0.5 * comp_ratio + 0.5 * np.exp(-10 * tv)


def metric_tv_only(img: torch.Tensor) -> float:
    """TV-only"""
    if img.dim() == 4:
        img = img[0]
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return np.exp(-10 * tv)


def metric_compression_only(img: torch.Tensor) -> float:
    """Compression-only"""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    return 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes


def metric_low_freq_energy(img: torch.Tensor) -> float:
    """Low-frequency energy from DCT"""
    if img.dim() == 4:
        img = img[0]
    gray = img.detach().cpu().numpy().mean(axis=0)
    dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    lf_energy = np.sum(np.abs(dct_coeffs[:8, :8])**2)
    return lf_energy / (np.sum(np.abs(dct_coeffs)**2) + 1e-10)


def metric_gradient_smoothness(img: torch.Tensor) -> float:
    """Inverse gradient magnitude"""
    if img.dim() == 4:
        img = img[0]
    gray = img.detach().cpu().numpy().mean(axis=0)
    sx = ndimage.sobel(gray, axis=0)
    sy = ndimage.sobel(gray, axis=1)
    return np.exp(-5 * np.sqrt(sx**2 + sy**2).mean())


# ============================================================================
# EXPERIMENT
# ============================================================================

def measure_scores(model, metric_fn, latent_dim, n_samples=100, device='cpu'):
    """Measure average score for an architecture."""
    model.eval()
    scores = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            img = model(z)
            scores.append(metric_fn(img))
    return np.mean(scores), np.std(scores)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Architectures with expected ordering: ResNet > CPPN > MLP > ViT > Uniform
    architectures = {
        'ResNet': (ResNetGen, 128),
        'CPPN': (CPPNGen, 8),
        'MLP': (MLPGen, 64),
        'ViT': (ViTGen, 128),
        'Uniform': (UniformGen, 1),  # latent_dim doesn't matter
    }

    metrics = {
        'Original': metric_original,
        'Additive': metric_additive,
        'TV-only': metric_tv_only,
        'Compress': metric_compression_only,
        'LowFreq': metric_low_freq_energy,
        'Gradient': metric_gradient_smoothness,
    }

    results = {arch: {} for arch in architectures}

    print("\n" + "="*70)
    print("ORDER METRIC ROBUSTNESS v2: Direct Score Comparison")
    print("="*70)

    for metric_name, metric_fn in metrics.items():
        print(f"\n--- Metric: {metric_name} ---")
        for arch_name, (cls, latent_dim) in architectures.items():
            if arch_name == 'Uniform':
                model = cls().to(device)
            else:
                model = cls(latent_dim=latent_dim).to(device)
            mean, std = measure_scores(model, metric_fn, latent_dim, n_samples=50, device=device)
            results[arch_name][metric_name] = {'mean': mean, 'std': std}
            print(f"  {arch_name}: {mean:.4f} ± {std:.4f}")

    # Compute rankings per metric
    print("\n" + "="*70)
    print("ARCHITECTURE RANKINGS BY METRIC")
    print("="*70)

    arch_names = list(architectures.keys())
    rankings = {}

    for metric_name in metrics:
        scores = [results[arch][metric_name]['mean'] for arch in arch_names]
        # Higher score = better = lower rank (1 is best)
        ranks = len(scores) - stats.rankdata(scores) + 1
        rankings[metric_name] = ranks

        sorted_archs = sorted(zip(arch_names, scores), key=lambda x: x[1], reverse=True)
        print(f"\n{metric_name}:")
        for i, (arch, score) in enumerate(sorted_archs, 1):
            print(f"  {i}. {arch}: {score:.4f}")

    # Compute pairwise Kendall's tau
    print("\n" + "="*70)
    print("RANK CORRELATION (Kendall's Tau)")
    print("="*70)

    metric_names = list(metrics.keys())
    tau_matrix = np.zeros((len(metrics), len(metrics)))

    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            tau, _ = stats.kendalltau(rankings[m1], rankings[m2])
            tau_matrix[i, j] = tau

    # Print matrix
    print("\n" + " "*12 + "".join(f"{m[:8]:>10}" for m in metric_names))
    for i, m1 in enumerate(metric_names):
        print(f"{m1[:10]:<12}" + "".join(f"{tau_matrix[i,j]:>10.2f}" for j in range(len(metric_names))))

    # Average tau (off-diagonal)
    mask = ~np.eye(len(metrics), dtype=bool)
    avg_tau = tau_matrix[mask].mean()
    min_tau = tau_matrix[mask].min()

    print(f"\nAverage Kendall's τ: {avg_tau:.3f}")
    print(f"Minimum Kendall's τ: {min_tau:.3f}")

    # Save results
    results_dir = Path('results/metric_robustness_v2')
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'scores': {arch: {m: {'mean': float(v['mean']), 'std': float(v['std'])}
                          for m, v in metrics_dict.items()}
                   for arch, metrics_dict in results.items()},
        'rankings': {m: list(map(int, r)) for m, r in rankings.items()},
        'tau_matrix': tau_matrix.tolist(),
        'average_tau': avg_tau,
        'minimum_tau': min_tau,
        'arch_names': arch_names,
        'metric_names': metric_names,
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score heatmap
    ax = axes[0]
    score_matrix = np.array([[results[arch][m]['mean'] for m in metric_names]
                             for arch in arch_names])
    im = ax.imshow(score_matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_yticks(range(len(arch_names)))
    ax.set_yticklabels(arch_names)
    ax.set_title('Structure Score by Metric and Architecture')
    plt.colorbar(im, ax=ax, label='Score')

    for i in range(len(arch_names)):
        for j in range(len(metric_names)):
            color = 'white' if score_matrix[i, j] < score_matrix.mean() else 'black'
            ax.text(j, i, f'{score_matrix[i,j]:.2f}', ha='center', va='center',
                   color=color, fontsize=9)

    # Ranking comparison
    ax = axes[1]
    x = np.arange(len(arch_names))
    width = 0.12
    for i, (metric_name, ranks) in enumerate(rankings.items()):
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, ranks, width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, rotation=45, ha='right')
    ax.set_ylabel('Rank (1=best structure)')
    ax.set_title(f'Architecture Rankings Across Metrics\n(avg τ = {avg_tau:.2f})')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig('figures/metric_robustness_v2.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/metric_robustness_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to figures/metric_robustness_v2.pdf")
    print(f"Results saved to {results_dir}/results.json")

    print("\n" + "="*70)
    if avg_tau >= 0.8:
        print(f"SUCCESS: Rankings are HIGHLY STABLE (avg τ = {avg_tau:.2f} ≥ 0.8)")
    elif avg_tau >= 0.6:
        print(f"GOOD: Rankings are STABLE (avg τ = {avg_tau:.2f} ≥ 0.6)")
    else:
        print(f"CAUTION: Rankings show variation (avg τ = {avg_tau:.2f} < 0.6)")
    print("="*70)

    return results


if __name__ == '__main__':
    main()
