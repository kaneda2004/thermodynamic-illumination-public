#!/usr/bin/env python3
"""
DIP Validation Experiment: 5 Targets x 5 Seeds

Goal: Validate that ResNet > ViT > MLP ranking for denoising is consistent
across different target images and noise seeds, not a fluke of one specific image.

Tests:
- 5 different target images: geometric, checkerboard, gradient, ring, random_smooth
- 5 different noise seeds per target
- 3 architectures: ResNet (~247K params), ViT (~563K params), MLP (~13M params)
- Total: 75 training runs

Metrics:
- Per-architecture: mean PSNR +/- std over 25 runs
- Ranking consistency: how often does ResNet > ViT? ResNet > MLP?
- Statistical test: Wilcoxon signed-rank test for paired comparisons

Usage:
    cd /Users/matt/Development/monochrome_noise_converger
    uv run python experiments/dip_validation_5x5.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import json
from PIL import Image, ImageDraw
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. TARGET IMAGE GENERATORS
# ==========================================

def create_geometric_shapes(size=64, seed=42):
    """Geometric shapes (rectangles, circles, lines)."""
    np.random.seed(seed)
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

    return np.array(img).astype(np.float32) / 255.0


def create_checkerboard(size=64, squares=8, seed=42):
    """Checkerboard pattern."""
    np.random.seed(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    sq_size = size // squares
    colors = [
        (0.9, 0.1, 0.1),  # Red
        (0.1, 0.1, 0.9),  # Blue
    ]

    for i in range(squares):
        for j in range(squares):
            color_idx = (i + j) % 2
            y_start, y_end = i * sq_size, (i + 1) * sq_size
            x_start, x_end = j * sq_size, (j + 1) * sq_size
            img[y_start:y_end, x_start:x_end] = colors[color_idx]

    return img


def create_gradient(size=64, seed=42):
    """Smooth gradient transitions."""
    np.random.seed(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Horizontal gradient for R channel
    for x in range(size):
        img[:, x, 0] = x / size

    # Vertical gradient for G channel
    for y in range(size):
        img[y, :, 1] = y / size

    # Diagonal gradient for B channel
    for y in range(size):
        for x in range(size):
            img[y, x, 2] = 0.5 + 0.5 * math.sin((x + y) * math.pi / size)

    return img


def create_ring_pattern(size=64, seed=42):
    """Concentric rings/circles pattern."""
    np.random.seed(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    center_x, center_y = size // 2, size // 2

    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Create alternating colored rings
            ring_idx = int(dist / 8) % 3
            if ring_idx == 0:
                img[y, x] = (0.9, 0.2, 0.2)  # Red
            elif ring_idx == 1:
                img[y, x] = (0.2, 0.9, 0.2)  # Green
            else:
                img[y, x] = (0.2, 0.2, 0.9)  # Blue

    return img


def create_random_smooth(size=64, seed=42):
    """Random smooth texture using low-frequency noise."""
    np.random.seed(seed)

    # Create low-frequency random values
    low_res = 8
    low_img = np.random.rand(low_res, low_res, 3).astype(np.float32)

    # Upsample with bilinear interpolation
    from PIL import Image as PILImage
    img = np.zeros((size, size, 3), dtype=np.float32)

    for c in range(3):
        pil_low = PILImage.fromarray((low_img[:, :, c] * 255).astype(np.uint8))
        pil_high = pil_low.resize((size, size), PILImage.BILINEAR)
        img[:, :, c] = np.array(pil_high).astype(np.float32) / 255.0

    return img


# ==========================================
# 2. ARCHITECTURES (from dip_dynamics.py)
# ==========================================

class ResNetGen(nn.Module):
    """Convolutional Generator (Low-Bit Prior) - ~247K params"""
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
    """Vision Transformer Generator (High-Bit Prior) - ~563K params"""
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
    """MLP Generator (Maximum Entropy Prior) - ~13M params"""
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
# 3. TRAINING AND METRICS
# ==========================================

def psnr(pred, target):
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def train_single_run(model, target_clean, target_noisy, steps=2000, lr=0.01):
    """
    Train the network to reconstruct the NOISY image.
    Track best PSNR against CLEAN image.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_psnr = -float('inf')
    best_iter = 0

    for i in range(steps):
        optimizer.zero_grad()
        out = model()

        # Loss is ONLY against noisy target
        loss = nn.MSELoss()(out, target_noisy)
        loss.backward()
        optimizer.step()

        # Track best PSNR vs clean
        with torch.no_grad():
            psnr_clean = psnr(out, target_clean)
            if psnr_clean > best_psnr:
                best_psnr = psnr_clean
                best_iter = i

    return best_psnr, best_iter


# ==========================================
# 4. MAIN EXPERIMENT
# ==========================================

def run_validation_experiment():
    """Run the full 5x5 validation experiment."""

    print("=" * 70)
    print("DIP Validation Experiment: 5 Targets x 5 Seeds x 3 Architectures")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    size = 64
    noise_level = 0.15
    steps = 2000
    lr = 0.01

    # Target generators
    target_generators = {
        'geometric': create_geometric_shapes,
        'checkerboard': create_checkerboard,
        'gradient': create_gradient,
        'ring': create_ring_pattern,
        'random_smooth': create_random_smooth,
    }

    # Noise seeds
    noise_seeds = [42, 123, 456, 789, 1024]

    # Architecture constructors
    architectures = {
        'ResNet': lambda: ResNetGen(),
        'ViT': lambda: ViTGen(),
        'MLP': lambda: MLPGen(),
    }

    # Count parameters
    print("Architecture Parameter Counts:")
    for arch_name, arch_fn in architectures.items():
        model = arch_fn()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  {arch_name}: {param_count:,} parameters")
    print()

    # Results storage
    results = {
        'config': {
            'size': size,
            'noise_level': noise_level,
            'steps': steps,
            'lr': lr,
            'targets': list(target_generators.keys()),
            'noise_seeds': noise_seeds,
            'architectures': list(architectures.keys()),
        },
        'runs': [],
        'summary': {},
    }

    # Track per-architecture PSNR values
    psnr_by_arch = {arch: [] for arch in architectures.keys()}

    # Run experiments
    total_runs = len(target_generators) * len(noise_seeds) * len(architectures)
    run_count = 0

    print(f"Running {total_runs} training runs...")
    print("-" * 70)

    for target_name, target_fn in target_generators.items():
        # Create clean target
        clean_np = target_fn(size=size, seed=42)  # Fixed seed for target itself
        clean_tensor = torch.tensor(clean_np).permute(2, 0, 1).unsqueeze(0)

        for noise_seed in noise_seeds:
            # Add noise with this seed
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)
            noise = torch.randn_like(clean_tensor) * noise_level
            noisy_tensor = torch.clamp(clean_tensor + noise, 0, 1)

            for arch_name, arch_fn in architectures.items():
                run_count += 1

                # Set seeds for reproducibility
                torch.manual_seed(noise_seed + hash(arch_name) % 10000)
                np.random.seed(noise_seed + hash(arch_name) % 10000)

                # Create fresh model
                model = arch_fn()

                # Train
                best_psnr, best_iter = train_single_run(
                    model, clean_tensor, noisy_tensor,
                    steps=steps, lr=lr
                )

                # Store result
                run_result = {
                    'target': target_name,
                    'noise_seed': noise_seed,
                    'architecture': arch_name,
                    'best_psnr': best_psnr,
                    'best_iter': best_iter,
                }
                results['runs'].append(run_result)
                psnr_by_arch[arch_name].append(best_psnr)

                # Progress
                print(f"[{run_count:3d}/{total_runs}] {target_name:12s} | seed={noise_seed:4d} | "
                      f"{arch_name:6s}: {best_psnr:.2f}dB @ iter {best_iter}")

    print("-" * 70)
    print()

    # ==========================================
    # 5. STATISTICAL ANALYSIS
    # ==========================================

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Per-architecture statistics
    print("\nPer-Architecture Statistics (25 runs each):")
    print("-" * 50)

    arch_stats = {}
    for arch_name, psnr_values in psnr_by_arch.items():
        mean_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        min_psnr = np.min(psnr_values)
        max_psnr = np.max(psnr_values)

        arch_stats[arch_name] = {
            'mean': mean_psnr,
            'std': std_psnr,
            'min': min_psnr,
            'max': max_psnr,
            'values': psnr_values,
        }

        print(f"  {arch_name:6s}: {mean_psnr:.2f} +/- {std_psnr:.2f} dB "
              f"(range: {min_psnr:.2f} - {max_psnr:.2f})")

    results['summary']['per_architecture'] = {
        arch: {k: v for k, v in stats.items() if k != 'values'}
        for arch, stats in arch_stats.items()
    }

    # Ranking consistency
    print("\nRanking Consistency (pairwise comparisons):")
    print("-" * 50)

    resnet_psnrs = np.array(psnr_by_arch['ResNet'])
    vit_psnrs = np.array(psnr_by_arch['ViT'])
    mlp_psnrs = np.array(psnr_by_arch['MLP'])

    resnet_beats_vit = np.sum(resnet_psnrs > vit_psnrs)
    resnet_beats_mlp = np.sum(resnet_psnrs > mlp_psnrs)
    vit_beats_mlp = np.sum(vit_psnrs > mlp_psnrs)

    n_runs = len(resnet_psnrs)

    print(f"  ResNet > ViT: {resnet_beats_vit}/{n_runs} runs ({100*resnet_beats_vit/n_runs:.1f}%)")
    print(f"  ResNet > MLP: {resnet_beats_mlp}/{n_runs} runs ({100*resnet_beats_mlp/n_runs:.1f}%)")
    print(f"  ViT > MLP:    {vit_beats_mlp}/{n_runs} runs ({100*vit_beats_mlp/n_runs:.1f}%)")

    results['summary']['ranking_consistency'] = {
        'resnet_beats_vit': {'count': int(resnet_beats_vit), 'total': n_runs,
                             'percentage': 100*resnet_beats_vit/n_runs},
        'resnet_beats_mlp': {'count': int(resnet_beats_mlp), 'total': n_runs,
                             'percentage': 100*resnet_beats_mlp/n_runs},
        'vit_beats_mlp': {'count': int(vit_beats_mlp), 'total': n_runs,
                          'percentage': 100*vit_beats_mlp/n_runs},
    }

    # Statistical tests
    print("\nStatistical Tests (Wilcoxon signed-rank, paired):")
    print("-" * 50)

    # ResNet vs ViT
    stat_rv, p_rv = stats.wilcoxon(resnet_psnrs, vit_psnrs, alternative='greater')
    print(f"  ResNet vs ViT: W={stat_rv:.1f}, p={p_rv:.6f} {'***' if p_rv < 0.001 else '**' if p_rv < 0.01 else '*' if p_rv < 0.05 else ''}")

    # ResNet vs MLP
    stat_rm, p_rm = stats.wilcoxon(resnet_psnrs, mlp_psnrs, alternative='greater')
    print(f"  ResNet vs MLP: W={stat_rm:.1f}, p={p_rm:.6f} {'***' if p_rm < 0.001 else '**' if p_rm < 0.01 else '*' if p_rm < 0.05 else ''}")

    # ViT vs MLP
    stat_vm, p_vm = stats.wilcoxon(vit_psnrs, mlp_psnrs, alternative='greater')
    print(f"  ViT vs MLP:    W={stat_vm:.1f}, p={p_vm:.6f} {'***' if p_vm < 0.001 else '**' if p_vm < 0.01 else '*' if p_vm < 0.05 else ''}")

    results['summary']['statistical_tests'] = {
        'resnet_vs_vit': {'statistic': float(stat_rv), 'p_value': float(p_rv)},
        'resnet_vs_mlp': {'statistic': float(stat_rm), 'p_value': float(p_rm)},
        'vit_vs_mlp': {'statistic': float(stat_vm), 'p_value': float(p_vm)},
    }

    # Effect sizes (Cohen's d)
    print("\nEffect Sizes (Cohen's d):")
    print("-" * 50)

    def cohens_d(x, y):
        pooled_std = np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)
        return (np.mean(x) - np.mean(y)) / pooled_std

    d_rv = cohens_d(resnet_psnrs, vit_psnrs)
    d_rm = cohens_d(resnet_psnrs, mlp_psnrs)
    d_vm = cohens_d(vit_psnrs, mlp_psnrs)

    print(f"  ResNet vs ViT: d={d_rv:.2f} ({'large' if abs(d_rv) > 0.8 else 'medium' if abs(d_rv) > 0.5 else 'small'})")
    print(f"  ResNet vs MLP: d={d_rm:.2f} ({'large' if abs(d_rm) > 0.8 else 'medium' if abs(d_rm) > 0.5 else 'small'})")
    print(f"  ViT vs MLP:    d={d_vm:.2f} ({'large' if abs(d_vm) > 0.8 else 'medium' if abs(d_vm) > 0.5 else 'small'})")

    results['summary']['effect_sizes'] = {
        'resnet_vs_vit': float(d_rv),
        'resnet_vs_mlp': float(d_rm),
        'vit_vs_mlp': float(d_vm),
    }

    # Per-target breakdown
    print("\nPer-Target Mean PSNR:")
    print("-" * 50)

    per_target = {target: {arch: [] for arch in architectures.keys()}
                  for target in target_generators.keys()}

    for run in results['runs']:
        per_target[run['target']][run['architecture']].append(run['best_psnr'])

    print(f"{'Target':<15} {'ResNet':>10} {'ViT':>10} {'MLP':>10}")
    print("-" * 50)

    for target in target_generators.keys():
        resnet_mean = np.mean(per_target[target]['ResNet'])
        vit_mean = np.mean(per_target[target]['ViT'])
        mlp_mean = np.mean(per_target[target]['MLP'])
        print(f"{target:<15} {resnet_mean:>10.2f} {vit_mean:>10.2f} {mlp_mean:>10.2f}")

    results['summary']['per_target'] = {
        target: {arch: np.mean(vals) for arch, vals in archs.items()}
        for target, archs in per_target.items()
    }

    # ==========================================
    # 6. SAVE RESULTS
    # ==========================================

    out_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/dip_validation')
    out_dir.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"Results saved to: {out_dir / 'results.json'}")
    print("=" * 70)

    # ==========================================
    # 7. CREATE VISUALIZATION
    # ==========================================

    create_visualization(results, arch_stats, per_target, out_dir)

    return results


def create_visualization(results, arch_stats, per_target, out_dir):
    """Create summary visualization."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Box plot of PSNR by architecture
    ax = axes[0]
    architectures = ['ResNet', 'ViT', 'MLP']
    colors = {'ResNet': '#2ecc71', 'ViT': '#9b59b6', 'MLP': '#e74c3c'}

    data = [arch_stats[arch]['values'] for arch in architectures]
    bp = ax.boxplot(data, labels=architectures, patch_artist=True)

    for patch, arch in zip(bp['boxes'], architectures):
        patch.set_facecolor(colors[arch])
        patch.set_alpha(0.7)

    ax.set_ylabel('Best PSNR (dB)', fontsize=11)
    ax.set_title('Denoising Performance by Architecture\n(25 runs each)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    for i, arch in enumerate(architectures):
        mean_val = arch_stats[arch]['mean']
        ax.scatter([i+1], [mean_val], color='white', s=50, zorder=5,
                   edgecolor='black', linewidth=1)

    # 2. Per-target comparison
    ax = axes[1]
    targets = list(per_target.keys())
    x = np.arange(len(targets))
    width = 0.25

    for i, arch in enumerate(architectures):
        means = [np.mean(per_target[t][arch]) for t in targets]
        ax.bar(x + (i - 1) * width, means, width, label=arch,
               color=colors[arch], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylabel('Mean PSNR (dB)', fontsize=11)
    ax.set_title('Performance by Target Image Type', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Ranking consistency pie chart
    ax = axes[2]

    summary = results['summary']['ranking_consistency']

    # Text summary
    ax.axis('off')

    text = f"""
    RANKING CONSISTENCY
    -------------------

    ResNet > ViT: {summary['resnet_beats_vit']['count']}/{summary['resnet_beats_vit']['total']}
                  ({summary['resnet_beats_vit']['percentage']:.1f}%)

    ResNet > MLP: {summary['resnet_beats_mlp']['count']}/{summary['resnet_beats_mlp']['total']}
                  ({summary['resnet_beats_mlp']['percentage']:.1f}%)

    ViT > MLP:    {summary['vit_beats_mlp']['count']}/{summary['vit_beats_mlp']['total']}
                  ({summary['vit_beats_mlp']['percentage']:.1f}%)

    -------------------

    STATISTICAL TESTS
    (Wilcoxon signed-rank)

    ResNet vs ViT: p={results['summary']['statistical_tests']['resnet_vs_vit']['p_value']:.2e}
    ResNet vs MLP: p={results['summary']['statistical_tests']['resnet_vs_mlp']['p_value']:.2e}
    ViT vs MLP:    p={results['summary']['statistical_tests']['vit_vs_mlp']['p_value']:.2e}

    -------------------

    EFFECT SIZES (Cohen's d)

    ResNet vs ViT: {results['summary']['effect_sizes']['resnet_vs_vit']:.2f}
    ResNet vs MLP: {results['summary']['effect_sizes']['resnet_vs_mlp']:.2f}
    ViT vs MLP:    {results['summary']['effect_sizes']['vit_vs_mlp']:.2f}
    """

    ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title('Statistical Summary', fontsize=12)

    plt.tight_layout()
    plt.savefig(out_dir / 'dip_validation_summary.png', dpi=150)
    plt.savefig(out_dir / 'dip_validation_summary.pdf')
    print(f"Visualization saved to: {out_dir / 'dip_validation_summary.png'}")
    plt.close()


if __name__ == "__main__":
    results = run_validation_experiment()
