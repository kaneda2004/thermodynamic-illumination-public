#!/usr/bin/env python3
"""
Thermodynamic Scaling Law: Expanded with 9x9 Conv and Windowed ViT
===================================================================
Adds two key architectures to fill the "medium bias" gap:
1. ResNet-9x9 (diluted locality - larger receptive field)
2. Local ViT (windowed attention with proper masking)

These are run with the same protocol as comprehensive_scaling_law.py
and results are merged.
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
# ARCHITECTURES
# ==========================================

class ResNet9x9Gen(nn.Module):
    """ResNet with 9x9 kernels - diluted locality bias"""
    def __init__(self, channels=3):
        super().__init__()
        self.input_shape = (1, 128, 4, 4)
        kernel_size = 9
        pad = kernel_size // 2

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size, padding=pad),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(128, 128),  # 8x8
            block(128, 64),   # 16x16
            block(64, 32),    # 32x32
            block(32, 16),    # 64x64
            nn.Conv2d(16, channels, kernel_size, padding=pad),
            nn.Sigmoid()
        )
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class WindowedViTGen(nn.Module):
    """ViT with Local (Windowed) Attention via proper masking"""
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4, window=1):
        super().__init__()
        self.patch_size = patch_size
        self.window = window
        num_patches = (img_size // patch_size) ** 2
        self.h = self.w = img_size // patch_size

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

        # Custom transformer layers to handle masking
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim*2,
                batch_first=True, norm_first=True
            )
            for _ in range(depth)
        ])

        # Create local attention mask
        self.register_buffer('mask', self.create_local_mask(self.h, self.w, window))

        self.to_pixels = nn.Linear(dim, 3 * patch_size * patch_size)
        self.sigmoid = nn.Sigmoid()

    def create_local_mask(self, h, w, window):
        """Create mask where patches can only attend to neighbors within window"""
        n = h * w
        mask = torch.ones(n, n) * float('-inf')  # Default: mask out

        for i in range(h):
            for j in range(w):
                idx = i * w + j
                # Allow attention to neighbors within window
                for di in range(-window, window+1):
                    for dj in range(-window, window+1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            n_idx = ni * w + nj
                            mask[idx, n_idx] = 0.0  # Allow attention
        return mask

    def forward(self):
        x = self.pos_embed

        for layer in self.layers:
            x = layer(x, src_mask=self.mask)

        x = self.to_pixels(x)
        b, n, p = x.shape
        x = x.view(b, self.h, self.w, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, self.h * self.patch_size, self.w * self.patch_size)
        return self.sigmoid(x)


# ==========================================
# MEASUREMENT FUNCTIONS (same as comprehensive)
# ==========================================

def calculate_order(img_tensor):
    """JPEG compression × TV smoothness score"""
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_val = (tv_h + tv_w).item()
    tv_score = math.exp(-10 * tv_val)

    return comp_score * tv_score


def get_weights_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def set_weights_vec(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def measure_structure(model_class, name, n_live=10, max_iter=100, seed=0):
    """Simplified nested sampling for structure measurement"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class()
    param_count = sum(p.numel() for p in model.parameters())

    # Initialize live points
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

        # ESS replacement
        donor = live[np.random.randint(len(live))]
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

        if bits >= 9.5:
            final_score = max(l['score'] for l in live)
            break

    return final_score, param_count


def get_test_data(size=64, noise_level=0.15):
    """Same test image as comprehensive experiment"""
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
    """DIP denoising measurement"""
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

        with torch.no_grad():
            mse_clean = nn.functional.mse_loss(output, x_clean).item()
            psnr = 10 * math.log10(1.0 / max(mse_clean, 1e-10))
            best_psnr = max(best_psnr, psnr)

    return best_psnr


# ==========================================
# MAIN
# ==========================================

ARCHITECTURES = [
    ('ResNet-9x9', ResNet9x9Gen),
    ('WindowedViT', WindowedViTGen),
]


def run_experiment(n_seeds=2):
    """Run expanded experiment"""
    print("=" * 60)
    print("EXPANDED SCALING LAW: 9x9 Conv + Windowed ViT")
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

            structure, params = measure_structure(model_class, name,
                                                   n_live=10, max_iter=100, seed=seed)
            structures.append(structure)

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


def merge_and_plot(new_results):
    """Merge with comprehensive results and create updated figure"""
    results_dir = Path(__file__).parent.parent / 'results'
    fig_dir = Path(__file__).parent.parent / 'figures'

    # Load existing comprehensive results
    with open(results_dir / 'comprehensive_scaling_law.json', 'r') as f:
        existing = json.load(f)

    # Merge
    all_results = existing['architectures'].copy()
    all_results.update(new_results)

    # Calculate new correlation
    names = list(all_results.keys())
    structures = [all_results[n]['structure_mean'] for n in names]
    psnrs = [all_results[n]['psnr_mean'] for n in names]
    structure_errs = [all_results[n]['structure_std'] for n in names]
    psnr_errs = [all_results[n]['psnr_std'] for n in names]

    r, p = stats.pearsonr(structures, psnrs)

    print(f"\n{'='*60}")
    print(f"MERGED RESULTS: {len(names)} architectures")
    print(f"Correlation: r = {r:.3f}, p = {p:.6f}")
    print(f"{'='*60}")

    # Create updated figure
    fig, ax = plt.subplots(figsize=(12, 9))

    colors = {
        'ResNet-4': '#2ecc71', 'ResNet-2': '#27ae60', 'ResNet-6': '#1abc9c',
        'ResNet-9x9': '#16a085',
        'U-Net': '#3498db', 'Depthwise': '#9b59b6', 'LocalAttn': '#e74c3c',
        'CPPN': '#f39c12', 'Fourier': '#e67e22',
        'WindowedViT': '#c0392b',
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

    # Noise floor
    noisy_psnr = 18.21
    ax.axhline(y=noisy_psnr, color='red', linestyle=':', alpha=0.5,
               label=f'Noise floor ({noisy_psnr:.1f} dB)')

    ax.set_xlabel('Thermodynamic Structure Score', fontsize=14)
    ax.set_ylabel('Best PSNR on Clean Target (dB)', fontsize=14)
    ax.set_title(f'The Thermodynamic Scaling Law (n={len(names)}, r={r:.2f}, p={p:.2e})',
                fontsize=16)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'scaling_law_comprehensive.pdf', dpi=150)
    plt.savefig(fig_dir / 'scaling_law_comprehensive.png', dpi=150)
    plt.close()

    # Save merged results
    output = {
        'correlation': {'pearson_r': r, 'p_value': p},
        'architectures': all_results
    }
    with open(results_dir / 'comprehensive_scaling_law.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFigures updated: {fig_dir / 'scaling_law_comprehensive.pdf'}")
    print(f"Results saved: {results_dir / 'comprehensive_scaling_law.json'}")

    # Print table
    print("\n{:<15} {:>10} {:>10}".format("Arch", "Structure", "PSNR"))
    print("-" * 38)
    for name in sorted(names, key=lambda n: -all_results[n]['structure_mean']):
        r = all_results[name]
        print(f"{name:<15} {r['structure_mean']:>10.4f} {r['psnr_mean']:>10.2f}")

    return r, p


if __name__ == '__main__':
    new_results = run_experiment(n_seeds=2)
    r, p = merge_and_plot(new_results)

    print(f"\n{'='*60}")
    print(f"FINAL: r = {r:.3f}, p = {p:.2e}")
    print(f"{'='*60}")
