"""
Scaling Law Experiment: Kernel Size Sweep
==========================================

Clean experiment varying only kernel size in a ResNet-style decoder.
This gives us 7-10 data points spanning the inductive bias spectrum.

Kernel sizes: 1, 3, 5, 7, 9, 11, 13, 15
- K=1: No spatial mixing (pixel-wise)
- K=3: Standard local bias
- K=7+: Larger receptive fields

We measure:
1. Bits to threshold (via simplified nested sampling)
2. DIP reconstruction capability
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


torch.manual_seed(42)
np.random.seed(42)


class VarKernelGen(nn.Module):
    """ResNet-style decoder with variable kernel size."""
    def __init__(self, kernel_size=3, latent_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.kernel_size = kernel_size
        pad = kernel_size // 2

        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8
            nn.Conv2d(128, 64, kernel_size, padding=pad), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(64, 32, kernel_size, padding=pad), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(32, 16, kernel_size, padding=pad), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 64
            nn.Conv2d(16, out_channels, kernel_size, padding=pad),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.blocks(self.fc(z).view(-1, 128, 4, 4))


def order_metric(img: torch.Tensor) -> float:
    """Compute structure score."""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    comp_ratio = 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return comp_ratio * np.exp(-10 * tv)


def simplified_nested_sampling(model, latent_dim, n_live=50, max_iters=200,
                                threshold=0.1, device='cpu'):
    """Estimate bits to reach threshold."""
    model.eval()

    live_z = torch.randn(n_live, latent_dim, device=device)
    with torch.no_grad():
        live_scores = np.array([order_metric(model(live_z[i:i+1]))
                                for i in range(n_live)])

    log_X = 0.0

    for iteration in range(max_iters):
        worst_idx = np.argmin(live_scores)
        worst_score = live_scores[worst_idx]

        if worst_score >= threshold:
            return -log_X / np.log(2), iteration

        log_X -= 1.0 / n_live

        # Replace via rejection
        for _ in range(500):
            new_z = torch.randn(1, latent_dim, device=device)
            with torch.no_grad():
                new_score = order_metric(model(new_z))
            if new_score > worst_score:
                live_z[worst_idx] = new_z
                live_scores[worst_idx] = new_score
                break
        else:
            return -log_X / np.log(2), iteration

    return -log_X / np.log(2), max_iters


def create_target_image(size=64):
    """Create structured target for DIP."""
    img = torch.zeros(3, size, size)
    # Rectangle
    img[0, 10:30, 10:30] = 1.0
    # Circle
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    circle = ((x - 45)**2 + (y - 45)**2) < 100
    img[1, circle] = 1.0
    # Gradient
    img[2] = torch.linspace(0, 1, size).unsqueeze(0).expand(size, -1)
    return img


def run_dip(model, target, latent_dim, n_iters=300, noise_level=0.15, device='cpu'):
    """Run Deep Image Prior."""
    model.train()
    noisy = (target + noise_level * torch.randn_like(target)).clamp(0, 1).to(device)
    target = target.to(device)

    z = torch.randn(1, latent_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_psnr, best_iter = 0, 0

    for i in range(n_iters):
        optimizer.zero_grad()
        out = model(z)
        loss = F.mse_loss(out, noisy.unsqueeze(0))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mse = F.mse_loss(out, target.unsqueeze(0))
            psnr = 10 * torch.log10(1 / (mse + 1e-10)).item()
            if psnr > best_psnr:
                best_psnr, best_iter = psnr, i

    noisy_mse = F.mse_loss(noisy, target)
    noisy_psnr = 10 * torch.log10(1 / (noisy_mse + 1e-10)).item()

    return best_psnr, noisy_psnr, best_iter


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    latent_dim = 128
    kernel_sizes = [1, 3, 5, 7, 9, 11, 13, 15]
    target = create_target_image(64)

    results = {}

    print("\n" + "="*70)
    print("KERNEL SIZE SWEEP EXPERIMENT")
    print("="*70)

    for k in kernel_sizes:
        print(f"\n--- Kernel Size {k}×{k} ---")

        # Measure bits
        model = VarKernelGen(kernel_size=k, latent_dim=latent_dim).to(device)
        bits, iters = simplified_nested_sampling(model, latent_dim, n_live=50,
                                                  max_iters=150, threshold=0.1, device=device)
        print(f"  Bits to τ=0.1: {bits:.2f} ({iters} iters)")

        # Measure structure score
        model.eval()
        scores = []
        with torch.no_grad():
            for _ in range(30):
                z = torch.randn(1, latent_dim, device=device)
                scores.append(order_metric(model(z)))
        struct_mean = np.mean(scores)
        print(f"  Structure Score: {struct_mean:.4f}")

        # Run DIP
        model = VarKernelGen(kernel_size=k, latent_dim=latent_dim).to(device)
        best_psnr, noisy_psnr, best_iter = run_dip(model, target, latent_dim, device=device)
        gap = best_psnr - noisy_psnr
        print(f"  DIP: {best_psnr:.2f}dB (gap {gap:+.2f}dB)")

        results[k] = {
            'bits': bits,
            'structure': struct_mean,
            'best_psnr': best_psnr,
            'noisy_psnr': noisy_psnr,
            'gap': gap,
            'best_iter': best_iter,
        }

    # Compute correlations
    ks = list(results.keys())
    bits_list = [results[k]['bits'] for k in ks]
    structs = [results[k]['structure'] for k in ks]
    psnrs = [results[k]['best_psnr'] for k in ks]
    gaps = [results[k]['gap'] for k in ks]

    # Low bits = high structure, so negate bits for correlation with PSNR
    r_bits_psnr, p1 = stats.pearsonr([-b for b in bits_list], psnrs)
    r_struct_psnr, p2 = stats.pearsonr(structs, psnrs)
    r_bits_gap, p3 = stats.pearsonr([-b for b in bits_list], gaps)

    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    print(f"(-Bits) vs PSNR: r = {r_bits_psnr:.3f}, p = {p1:.4f}")
    print(f"Structure vs PSNR: r = {r_struct_psnr:.3f}, p = {p2:.4f}")
    print(f"(-Bits) vs Gap: r = {r_bits_gap:.3f}, p = {p3:.4f}")

    # Save
    results_dir = Path('results/scaling_law_kernel_sweep')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'results.json', 'w') as f:
        json.dump({
            'by_kernel': {str(k): v for k, v in results.items()},
            'correlations': {
                'neg_bits_vs_psnr': {'r': r_bits_psnr, 'p': p1},
                'structure_vs_psnr': {'r': r_struct_psnr, 'p': p2},
                'neg_bits_vs_gap': {'r': r_bits_gap, 'p': p3},
            }
        }, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(bits_list, psnrs, s=100, c='steelblue', edgecolors='black')
    for i, k in enumerate(ks):
        ax.annotate(f'{k}×{k}', (bits_list[i], psnrs[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Bits to τ=0.1')
    ax.set_ylabel('Best PSNR (dB)')
    ax.set_title(f'Bits vs PSNR (r={r_bits_psnr:.3f})')
    ax.invert_xaxis()  # Low bits = better
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ['green' if g > 0 else 'red' for g in gaps]
    ax.scatter(bits_list, gaps, s=100, c=colors, edgecolors='black')
    for i, k in enumerate(ks):
        ax.annotate(f'{k}×{k}', (bits_list[i], gaps[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bits to τ=0.1')
    ax.set_ylabel('Denoising Gap (dB)')
    ax.set_title(f'Bits vs Denoising (r={r_bits_gap:.3f})')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/scaling_law_kernel_sweep.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/scaling_law_kernel_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to figures/scaling_law_kernel_sweep.pdf")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Kernel':<10} {'Bits':>8} {'Struct':>8} {'PSNR':>8} {'Gap':>8}")
    print("-"*45)
    for k in ks:
        r = results[k]
        print(f"{k}×{k:<8} {r['bits']:>8.2f} {r['structure']:>8.4f} {r['best_psnr']:>8.2f} {r['gap']:>+8.2f}")

    print(f"\n{'='*70}")
    print(f"FINAL: r = {r_bits_gap:.3f} (n = {len(ks)} kernel sizes)")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
