#!/usr/bin/env python3
"""
Thermodynamic Illumination: Ablations & Mechanisms
1. Hybrid Architecture Test (Conv-Stem ViT vs Pure ViT)
2. Spectral Analysis (Why does structure emerge?)

Hypothesis:
- Hybrid ViT will show strong bias (proving Convolution is the 'active ingredient').
- ConvNets/Hybrids will show 1/f spectral decay (natural image stats).
- Pure ViT/MLP will show flat spectrum (white noise).

Usage:
    uv run python experiments/ablation_hybrid_spectral.py
"""

import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
import math
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. ARCHITECTURES
# ==========================================

class ResNetGen(nn.Module):
    """Convolutional Generator (Strong Bias Baseline)"""
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
        self._fixed_input = None

    def forward(self):
        if self._fixed_input is None:
            self._fixed_input = torch.randn(self.input_shape).to(next(self.parameters()).device)
        return self.net(self._fixed_input)


class ViTGen(nn.Module):
    """Pure Vision Transformer (Soft Bias)"""
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        # Positional Embeddings (The "Seed")
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_pixels = nn.Linear(dim, self.patch_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        x = self.transformer(self.pos_embed)
        x = self.to_pixels(x)

        # Reshape patches to image
        b, n, p = x.shape
        h = w = int(math.sqrt(n))
        x = x.view(b, h, w, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return self.sigmoid(x)


class HybridViTGen(nn.Module):
    """Hybrid: Convolutional Stem + Transformer (Composed Bias)

    This tests whether adding a conv stem to ViT provides the
    thermodynamic advantage of convolutions.
    """
    def __init__(self, img_size=64, dim=128, depth=4, heads=4):
        super().__init__()

        # 1. Conv Stem: Processing 4x4 seed into feature map
        self.seed_shape = (1, 32, 4, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(64, dim, 3, padding=1),
            nn.ReLU()
        )

        # Transformer processes the 8x8 feature map (64 tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Conv Decoder to upsample back to 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, kernel_size=4, stride=4),  # 8x8 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),   # 32x32 -> 64x64
            nn.Sigmoid()
        )

        self._fixed_input = None

    def forward(self):
        if self._fixed_input is None:
            self._fixed_input = torch.randn(self.seed_shape).to(next(self.parameters()).device)

        # Conv Stem
        x = self.stem(self._fixed_input)  # [1, dim, 8, 8]

        # Flatten for Transformer
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [1, 64, dim]

        # Add Pos Embed & Transform
        x = x + self.pos_embed
        x = self.transformer(x)

        # Reshape back for Conv Decoder
        x = x.transpose(1, 2).view(b, c, h, w)

        return self.decoder(x)


class MLPGen(nn.Module):
    """Dense Generator (No Bias Baseline)"""
    def __init__(self, out_res=64):
        super().__init__()
        self.out_res = out_res
        self.input_shape = (1, 64)

        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * out_res * out_res),
            nn.Sigmoid()
        )
        self._fixed_input = None

    def forward(self):
        if self._fixed_input is None:
            self._fixed_input = torch.randn(self.input_shape).to(next(self.parameters()).device)
        out = self.net(self._fixed_input)
        return out.view(1, 3, self.out_res, self.out_res)


# ==========================================
# 2. METRIC & ENGINE
# ==========================================

def calculate_order(img_tensor):
    """Combined JPEG + TV score with soft floor."""
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Compressibility
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    # TV with soft sigmoid mapping
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_score = math.exp(-10 * (tv_h + tv_w).item())

    return comp_score * tv_score


def get_weights_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())


def set_weights_vec(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def elliptical_slice_sampling(model, current_weights, threshold, max_attempts=50):
    """ESS for weight-space sampling."""
    nu = torch.randn_like(current_weights)
    theta = torch.rand(1).item() * 2 * math.pi
    theta_min, theta_max = theta - 2 * math.pi, theta

    for _ in range(max_attempts):
        new_weights = current_weights * math.cos(theta) + nu * math.sin(theta)
        set_weights_vec(model, new_weights)
        with torch.no_grad():
            score = calculate_order(model())

        if score > threshold:
            return new_weights, score
        else:
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = torch.rand(1).item() * (theta_max - theta_min) + theta_min

    return current_weights, threshold


def run_nested(model_class, name, n_live=10, max_iter=100, verbose=True):
    """Run nested sampling and return results + best image for spectral analysis."""
    if verbose:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

    model = model_class()
    param_count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {param_count:,}")

    # Init Live Points
    live = []
    for _ in range(n_live):
        w = torch.randn(len(get_weights_vec(model)))
        set_weights_vec(model, w)
        with torch.no_grad():
            score = calculate_order(model())
        live.append({'w': w, 'score': score})
    live.sort(key=lambda x: x['score'])

    # Main Loop
    results = []
    for i in range(max_iter):
        dead = live.pop(0)
        bits = i / n_live / math.log(2)
        results.append({'bits': bits, 'score': dead['score']})

        if verbose and i % 20 == 0:
            print(f"  Iter {i:3d}: Bits {bits:5.2f} | Score {dead['score']:.4f}")

        survivor = live[np.random.randint(len(live))]
        nw, ns = elliptical_slice_sampling(model, survivor['w'], dead['score'])

        # Insert sorted
        new_pt = {'w': nw, 'score': ns}
        inserted = False
        for idx, p in enumerate(live):
            if ns < p['score']:
                live.insert(idx, new_pt)
                inserted = True
                break
        if not inserted:
            live.append(new_pt)

    # Get best sample for spectral analysis
    best_w = live[-1]['w']
    set_weights_vec(model, best_w)
    with torch.no_grad():
        best_img = model().squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        best_score = calculate_order(model())

    if verbose:
        print(f"  Final best score: {best_score:.4f}")

    return results, best_img


def get_random_sample(model_class):
    """Get a single random sample for spectral comparison."""
    model = model_class()
    w = torch.randn(len(get_weights_vec(model)))
    set_weights_vec(model, w)
    with torch.no_grad():
        img = model().squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return img


# ==========================================
# 3. SPECTRAL ANALYSIS
# ==========================================

def compute_radial_spectrum(img):
    """Compute radially-averaged power spectrum of an image."""
    # Convert to grayscale
    gray = np.mean(img, axis=2)

    # 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2  # Power spectrum

    # Radial averaging
    y, x = np.indices(gray.shape)
    center = np.array(gray.shape) / 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)

    return radialprofile


def plot_spectrum_comparison(images_dict, save_path):
    """Plot spectral comparison across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Sample images
    ax = axes[0]
    n_archs = len(images_dict)
    for i, (name, img) in enumerate(images_dict.items()):
        # Create small subplot
        ax_img = fig.add_axes([0.05 + i * 0.22, 0.55, 0.18, 0.35])
        ax_img.imshow(img)
        ax_img.set_title(name, fontsize=9)
        ax_img.axis('off')

    axes[0].set_visible(False)

    # Right: Spectral analysis
    ax = axes[1]
    colors = {'ResNet': '#2ecc71', 'Hybrid ViT': '#e67e22', 'Pure ViT': '#9b59b6', 'MLP': '#e74c3c'}

    for name, img in images_dict.items():
        spectrum = compute_radial_spectrum(img)
        # Plot log-log
        freqs = np.arange(1, min(32, len(spectrum)))
        ax.loglog(freqs, spectrum[1:len(freqs)+1], label=name,
                  color=colors.get(name, 'gray'), linewidth=2)

    # Add 1/f reference line
    freqs = np.arange(1, 32)
    ax.loglog(freqs, 1e8 / freqs ** 2, '--', color='gray', alpha=0.5, label='1/f² (natural)')

    ax.set_xlabel('Spatial Frequency', fontsize=11)
    ax.set_ylabel('Power', fontsize=11)
    ax.set_title('Spectral Fingerprint: Why Structure Emerges', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_hybrid_comparison(results_dict, save_path):
    """Plot the hybrid ablation results."""
    plt.figure(figsize=(10, 6))

    colors = {
        'ResNet (Conv)': '#2ecc71',
        'Hybrid ViT': '#e67e22',
        'Pure ViT': '#9b59b6',
        'MLP': '#e74c3c'
    }

    for name, results in results_dict.items():
        bits = [r['bits'] for r in results]
        scores = [r['score'] for r in results]
        plt.plot(bits, scores, label=name, color=colors.get(name, 'gray'), linewidth=2)

    plt.xlabel('NS depth explored (-log₂ X)', fontsize=11)
    plt.ylabel('Structure Order', fontsize=11)
    plt.title('Ablation: Does a Conv Stem Give ViT Thermodynamic Advantage?', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.savefig(save_path.with_suffix('.pdf'))
    print(f"Saved: {save_path}")
    plt.close()


def save_sample_grid(images_dict, save_path):
    """Save a grid of sample images."""
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, (name, img) in zip(axes, images_dict.items()):
        ax.imshow(img)
        ax.set_title(name, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


# ==========================================
# 4. EXECUTION
# ==========================================

def run_ablation_experiment(n_live=15, max_iter=100, n_runs=2):
    """Run the full ablation experiment."""
    print("=" * 60)
    print("Thermodynamic Illumination: Ablation Studies")
    print("=" * 60)
    print("1. Hybrid Architecture Test (Conv-Stem ViT vs Pure ViT)")
    print("2. Spectral Analysis (Why does structure emerge?)")
    print("=" * 60)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Run all architectures
    all_results = {
        'ResNet (Conv)': [],
        'Hybrid ViT': [],
        'Pure ViT': [],
        'MLP': []
    }
    best_images = {}

    for run in range(n_runs):
        print(f"\n{'#'*60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'#'*60}")

        res_resnet, img_resnet = run_nested(ResNetGen, "ResNet (Conv)", n_live, max_iter)
        res_hybrid, img_hybrid = run_nested(HybridViTGen, "Hybrid ViT", n_live, max_iter)
        res_vit, img_vit = run_nested(ViTGen, "Pure ViT", n_live, max_iter)
        res_mlp, img_mlp = run_nested(MLPGen, "MLP", n_live, max_iter)

        all_results['ResNet (Conv)'].append(res_resnet)
        all_results['Hybrid ViT'].append(res_hybrid)
        all_results['Pure ViT'].append(res_vit)
        all_results['MLP'].append(res_mlp)

        # Keep best images from last run
        best_images = {
            'ResNet': img_resnet,
            'Hybrid ViT': img_hybrid,
            'Pure ViT': img_vit,
            'MLP': img_mlp
        }

    # Aggregate results (average across runs)
    def aggregate(runs_list):
        n_iters = len(runs_list[0])
        bits = [runs_list[0][i]['bits'] for i in range(n_iters)]
        scores = [np.mean([run[i]['score'] for run in runs_list]) for i in range(n_iters)]
        return [{'bits': b, 'score': s} for b, s in zip(bits, scores)]

    avg_results = {name: aggregate(runs) for name, runs in all_results.items()}

    # Plot hybrid comparison
    plot_hybrid_comparison(avg_results, out_dir / "ablation_hybrid_comparison.png")

    # Save sample images
    save_sample_grid(best_images, out_dir / "ablation_sample_grid.png")

    # Spectral analysis
    plot_spectrum_comparison(best_images, out_dir / "ablation_spectral_analysis.png")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for name, results in avg_results.items():
        final_score = results[-1]['score']
        # Find bits to threshold
        bits_to_01 = None
        for r in results:
            if r['score'] >= 0.1:
                bits_to_01 = r['bits']
                break
        bits_str = f"{bits_to_01:.2f}" if bits_to_01 else f">{results[-1]['bits']:.2f}"
        print(f"{name:20s}: Final={final_score:.4f}, Bits to τ=0.1: {bits_str}")

    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)
    hybrid_final = avg_results['Hybrid ViT'][-1]['score']
    vit_final = avg_results['Pure ViT'][-1]['score']
    resnet_final = avg_results['ResNet (Conv)'][-1]['score']

    if hybrid_final > vit_final * 10:  # Significant improvement
        print("✓ Hybrid ViT shows STRONG improvement over Pure ViT")
        print("  → Convolution IS the active ingredient for structural bias")
    else:
        print("✗ Hybrid ViT shows minimal improvement over Pure ViT")
        print("  → Conv stem alone may not be sufficient")

    if hybrid_final > resnet_final * 0.5:
        print("✓ Hybrid ViT approaches ResNet performance")
    else:
        print("  Hybrid ViT still behind ResNet (pure conv still best)")

    return avg_results, best_images


if __name__ == "__main__":
    results, images = run_ablation_experiment(n_live=15, max_iter=100, n_runs=2)
