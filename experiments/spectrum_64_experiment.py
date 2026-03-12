#!/usr/bin/env python3
"""
Thermodynamic Illumination: The Inductive Bias Spectrum
Scaling to 64x64 RGB: ResNet vs ViT vs MLP

This experiment addresses the modern ConvNet vs Transformer debate:
- ConvNets: Hard-coded inductive bias (locality, translation invariance)
- Transformers: Weak inductive bias (permutation invariant patches + positional embeddings)
- MLPs: No inductive bias (random projection)

Hypothesis:
1. ResNet (Strong Bias): Immediate structure (0 bits).
2. ViT (Medium Bias): Slow emergence of structure.
3. MLP (No Bias): Noise throughout.

Usage:
    uv run python experiments/spectrum_64_experiment.py
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
# 1. ARCHITECTURES (64x64 Output)
# ==========================================

class ResNetGen(nn.Module):
    """
    Convolutional Generator with Residual-style Blocks.
    Bias: Locality, Translation Invariance, Smoothness.
    """
    def __init__(self, channels=3):
        super().__init__()
        # Start at 4x4 -> Upsample to 64x64
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
        self._fixed_input = None

    def forward(self):
        if self._fixed_input is None:
            self._fixed_input = torch.randn(self.input_shape).to(next(self.parameters()).device)
        return self.net(self._fixed_input)


class ViTGen(nn.Module):
    """
    Vision Transformer Generator.
    Bias: Patch-based processing, Permutation invariance (weakened by PosEmbed).
    Generates images patch-by-patch from learnable embeddings.
    """
    def __init__(self, img_size=64, patch_size=8, dim=128, depth=4, heads=4, channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches for 64x64 with 8x8 patches
        self.patch_dim = channels * patch_size * patch_size  # 192 for RGB 8x8

        # Positional Embeddings (The "Seed" - this IS the latent)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Projection to Pixels
        self.to_pixels = nn.Linear(dim, self.patch_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        # 1. Pass Positional Embeddings through Transformer
        # (No external input needed, the pos_embed IS the latent)
        x = self.transformer(self.pos_embed)

        # 2. Project to Patches
        x = self.to_pixels(x)  # [1, num_patches, patch_dim]

        # 3. Apply sigmoid
        x = self.sigmoid(x)

        # 4. Reshape to Image: [1, H*W, C*P*P] -> [1, C, H, W]
        b, n, p = x.shape
        h = w = int(math.sqrt(n))
        p_sz = self.patch_size
        c = self.channels

        # Fold patches into image
        x = x.view(b, h, w, c, p_sz, p_sz)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h * p_sz, w * p_sz)
        return x


class MLPGen(nn.Module):
    """
    Dense Generator.
    Bias: None (Random Projection).
    """
    def __init__(self, channels=3, out_res=64):
        super().__init__()
        self.out_res = out_res
        self.channels = channels
        self.input_shape = (1, 64)

        # Reduced width to keep parameter count manageable at 64x64
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, channels * out_res * out_res),
            nn.Sigmoid()
        )
        self._fixed_input = None

    def forward(self):
        if self._fixed_input is None:
            self._fixed_input = torch.randn(self.input_shape).to(next(self.parameters()).device)
        out = self.net(self._fixed_input)
        return out.view(1, self.channels, self.out_res, self.out_res)


# ==========================================
# 2. METRIC (64x64 Tuned)
# ==========================================

def calculate_order(img_tensor):
    """
    Combined JPEG Compression + Total Variation score.
    Uses soft sigmoid mapping to avoid flatlining in the noise region.
    """
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # A. JPEG Compression Ratio
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    # B. TV Score (Normalized for 64x64)
    # Average pixel diff. White noise ~0.33. Natural images <0.05
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_val = (tv_h + tv_w).item()

    # Soft Sigmoid-like mapping for smoothness to avoid flatlining
    # Maps 0.0 (smooth) -> 1.0, 0.3 (noise) -> ~0.05
    tv_score = math.exp(-10 * tv_val)

    return comp_score * tv_score


# ==========================================
# 3. CORE NESTED SAMPLING (Generic)
# ==========================================

def get_weights_vec(model):
    """Extract all model weights as a single vector."""
    return torch.nn.utils.parameters_to_vector(model.parameters())


def set_weights_vec(model, vec):
    """Set model weights from a single vector."""
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def elliptical_slice_sampling(model, current_weights, threshold, max_attempts=30):
    """
    Samples new weights w' such that score(w') > threshold using ESS.
    Prior: Gaussian N(0, 1) on weights.
    """
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

    return current_weights, threshold  # Fail safe


def run_nested(model_class, name, n_live=10, max_iter=100, verbose=True):
    """
    Runs nested sampling on the weights of a model.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

    model = model_class()
    param_count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {param_count:,}")

    # Initialize Live Points from prior N(0, 1)
    live = []
    for _ in range(n_live):
        w = torch.randn(len(get_weights_vec(model)))
        set_weights_vec(model, w)
        with torch.no_grad():
            score = calculate_order(model())
        live.append({'w': w, 'score': score})
    live.sort(key=lambda x: x['score'])

    results = []
    for i in range(max_iter):
        dead = live.pop(0)
        bits = i / n_live / math.log(2)  # Correct bits calculation
        results.append({'bits': bits, 'score': dead['score']})

        if verbose and i % 20 == 0:
            print(f"  Iter {i:3d}: Bits {bits:5.2f} | Score {dead['score']:.4f}")

        # Sample new point from a random survivor
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

    if verbose:
        final_score = max(r['score'] for r in results)
        print(f"  Final: Max Score = {final_score:.4f}")

    return results


def save_sample_images(model_class, name, n_samples=6, save_path=None):
    """Generate and save sample images from random weights."""
    model = model_class()
    param_len = len(get_weights_vec(model))

    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    fig.suptitle(f'{name} Random Samples (64×64)', fontsize=12)

    for i, ax in enumerate(axes):
        # Random weights from prior
        w = torch.randn(param_len)
        set_weights_vec(model, w)
        with torch.no_grad():
            img = model()
            score = calculate_order(img)

        # Convert to displayable format
        img_np = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img_np)
        ax.set_title(f'Score: {score:.3f}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


# ==========================================
# 4. EXECUTION
# ==========================================

def run_experiment(n_live=15, max_iter=120, n_runs=3):
    """
    Run the full 64x64 spectrum experiment.
    """
    print("=" * 60)
    print("Thermodynamic Illumination: The Inductive Bias Spectrum")
    print("64×64 RGB: ResNet vs ViT vs MLP")
    print("=" * 60)
    print(f"Settings: n_live={n_live}, max_iter={max_iter}, n_runs={n_runs}")

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Save sample images
    print("\nGenerating sample images...")
    save_sample_images(ResNetGen, "ResNet (Conv)", save_path=out_dir / "spectrum64_resnet_samples.png")
    save_sample_images(ViTGen, "ViT (Attention)", save_path=out_dir / "spectrum64_vit_samples.png")
    save_sample_images(MLPGen, "MLP (Dense)", save_path=out_dir / "spectrum64_mlp_samples.png")

    # Run multiple experiments
    all_resnet = []
    all_vit = []
    all_mlp = []

    for run in range(n_runs):
        print(f"\n{'#'*60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'#'*60}")

        res_resnet = run_nested(ResNetGen, "ResNet", n_live, max_iter)
        res_vit = run_nested(ViTGen, "ViT", n_live, max_iter)
        res_mlp = run_nested(MLPGen, "MLP", n_live, max_iter)

        all_resnet.append(res_resnet)
        all_vit.append(res_vit)
        all_mlp.append(res_mlp)

    # Aggregate results
    def aggregate_runs(runs_list):
        n_iters = len(runs_list[0])
        bits = [runs_list[0][i]['bits'] for i in range(n_iters)]
        scores_mean = []
        scores_std = []
        for i in range(n_iters):
            scores_at_i = [run[i]['score'] for run in runs_list]
            scores_mean.append(np.mean(scores_at_i))
            scores_std.append(np.std(scores_at_i))
        return bits, scores_mean, scores_std

    bits_resnet, scores_resnet, std_resnet = aggregate_runs(all_resnet)
    bits_vit, scores_vit, std_vit = aggregate_runs(all_vit)
    bits_mlp, scores_mlp, std_mlp = aggregate_runs(all_mlp)

    # Plot results
    plt.figure(figsize=(10, 6))

    def plot_with_error(bits, scores, std, label, color):
        plt.plot(bits, scores, label=label, color=color, linewidth=2)
        plt.fill_between(bits,
                         np.array(scores) - np.array(std),
                         np.array(scores) + np.array(std),
                         color=color, alpha=0.2)

    plot_with_error(bits_resnet, scores_resnet, std_resnet, "ResNet (Conv) - Strong Bias", "#2ecc71")
    plot_with_error(bits_vit, scores_vit, std_vit, "ViT (Attention) - No Structural Bias", "#f39c12")
    plot_with_error(bits_mlp, scores_mlp, std_mlp, "MLP (Dense) - No Bias", "#e74c3c")

    plt.xlabel("NS depth explored (-log₂ X)", fontsize=11)
    plt.ylabel("Structure Order (Compression × Smoothness)", fontsize=11)
    plt.title("The Inductive Bias Spectrum: 64×64 RGB Architectures", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add threshold line
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = out_dir / "spectrum_64_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.savefig(out_dir / "spectrum_64_comparison.pdf")
    print(f"\nSaved: {save_path}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    def bits_to_threshold(scores, bits, tau):
        for i, s in enumerate(scores):
            if s >= tau:
                return bits[i]
        return float('inf')

    threshold = 0.1
    resnet_bits = bits_to_threshold(scores_resnet, bits_resnet, threshold)
    vit_bits = bits_to_threshold(scores_vit, bits_vit, threshold)
    mlp_bits = bits_to_threshold(scores_mlp, bits_mlp, threshold)

    print(f"\nBits to reach τ={threshold}:")
    print(f"  ResNet: {resnet_bits:.2f} bits" if resnet_bits < float('inf') else f"  ResNet: >{max(bits_resnet):.2f} bits")
    print(f"  ViT:    {vit_bits:.2f} bits" if vit_bits < float('inf') else f"  ViT:    >{max(bits_vit):.2f} bits")
    print(f"  MLP:    {mlp_bits:.2f} bits" if mlp_bits < float('inf') else f"  MLP:    >{max(bits_mlp):.2f} bits")

    print(f"\nFinal scores at {max(bits_resnet):.1f} bits:")
    print(f"  ResNet: {scores_resnet[-1]:.4f} ± {std_resnet[-1]:.4f}")
    print(f"  ViT:    {scores_vit[-1]:.4f} ± {std_vit[-1]:.4f}")
    print(f"  MLP:    {scores_mlp[-1]:.4f} ± {std_mlp[-1]:.4f}")

    # Parameter counts
    print(f"\nParameter counts:")
    print(f"  ResNet: {sum(p.numel() for p in ResNetGen().parameters()):,}")
    print(f"  ViT:    {sum(p.numel() for p in ViTGen().parameters()):,}")
    print(f"  MLP:    {sum(p.numel() for p in MLPGen().parameters()):,}")

    return {
        'resnet': {'bits': bits_resnet, 'scores': scores_resnet, 'std': std_resnet},
        'vit': {'bits': bits_vit, 'scores': scores_vit, 'std': std_vit},
        'mlp': {'bits': bits_mlp, 'scores': scores_mlp, 'std': std_mlp}
    }


if __name__ == "__main__":
    # Run with moderate settings
    results = run_experiment(n_live=15, max_iter=120, n_runs=3)
