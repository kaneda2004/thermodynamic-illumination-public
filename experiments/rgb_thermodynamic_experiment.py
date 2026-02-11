#!/usr/bin/env python3
"""
Thermodynamic Illumination: Non-Toy RGB Experiment

Scales the framework from binary 32x32 to continuous RGB 32x32x3 images
using realistic neural network architectures.

Hypothesis: ConvNet should reach higher structure scores with fewer bits
of prior volume than LinearNet, proving stronger inductive bias.

Usage:
    uv run python experiments/rgb_thermodynamic_experiment.py
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

class ConvGen(nn.Module):
    """
    Convolutional Generator (Deep Image Prior style).
    Strong inductive bias for spatial locality and smoothness.
    """
    def __init__(self, channels=3):
        super().__init__()
        # Input: Fixed seed (1, 64, 4, 4)
        self.input_shape = (1, 64, 4, 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Sigmoid()  # Output [0, 1]
        )
        self._fixed_input = None

    def forward(self, x=None):
        # We sample *weights*, so input is fixed
        if self._fixed_input is None or self._fixed_input.device != next(self.parameters()).device:
            self._fixed_input = torch.ones(self.input_shape).to(next(self.parameters()).device)
        return self.net(self._fixed_input)


class LinearGen(nn.Module):
    """
    MLP/Linear Generator.
    Weak inductive bias; treats pixels largely as independent outputs.
    """
    def __init__(self, channels=3, out_res=32):
        super().__init__()
        self.out_res = out_res
        self.channels = channels
        # Input: Fixed seed (1, 128)
        self.input_shape = (1, 128)
        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, channels * out_res * out_res),
            nn.Sigmoid()
        )
        self._fixed_input = None

    def forward(self, x=None):
        if self._fixed_input is None or self._fixed_input.device != next(self.parameters()).device:
            self._fixed_input = torch.ones(self.input_shape).to(next(self.parameters()).device)
        out = self.net(self._fixed_input)
        return out.view(1, self.channels, self.out_res, self.out_res)


# ==========================================
# 2. RGB ORDER METRIC
# ==========================================

def calculate_order(img_tensor):
    """
    Quantifies 'Structure' of an RGB image.
    Range: [0, 1] (1 = highly structured, 0 = noise)

    Combines:
    1. Compressibility (JPEG ratio) - structured images compress well
    2. Smoothness (Total Variation) - natural images have low TV
    """
    # Detach and convert to [0, 255] uint8
    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # A. Compressibility (JPEG ratio)
    buffer = io.BytesIO()
    pil_img = Image.fromarray(img_np)
    pil_img.save(buffer, format='JPEG', quality=85)
    compressed_size = len(buffer.getvalue())
    raw_size = img_np.nbytes
    # Higher ratio = more compressible = more structure
    # Random noise is hard to compress
    comp_score = 1.0 - (compressed_size / raw_size)
    comp_score = max(0, comp_score)

    # B. Smoothness (Total Variation)
    # TV: sum of absolute differences between adjacent pixels
    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    # Normalize: random uniform noise has TV ~0.33, smooth images ~0.01-0.1
    tv_raw = (tv_h + tv_w).item()
    # Map to [0, 1]: smooth (tv<0.05) -> 1.0, noisy (tv>0.4) -> 0.0
    tv_score = max(0, min(1, 1.0 - tv_raw * 2.5))

    # Multiplicative combination (like the paper)
    return comp_score * tv_score


# ==========================================
# 3. NESTED SAMPLING ENGINE (Weight Space)
# ==========================================

def get_weights_vec(model):
    """Extract all model weights as a single vector."""
    return torch.nn.utils.parameters_to_vector(model.parameters())


def set_weights_vec(model, vec):
    """Set model weights from a single vector."""
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def elliptical_slice_sampling(model, current_weights, threshold, max_attempts=50):
    """
    Samples new weights w' such that O(w') > threshold using ESS.
    Prior: Gaussian N(0, 1) on weights.

    ESS is exact (no rejection) for Gaussian priors, making it efficient
    for high-dimensional weight spaces (>10,000 params).
    """
    # 1. Sample auxiliary nu ~ N(0, I)
    nu = torch.randn_like(current_weights)

    # 2. Setup ellipse bracket
    theta = torch.rand(1).item() * 2 * math.pi
    theta_min = theta - 2 * math.pi
    theta_max = theta

    # 3. Slice sampling loop
    for _ in range(max_attempts):
        # Propose new weights on ellipse
        new_weights = current_weights * math.cos(theta) + nu * math.sin(theta)

        # Evaluate
        set_weights_vec(model, new_weights)
        with torch.no_grad():
            img = model()
            score = calculate_order(img)

        if score > threshold:
            return new_weights, score
        else:
            # Shrink bracket toward current position
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = torch.rand(1).item() * (theta_max - theta_min) + theta_min

    # Fallback: return current (shouldn't happen often with proper setup)
    return current_weights, threshold


def nested_sampling(model_class, num_live=20, max_iter=200, verbose=True):
    """
    Runs Nested Sampling on the weights of a model.

    Returns a list of {iter, bits, score} dictionaries tracking
    the exploration of prior volume.
    """
    model = model_class()
    param_len = len(get_weights_vec(model))

    if verbose:
        print(f"Model: {model_class.__name__} | Params: {param_len:,}")

    # Initialize live points from prior N(0, 1)
    live_points = []
    for _ in range(num_live):
        w = torch.randn(param_len)
        set_weights_vec(model, w)
        with torch.no_grad():
            score = calculate_order(model())
        live_points.append({'w': w, 'score': score})

    # Sort by score (ascending - lowest first)
    live_points.sort(key=lambda x: x['score'])

    results = []

    # Main nested sampling loop
    for i in range(max_iter):
        # 1. Record dead point (lowest score)
        dead = live_points.pop(0)
        log_vol = -i / num_live  # Log volume shrinkage
        bits = -log_vol / math.log(2)
        results.append({'iter': i, 'bits': bits, 'score': dead['score']})

        if verbose and i % 20 == 0:
            print(f"  Iter {i:3d}: Bits={bits:5.2f}, Score={dead['score']:.4f}")

        # 2. Sample replacement from a random survivor
        survivor_idx = np.random.randint(len(live_points))
        survivor = live_points[survivor_idx]

        # ESS to find new sample with score > threshold
        new_w, new_score = elliptical_slice_sampling(
            model, survivor['w'], dead['score']
        )

        # 3. Insert new point (maintain sorted order)
        new_point = {'w': new_w, 'score': new_score}
        inserted = False
        for idx, p in enumerate(live_points):
            if new_score < p['score']:
                live_points.insert(idx, new_point)
                inserted = True
                break
        if not inserted:
            live_points.append(new_point)

    if verbose:
        final_score = max(r['score'] for r in results)
        print(f"  Final: Max Score={final_score:.4f} at {results[-1]['bits']:.2f} bits")

    return results


def save_sample_images(model_class, n_samples=6, save_path=None):
    """Generate and save sample images from random weights."""
    model = model_class()
    param_len = len(get_weights_vec(model))

    fig, axes = plt.subplots(1, n_samples, figsize=(12, 2.5))
    fig.suptitle(f'{model_class.__name__} Random Samples', fontsize=12)

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
# 4. RUN EXPERIMENT
# ==========================================

def run_experiment(n_live=15, max_iter=150, n_runs=3):
    """
    Run the full thermodynamic illumination experiment comparing
    ConvNet vs LinearNet on RGB image generation.
    """
    print("=" * 60)
    print("Thermodynamic Illumination: RGB Experiment")
    print("=" * 60)
    print(f"Settings: n_live={n_live}, max_iter={max_iter}, n_runs={n_runs}")
    print()

    # Create output directory
    out_dir = Path('figures')
    out_dir.mkdir(exist_ok=True)

    # Save sample images first
    print("Generating sample images...")
    save_sample_images(ConvGen, save_path=out_dir / 'rgb_convnet_samples.png')
    save_sample_images(LinearGen, save_path=out_dir / 'rgb_linearnet_samples.png')
    print()

    # Run multiple experiments and average
    all_conv = []
    all_linear = []

    for run in range(n_runs):
        print(f"--- Run {run + 1}/{n_runs} ---")

        print("ConvNet:")
        res_conv = nested_sampling(ConvGen, num_live=n_live, max_iter=max_iter)
        all_conv.append(res_conv)

        print("LinearNet:")
        res_linear = nested_sampling(LinearGen, num_live=n_live, max_iter=max_iter)
        all_linear.append(res_linear)
        print()

    # Aggregate results
    def aggregate_runs(runs_list):
        """Average scores across runs at each iteration."""
        n_iters = len(runs_list[0])
        bits = [runs_list[0][i]['bits'] for i in range(n_iters)]
        scores_mean = []
        scores_std = []
        for i in range(n_iters):
            scores_at_i = [run[i]['score'] for run in runs_list]
            scores_mean.append(np.mean(scores_at_i))
            scores_std.append(np.std(scores_at_i))
        return bits, scores_mean, scores_std

    bits_conv, scores_conv, std_conv = aggregate_runs(all_conv)
    bits_lin, scores_lin, std_lin = aggregate_runs(all_linear)

    # Plot results
    plt.figure(figsize=(10, 6))

    # ConvNet
    plt.plot(bits_conv, scores_conv, label='ConvNet (Strong Bias)',
             linewidth=2, color='#2ecc71')
    plt.fill_between(bits_conv,
                     np.array(scores_conv) - np.array(std_conv),
                     np.array(scores_conv) + np.array(std_conv),
                     color='#2ecc71', alpha=0.2)

    # LinearNet
    plt.plot(bits_lin, scores_lin, label='LinearNet (Weak Bias)',
             linewidth=2, color='#e74c3c')
    plt.fill_between(bits_lin,
                     np.array(scores_lin) - np.array(std_lin),
                     np.array(scores_lin) + np.array(std_lin),
                     color='#e74c3c', alpha=0.2)

    plt.xlabel('Bits of Prior Volume Explored (-log₂ V)', fontsize=11)
    plt.ylabel('Structure Order (Smoothness × Compression)', fontsize=11)
    plt.title('Thermodynamic Illumination: Conv vs Linear on RGB 32×32 Images', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add threshold line
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='τ=0.1')

    plt.tight_layout()
    save_path = out_dir / 'rgb_thermodynamic_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.savefig(out_dir / 'rgb_thermodynamic_comparison.pdf')
    print(f"Saved: {save_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Find bits to reach threshold
    threshold = 0.1

    def bits_to_threshold(scores, bits, tau):
        for i, s in enumerate(scores):
            if s >= tau:
                return bits[i]
        return float('inf')

    conv_bits = bits_to_threshold(scores_conv, bits_conv, threshold)
    lin_bits = bits_to_threshold(scores_lin, bits_lin, threshold)

    print(f"Bits to reach τ={threshold}:")
    print(f"  ConvNet:   {conv_bits:.2f} bits")
    print(f"  LinearNet: {lin_bits:.2f} bits" if lin_bits < float('inf') else f"  LinearNet: >{max(bits_lin):.2f} bits (never reached)")

    if conv_bits < float('inf') and lin_bits < float('inf'):
        ratio = 2 ** (lin_bits - conv_bits)
        print(f"  Efficiency ratio: {ratio:.1f}× more samples needed for LinearNet")
    elif conv_bits < float('inf'):
        print(f"  ConvNet reached threshold; LinearNet did not")

    print(f"\nFinal scores at {max(bits_conv):.1f} bits:")
    print(f"  ConvNet:   {scores_conv[-1]:.4f} ± {std_conv[-1]:.4f}")
    print(f"  LinearNet: {scores_lin[-1]:.4f} ± {std_lin[-1]:.4f}")

    return {
        'conv': {'bits': bits_conv, 'scores': scores_conv, 'std': std_conv},
        'linear': {'bits': bits_lin, 'scores': scores_lin, 'std': std_lin}
    }


if __name__ == "__main__":
    # Run with moderate settings for demonstration
    # Increase n_live and max_iter for publication-quality results
    results = run_experiment(n_live=15, max_iter=150, n_runs=3)
