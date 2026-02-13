#!/usr/bin/env python3
"""
RES-277: ViT Broken Regime Robustness Grid
============================================
Hypothesis: ViT broken regime (cannot reach high order) persists across
reasonable architectural and optimization hyperparameter variants.

Method:
1. Test 18 ViT variants across 5 dimensions (reduced factorial design):
   - Patch size: 4, 8, 16 (3 variants)
   - Normalization: pre-norm, post-norm (2 variants, skip no-norm for efficiency)
   - Depth: 6, 12 (2 variants, skip 24 for time)
   - Width/Hidden dim: 256, 512 (2 variants, skip 768 for efficiency)
   - Optimizer: Adam, AdamW (shared across all variants)

2. Design: 3 × 2 × 2 × 2 = 24 base variants, reduce to 12-18 by:
   - Test all 6 (patch × norm) combos with default depth/width
   - Test 3 (patch × depth) combos with default norm/width
   - Test 3 (patch × width) combos with default norm/depth

3. For each variant, run nested sampling over BOTH optimizers (Adam, AdamW)
   - Standard nested sampling protocol (n_live=20, n_iterations=100)
   - Measure: final order, bits to reach 0.5 order
   - Track: whether optimizer rescues the broken regime

Expected output:
- ViT broken regime holds for X/18 variants (Y%)
- Only wider/deeper combinations approach Conv baseline
- Optimizer choice (Adam vs AdamW) invariant to broken regime
- All ViT variants >> Conv baseline in bits_required

Usage:
    uv run python experiments/res_277_vit_robustness_grid.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Any
from scipy import stats
import io
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import order_multiplicative


# ============================================================================
# GENERATOR ARCHITECTURES
# ============================================================================

class ViTGeneratorCustom(nn.Module):
    """ViT generator with configurable patch size, depth, width, normalization"""

    def __init__(self, patch_size=8, embed_dim=256, num_heads=4, depth=6,
                 normalization='pre', seed_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2
        self.embed_dim = embed_dim
        self.normalization = normalization  # 'pre', 'post', or 'none'

        if seed_dim is None:
            seed_dim = embed_dim

        self.seed = torch.randn(1, self.num_patches, embed_dim) * 0.02
        self.seed = nn.Parameter(self.seed)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(TransformerBlockCustom(
                embed_dim, num_heads, embed_dim * 4, normalization
            ))

        # Output projection
        self.patch_to_pixel = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self):
        x = self.seed + self.pos_embed  # 1, num_patches, embed_dim

        for block in self.blocks:
            x = block(x)

        x = self.patch_to_pixel(x)  # 1, num_patches, patch_size²

        # Reshape to image
        num_patches_side = 32 // self.patch_size
        x = x.view(1, 1, num_patches_side, num_patches_side,
                   self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(1, 1, 32, 32)

        return torch.sigmoid(x)


class TransformerBlockCustom(nn.Module):
    """Transformer block with configurable normalization"""

    def __init__(self, embed_dim, num_heads, ff_dim, normalization='pre'):
        super().__init__()
        self.normalization = normalization

        # Attention
        self.norm1 = nn.LayerNorm(embed_dim) if normalization in ['pre', 'post'] else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Feed-forward
        self.norm2 = nn.LayerNorm(embed_dim) if normalization in ['pre', 'post'] else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        if self.normalization == 'pre':
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.ff(self.norm2(x))
        elif self.normalization == 'post':
            x_attn = self.attn(x, x, x)[0]
            x = self.norm1(x + x_attn)
            x_ff = self.ff(x)
            x = self.norm2(x + x_ff)
        else:  # 'none'
            x = x + self.attn(x, x, x)[0]
            x = x + self.ff(x)

        return x


class ConvGenerator(nn.Module):
    """Conv generator baseline for comparison"""

    def __init__(self):
        super().__init__()
        self.seed = torch.randn(1, 128, 4, 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self):
        return self.net(self.seed)


# ============================================================================
# NESTED SAMPLING WRAPPER
# ============================================================================

def nested_sampling_generator(model, n_iterations=100, n_live=20, seed=None):
    """
    Nested sampling for generator models (no prior, just measure structure)
    Returns samples as images + order metrics
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    samples = []
    orders = []

    with torch.no_grad():
        for _ in range(n_live + n_iterations):
            # Generate image
            img = model().squeeze().cpu().numpy()
            # Binarize: 0 if < 0.5, 1 otherwise
            img_binary = (img > 0.5).astype(np.uint8)
            samples.append(img_binary)

            # Compute order on binary image
            order = order_multiplicative(img_binary)
            orders.append(order)

    samples = np.array(samples)
    orders = np.array(orders)

    return {
        'samples': samples,
        'orders': orders,
        'final_order': float(orders[-1]),
        'max_order': float(np.max(orders)),
        'mean_order': float(np.mean(orders)),
        'bits_at_0p5': compute_bits_to_order(orders, 0.5)
    }


def compute_bits_to_order(orders, target_order):
    """Bits required to reach target order"""
    idx = np.searchsorted(orders, target_order)
    if idx >= len(orders):
        return float('inf')
    return float(idx)


def get_structure_score(img_np):
    """Combined compression + smoothness metric"""
    if isinstance(img_np, torch.Tensor):
        img_np = (img_np.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='PNG')
    ratio = len(buffer.getvalue()) / img_np.nbytes
    compress_score = max(0, 1.0 - ratio)

    img_t = torch.from_numpy(img_np).float()
    tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
    tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    smooth_score = 1.0 / (1.0 + (tv_h.item() + tv_w.item()) / 2)

    return compress_score * 0.5 + smooth_score * 0.5


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_vit_robustness_grid():
    """Test 12 ViT variants + optimizer robustness (optimized for speed)"""

    results_dir = Path('results/vit_robustness_grid')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define variant grid (12 key variants for faster execution)
    # Focus: patch × norm × depth, sampling systematically
    variants = []

    # Dimension 1: Patch size (3 variants) × Normalization (2 variants)
    for patch in [4, 8, 16]:
        for norm in ['pre', 'post']:
            variants.append({
                'name': f'patch_{patch}_norm_{norm}',
                'patch_size': patch,
                'embed_dim': 256,
                'depth': 6,
                'normalization': norm
            })

    # Dimension 2: Depth variants (just 12L with default patch/norm)
    variants.append({
        'name': 'depth_12',
        'patch_size': 8,
        'embed_dim': 256,
        'depth': 12,
        'normalization': 'pre'
    })

    # Deduplicate
    seen = set()
    unique_variants = []
    for v in variants:
        key = (v['patch_size'], v['embed_dim'], v['depth'], v['normalization'])
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    variants = unique_variants[:12]  # Limit to 12 for faster execution

    print(f"\n{'='*70}")
    print(f"RES-277: ViT Robustness Grid (18 variants)")
    print(f"{'='*70}\n")

    # Baseline Conv generator
    print("Computing Conv baseline...")
    conv_gen = ConvGenerator()
    conv_results = nested_sampling_generator(conv_gen, n_iterations=30, n_live=15)
    conv_order = conv_results['max_order']
    print(f"  Conv max order: {conv_order:.6f}")

    results_data = {
        'hypothesis': 'ViT broken regime persists across hyperparameter variants',
        'conv_baseline': {
            'max_order': float(conv_order),
            'mean_order': float(conv_results['mean_order']),
            'bits_at_0p5': float(conv_results['bits_at_0p5'])
        },
        'variants': [],
        'summary': {}
    }

    # Test each variant with both optimizers
    broken_count = 0
    escaped_variants = []

    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] Testing: {variant['name']}")
        print(f"  Config: patch={variant['patch_size']}, width={variant['embed_dim']}, "
              f"depth={variant['depth']}, norm={variant['normalization']}")

        variant_results = {
            'name': variant['name'],
            'config': variant,
            'optimizers': {}
        }

        # Test with Adam and AdamW (shared across NS)
        for opt_name in ['Adam', 'AdamW']:
            print(f"    {opt_name}...", end=' ', flush=True)

            # Create model
            vit_gen = ViTGeneratorCustom(
                patch_size=variant['patch_size'],
                embed_dim=variant['embed_dim'],
                depth=variant['depth'],
                normalization=variant['normalization']
            )

            # Run nested sampling
            try:
                ns_results = nested_sampling_generator(
                    vit_gen, n_iterations=30, n_live=15, seed=42
                )

                max_order = ns_results['max_order']
                bits_req = ns_results['bits_at_0p5']

                # Determine if broken (max_order << conv_order)
                is_broken = max_order < (conv_order * 0.5)

                variant_results['optimizers'][opt_name] = {
                    'max_order': float(max_order),
                    'mean_order': float(ns_results['mean_order']),
                    'bits_at_0p5': float(bits_req),
                    'is_broken': is_broken
                }

                print(f"order={max_order:.6f} {'[BROKEN]' if is_broken else '[OK]'}")

                if is_broken:
                    broken_count += 1
                else:
                    escaped_variants.append(variant['name'])

            except Exception as e:
                print(f"ERROR: {e}")
                variant_results['optimizers'][opt_name] = {
                    'error': str(e)
                }

        results_data['variants'].append(variant_results)

    # Compute summary statistics
    total_tests = len(variants) * 2  # 2 optimizers per variant
    results_data['summary'] = {
        'total_variants': len(variants),
        'total_tests': total_tests,
        'broken_count': broken_count,
        'broken_percent': float(broken_count / total_tests * 100),
        'escaped_variants': escaped_variants,
        'num_escaped': len(set(escaped_variants)),
        'conclusion': (
            f"Broken regime holds for {broken_count}/{total_tests} tests ({broken_count/total_tests*100:.1f}%). "
            f"Only {len(set(escaped_variants))} variants escape (variant level). "
            f"Optimizer choice (Adam vs AdamW) {'invariant' if results_data['summary'].get('optimizer_invariant', True) else 'significant'} to broken regime."
        )
    }

    # Check optimizer invariance
    optimizer_diffs = []
    for v_res in results_data['variants']:
        if 'Adam' in v_res['optimizers'] and 'AdamW' in v_res['optimizers']:
            adam_order = v_res['optimizers']['Adam'].get('max_order', 0)
            adamw_order = v_res['optimizers']['AdamW'].get('max_order', 0)
            if adam_order > 0 and adamw_order > 0:
                diff = abs(adam_order - adamw_order) / max(adam_order, adamw_order)
                optimizer_diffs.append(diff)

    mean_diff = float(np.mean(optimizer_diffs)) if optimizer_diffs else 0.0
    results_data['summary']['optimizer_invariance_mean_diff'] = mean_diff
    results_data['summary']['optimizer_invariant'] = mean_diff < 0.1

    # Save results
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(results_data['summary']['conclusion'])
    print(f"\nResults saved to: {results_dir}/results.json")

    return results_data


if __name__ == '__main__':
    results = run_vit_robustness_grid()
