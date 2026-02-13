#!/usr/bin/env python3
"""
RES-292: Large-Scale Thermodynamic Volume Validation

Monte Carlo validation of thermodynamic volume across 11 neural architectures.
This experiment provides independent validation of nested sampling results by
directly measuring the fraction of parameter space producing structured outputs.

Hypothesis: The thermodynamic volume (fraction of parameter space with Order > τ)
varies by 5+ orders of magnitude across architectures, with coordinate-based priors
(CPPN, Fourier) having high volume and latent decoders (MLP, ViT) having near-zero volume.

Method:
1. Sample 10,000 random weight initializations per architecture
2. Generate image from each initialization
3. Compute multiplicative Order metric for each image
4. Calculate fraction exceeding thresholds [0.05, 0.1, 0.2, 0.3, 0.5]

Expected runtime: ~2 hours (10,000 samples × 11 architectures)

Usage:
    cd /Users/matt/Development/monochrome_noise_converger
    uv run python experiments/res_292_thermodynamic_volume_validation.py
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import zlib
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Ensure project root is in path
PROJECT_ROOT = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_SAMPLES = 10000
IMAGE_SIZE = 64
ORDER_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]
RANDOM_SEED = 42

# Architectures to test (skip GaussianProcess - too slow for 10k samples)
ARCHITECTURES = [
    'CPPN', 'CoordMLP', 'FourierMLP',
    'MLPDecoder', 'ConvDecoder', 'ViTDecoder',
    'FourierBasis',
    'WalkCarver', 'SpanningTreeMaze',
    'LSTMDecoder', 'MixtureOfExperts',
]


# =============================================================================
# ORDER METRIC (from paper_2/src/metrics/thermodynamic.py)
# =============================================================================

def compute_compressibility(img: np.ndarray) -> float:
    """Bit-level compressibility with tiling to overcome zlib header overhead."""
    if img.size == 0:
        return 0.0
    img = (img > 0.5).astype(np.uint8)
    tiled = np.tile(img, (2, 2))
    packed = np.packbits(tiled.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0.0, min(1.0, 1 - (compressed_bits / raw_bits)))


def compute_edge_density(img: np.ndarray) -> float:
    """Fraction of pixels adjacent to different-valued pixel."""
    if img.size == 0:
        return 0.0
    img = (img > 0.5).astype(np.uint8)
    padded = np.pad(img, 1, mode='edge')
    edges = 0
    for di, dj in [(0, 1), (1, 0)]:
        shifted = padded[1+di:1+di+img.shape[0], 1+dj:1+dj+img.shape[1]]
        edges += np.sum(img != shifted)
    return edges / (2 * img.size)


def compute_spectral_coherence(img: np.ndarray) -> float:
    """FFT-based coherence: ratio of low-frequency power to total power."""
    if img.size == 0:
        return 0.0
    img_float = img.astype(float) - 0.5
    f = np.fft.fft2(img_float)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (min(h, w) / 4)
    low_power = np.sum(power[low_mask])
    total_power = np.sum(power) + 1e-10
    return min(1.0, low_power / total_power)


def compute_symmetry(img: np.ndarray) -> float:
    """Average of horizontal and vertical reflection symmetry."""
    if img.size == 0:
        return 0.0
    h_sym = np.mean(img == np.fliplr(img))
    v_sym = np.mean(img == np.flipud(img))
    return (h_sym + v_sym) / 2


def _compute_connected_components(img: np.ndarray) -> int:
    """Count foreground connected components using flood fill."""
    if img.size == 0:
        return 0
    img = (img > 0.5).astype(np.uint8)
    visited = np.zeros_like(img, dtype=bool)
    count = 0

    def flood_fill(i: int, j: int) -> None:
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= img.shape[0] or cj < 0 or cj >= img.shape[1]:
                continue
            if visited[ci, cj] or img[ci, cj] != 1:
                continue
            visited[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not visited[i, j] and img[i, j] == 1:
                flood_fill(i, j)
                count += 1
    return count


def _gaussian_gate(x: float, center: float, sigma: float) -> float:
    """Bell curve gate: peaks at center, falls off with sigma."""
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def compute_order(img: np.ndarray, resolution_ref: int = 32) -> float:
    """
    Scale-invariant multiplicative order metric.

    Image must satisfy ALL of:
    - Non-trivial density (not empty/full)
    - Some edges (not solid block)
    - Spectral coherence (not noise)
    - Reasonable compressibility (structured, not trivial)
    """
    if img.size == 0:
        return 0.0

    resolution = img.shape[0]
    scale_factor = resolution_ref / resolution

    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)
    symmetry = compute_symmetry(img)
    components = _compute_connected_components(img)

    # Gate 1: Density (bell curve centered at 0.5)
    density_gate = _gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density (ADAPTIVE - scale with resolution)
    edge_gate = _gaussian_gate(
        edge_density,
        center=0.15 * scale_factor,
        sigma=0.08 * scale_factor
    )

    # Gate 3: Coherence (sigmoid, threshold at 0.3)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility (piecewise linear)
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Bonuses for additional structure
    symmetry_bonus = 0.3 * symmetry

    if components == 0:
        component_bonus = 0
    elif components <= 5:
        component_bonus = 0.2 * (components / 5)
    else:
        component_bonus = max(0, 0.2 * (1 - (components - 5) / 20))

    base_score = density_gate * edge_gate * coherence_gate * compress_gate
    raw_score = base_score * (1 + symmetry_bonus + component_bonus)

    return min(1.0, raw_score)


# =============================================================================
# ARCHITECTURE IMPLEMENTATIONS (self-contained, no external dependencies)
# =============================================================================

class BaseArchitecture(ABC):
    """Abstract base class for all architectures."""

    def __init__(self, sigma: float = 1.0, size: int = 64):
        self.sigma = sigma
        self.size = size
        self._param_count = 0

    @abstractmethod
    def sample_random_init(self) -> None:
        """Initialize all weights from N(0, sigma^2)."""
        pass

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate a size x size grayscale image in [0, 1]."""
        pass

    def sample(self) -> np.ndarray:
        """Convenience: reinit and generate."""
        self.sample_random_init()
        return self.generate()

    def get_param_count(self) -> int:
        return self._param_count


class CPPN(BaseArchitecture):
    """Compositional Pattern Producing Network with periodic activations."""

    ACTIVATIONS = {
        'sin': lambda x: np.sin(x * np.pi),
        'cos': lambda x: np.cos(x * np.pi),
        'gauss': lambda x: np.exp(-x**2 * 2),
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10))),
        'tanh': np.tanh,
    }

    def __init__(self, sigma: float = 1.0, size: int = 64):
        super().__init__(sigma, size)
        self.n_inputs = 4  # x, y, r, bias

    def sample_random_init(self) -> None:
        self.w1 = np.random.randn(self.n_inputs, 8) * self.sigma
        self.b1 = np.random.randn(8) * self.sigma
        self.w2 = np.random.randn(8, 8) * self.sigma
        self.b2 = np.random.randn(8) * self.sigma
        self.w3 = np.random.randn(8, 1) * self.sigma
        self.b3 = np.random.randn(1) * self.sigma
        self.activations = np.random.choice(list(self.ACTIVATIONS.keys()), size=16, replace=True)
        self._param_count = self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.w3.size + self.b3.size

    def generate(self) -> np.ndarray:
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        inputs = np.stack([x.flatten(), y.flatten(), r.flatten(), bias.flatten()], axis=1)

        h1 = inputs @ self.w1 + self.b1
        for i in range(8):
            h1[:, i] = self.ACTIVATIONS[self.activations[i]](h1[:, i])

        h2 = h1 @ self.w2 + self.b2
        for i in range(8):
            h2[:, i] = self.ACTIVATIONS[self.activations[8 + i]](h2[:, i])

        out = self.ACTIVATIONS['sigmoid'](h2 @ self.w3 + self.b3)
        return out.reshape(self.size, self.size).astype(np.float32)


class CoordMLP(BaseArchitecture):
    """Standard MLP on (x, y) coordinates with ReLU."""

    def __init__(self, sigma: float = 1.0, size: int = 64, hidden_dim: int = 32):
        super().__init__(sigma, size)
        self.hidden_dim = hidden_dim

    def sample_random_init(self) -> None:
        self.w1 = np.random.randn(2, self.hidden_dim) * self.sigma
        self.b1 = np.random.randn(self.hidden_dim) * self.sigma
        self.w2 = np.random.randn(self.hidden_dim, self.hidden_dim) * self.sigma
        self.b2 = np.random.randn(self.hidden_dim) * self.sigma
        self.w3 = np.random.randn(self.hidden_dim, 1) * self.sigma
        self.b3 = np.random.randn(1) * self.sigma
        self._param_count = self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.w3.size + self.b3.size

    def generate(self) -> np.ndarray:
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        inputs = np.stack([x.flatten(), y.flatten()], axis=1)
        h1 = np.maximum(0, inputs @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        out = 1 / (1 + np.exp(-np.clip(h2 @ self.w3 + self.b3, -10, 10)))
        return out.reshape(self.size, self.size).astype(np.float32)


class FourierMLP(BaseArchitecture):
    """MLP with Fourier feature encoding of coordinates."""

    def __init__(self, sigma: float = 1.0, size: int = 64, n_frequencies: int = 8):
        super().__init__(sigma, size)
        self.n_frequencies = n_frequencies
        self.input_dim = 4 * n_frequencies

    def sample_random_init(self) -> None:
        hidden = 32
        self.frequencies = np.random.randn(self.n_frequencies) * 3.0
        self.w1 = np.random.randn(self.input_dim, hidden) * self.sigma
        self.b1 = np.random.randn(hidden) * self.sigma
        self.w2 = np.random.randn(hidden, 1) * self.sigma
        self.b2 = np.random.randn(1) * self.sigma
        self._param_count = self.frequencies.size + self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def generate(self) -> np.ndarray:
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        x_flat, y_flat = x.flatten(), y.flatten()

        features = []
        for freq in self.frequencies:
            features.extend([
                np.sin(freq * np.pi * x_flat), np.cos(freq * np.pi * x_flat),
                np.sin(freq * np.pi * y_flat), np.cos(freq * np.pi * y_flat),
            ])
        inputs = np.stack(features, axis=1)
        h1 = np.maximum(0, inputs @ self.w1 + self.b1)
        out = 1 / (1 + np.exp(-np.clip(h1 @ self.w2 + self.b2, -10, 10)))
        return out.reshape(self.size, self.size).astype(np.float32)


class MLPDecoder(BaseArchitecture):
    """Standard MLP decoder from latent z to image."""

    def __init__(self, sigma: float = 1.0, size: int = 64, latent_dim: int = 16):
        super().__init__(sigma, size)
        self.latent_dim = latent_dim

    def sample_random_init(self) -> None:
        hidden = 128
        output_dim = self.size * self.size
        self.z = np.random.randn(self.latent_dim)
        self.w1 = np.random.randn(self.latent_dim, hidden) * self.sigma
        self.b1 = np.random.randn(hidden) * self.sigma
        self.w2 = np.random.randn(hidden, hidden) * self.sigma
        self.b2 = np.random.randn(hidden) * self.sigma
        self.w3 = np.random.randn(hidden, output_dim) * self.sigma
        self.b3 = np.random.randn(output_dim) * self.sigma
        self._param_count = self.z.size + self.w1.size + self.b1.size + self.w2.size + self.b2.size + self.w3.size + self.b3.size

    def generate(self) -> np.ndarray:
        h1 = np.maximum(0, self.z @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        out = 1 / (1 + np.exp(-np.clip(h2 @ self.w3 + self.b3, -10, 10)))
        return out.reshape(self.size, self.size).astype(np.float32)


class ConvDecoder(BaseArchitecture):
    """Convolutional decoder from latent grid to image."""

    def __init__(self, sigma: float = 1.0, size: int = 64, n_channels: int = 16):
        super().__init__(sigma, size)
        self.n_channels = n_channels

    def sample_random_init(self) -> None:
        self.latent = np.random.randn(self.n_channels, 4, 4) * self.sigma
        n_stages = int(np.log2(self.size // 4))
        self.kernels = []
        in_ch = self.n_channels
        for i in range(n_stages):
            out_ch = max(1, in_ch // 2) if i < n_stages - 1 else 1
            kernel = np.random.randn(in_ch, out_ch, 3, 3) * self.sigma
            self.kernels.append(kernel)
            in_ch = out_ch
        self._param_count = self.latent.size + sum(k.size for k in self.kernels)

    def _upsample_2x(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(x, 2, axis=-1), 2, axis=-2)

    def _conv2d(self, x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        in_ch, out_ch, kh, kw = kernel.shape
        h, w = x.shape[-2:]
        x_padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='edge')
        out = np.zeros((out_ch, h, w))
        for oc in range(out_ch):
            for ic in range(in_ch):
                for di in range(kh):
                    for dj in range(kw):
                        out[oc] += x_padded[ic, di:di+h, dj:dj+w] * kernel[ic, oc, di, dj]
        return out

    def generate(self) -> np.ndarray:
        x = self.latent.copy()
        for kernel in self.kernels:
            x = self._upsample_2x(x)
            x = self._conv2d(x, kernel)
            if x.shape[0] > 1:
                x = np.maximum(0, x)
        out = 1 / (1 + np.exp(-np.clip(x[0], -10, 10)))
        if out.shape[0] != self.size:
            scale = self.size / out.shape[0]
            out_resized = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    si, sj = i / scale, j / scale
                    i0, j0 = int(si), int(sj)
                    i1, j1 = min(i0 + 1, out.shape[0] - 1), min(j0 + 1, out.shape[1] - 1)
                    di, dj = si - i0, sj - j0
                    out_resized[i, j] = out[i0, j0] * (1 - di) * (1 - dj) + out[i1, j0] * di * (1 - dj) + out[i0, j1] * (1 - di) * dj + out[i1, j1] * di * dj
            out = out_resized
        return out.astype(np.float32)


class ViTDecoder(BaseArchitecture):
    """Simplified Vision Transformer decoder using self-attention."""

    def __init__(self, sigma: float = 1.0, size: int = 64, patch_size: int = 8, d_model: int = 32):
        super().__init__(sigma, size)
        self.patch_size = patch_size
        self.n_patches = (size // patch_size) ** 2
        self.d_model = d_model

    def sample_random_init(self) -> None:
        self.patch_embeds = np.random.randn(self.n_patches, self.d_model) * self.sigma
        self.pos_embeds = np.random.randn(self.n_patches, self.d_model) * self.sigma
        self.Wq = np.random.randn(self.d_model, self.d_model) * self.sigma
        self.Wk = np.random.randn(self.d_model, self.d_model) * self.sigma
        self.Wv = np.random.randn(self.d_model, self.d_model) * self.sigma
        patch_pixels = self.patch_size ** 2
        self.Wout = np.random.randn(self.d_model, patch_pixels) * self.sigma
        self.bout = np.random.randn(patch_pixels) * self.sigma
        self._param_count = self.patch_embeds.size + self.pos_embeds.size + self.Wq.size + self.Wk.size + self.Wv.size + self.Wout.size + self.bout.size

    def generate(self) -> np.ndarray:
        x = self.patch_embeds + self.pos_embeds
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        d_k = self.d_model ** 0.5
        scores = (Q @ K.T) / d_k
        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
        x = attn @ V
        patches = 1 / (1 + np.exp(-np.clip(x @ self.Wout + self.bout, -10, 10)))
        n_patches_side = int(np.sqrt(self.n_patches))
        img = np.zeros((self.size, self.size))
        for i in range(n_patches_side):
            for j in range(n_patches_side):
                patch_idx = i * n_patches_side + j
                patch = patches[patch_idx].reshape(self.patch_size, self.patch_size)
                img[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patch
        return img.astype(np.float32)


class FourierBasis(BaseArchitecture):
    """Random Fourier basis functions (weighted sums of 2D sinusoids)."""

    def __init__(self, sigma: float = 1.0, size: int = 64, n_basis: int = 16):
        super().__init__(sigma, size)
        self.n_basis = n_basis

    def sample_random_init(self) -> None:
        self.freq_x = np.random.randn(self.n_basis) * 2.0
        self.freq_y = np.random.randn(self.n_basis) * 2.0
        self.phase = np.random.uniform(0, 2 * np.pi, self.n_basis)
        self.amplitude = np.random.randn(self.n_basis) * self.sigma
        self._param_count = 4 * self.n_basis

    def generate(self) -> np.ndarray:
        coords = np.linspace(-np.pi, np.pi, self.size)
        x, y = np.meshgrid(coords, coords)
        img = np.zeros((self.size, self.size))
        for i in range(self.n_basis):
            img += self.amplitude[i] * np.sin(self.freq_x[i] * x + self.freq_y[i] * y + self.phase[i])
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)


class WalkCarver(BaseArchitecture):
    """Random walk maze carver."""

    def __init__(self, sigma: float = 1.0, size: int = 64, n_walks: int = 20):
        super().__init__(sigma, size)
        self.n_walks = n_walks

    def sample_random_init(self) -> None:
        self.start_points = np.random.randint(0, self.size, (self.n_walks, 2))
        self.walk_lengths = np.random.randint(10, self.size * 2, self.n_walks)
        self.directions = np.random.choice(4, size=sum(self.walk_lengths))
        self._param_count = self.start_points.size + self.walk_lengths.size + len(self.directions)

    def generate(self) -> np.ndarray:
        img = np.ones((self.size, self.size), dtype=np.float32)
        dir_idx = 0
        for walk_idx in range(self.n_walks):
            x, y = self.start_points[walk_idx]
            for _ in range(self.walk_lengths[walk_idx]):
                img[int(y) % self.size, int(x) % self.size] = 0
                direction = self.directions[dir_idx]
                dir_idx += 1
                if direction == 0: x += 1
                elif direction == 1: x -= 1
                elif direction == 2: y += 1
                else: y -= 1
        return img


class SpanningTreeMaze(BaseArchitecture):
    """MST-based maze generator."""

    def __init__(self, sigma: float = 1.0, size: int = 64, cell_size: int = 4):
        super().__init__(sigma, size)
        self.cell_size = cell_size
        self.grid_size = size // cell_size

    def sample_random_init(self) -> None:
        n_h_edges = self.grid_size * (self.grid_size - 1)
        n_v_edges = (self.grid_size - 1) * self.grid_size
        self.h_weights = np.random.rand(n_h_edges)
        self.v_weights = np.random.rand(n_v_edges)
        self._param_count = n_h_edges + n_v_edges

    def _find(self, parent: np.ndarray, i: int) -> int:
        if parent[i] != i:
            parent[i] = self._find(parent, parent[i])
        return parent[i]

    def _union(self, parent: np.ndarray, rank: np.ndarray, x: int, y: int):
        xroot, yroot = self._find(parent, x), self._find(parent, y)
        if rank[xroot] < rank[yroot]: parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]: parent[yroot] = xroot
        else: parent[yroot] = xroot; rank[xroot] += 1

    def generate(self) -> np.ndarray:
        edges = []
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                edges.append((self.h_weights[idx], i * self.grid_size + j, i * self.grid_size + j + 1, 'h', i, j))
                idx += 1
        idx = 0
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                edges.append((self.v_weights[idx], i * self.grid_size + j, (i + 1) * self.grid_size + j, 'v', i, j))
                idx += 1
        edges.sort(key=lambda x: x[0])

        img = np.ones((self.size, self.size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start, x_start = i * self.cell_size + 1, j * self.cell_size + 1
                img[y_start:y_start + self.cell_size - 2, x_start:x_start + self.cell_size - 2] = 0

        n_cells = self.grid_size ** 2
        parent, rank = np.arange(n_cells), np.zeros(n_cells, dtype=int)
        for weight, cell1, cell2, direction, i, j in edges:
            if self._find(parent, cell1) != self._find(parent, cell2):
                self._union(parent, rank, cell1, cell2)
                if direction == 'h':
                    y_start, x_pos = i * self.cell_size + 1, (j + 1) * self.cell_size
                    img[y_start:y_start + self.cell_size - 2, x_pos - 1:x_pos + 1] = 0
                else:
                    y_pos, x_start = (i + 1) * self.cell_size, j * self.cell_size + 1
                    img[y_pos - 1:y_pos + 1, x_start:x_start + self.cell_size - 2] = 0
        return img


class LSTMDecoder(BaseArchitecture):
    """LSTM-based sequential pixel generation."""

    def __init__(self, sigma: float = 1.0, size: int = 64, hidden_dim: int = 32):
        super().__init__(sigma, size)
        self.hidden_dim = hidden_dim

    def sample_random_init(self) -> None:
        input_dim = self.size + self.hidden_dim
        self.Wi = np.random.randn(input_dim, self.hidden_dim) * self.sigma
        self.Wf = np.random.randn(input_dim, self.hidden_dim) * self.sigma
        self.Wc = np.random.randn(input_dim, self.hidden_dim) * self.sigma
        self.Wo = np.random.randn(input_dim, self.hidden_dim) * self.sigma
        self.bi = np.random.randn(self.hidden_dim) * self.sigma
        self.bf = np.random.randn(self.hidden_dim) * self.sigma
        self.bc = np.random.randn(self.hidden_dim) * self.sigma
        self.bo = np.random.randn(self.hidden_dim) * self.sigma
        self.Wout = np.random.randn(self.hidden_dim, self.size) * self.sigma
        self.bout = np.random.randn(self.size) * self.sigma
        self._param_count = 4 * (self.Wi.size + self.bi.size) + self.Wout.size + self.bout.size

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def generate(self) -> np.ndarray:
        h, c = np.zeros(self.hidden_dim), np.zeros(self.hidden_dim)
        prev_row = np.zeros(self.size)
        img = np.zeros((self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            x = np.concatenate([prev_row, h])
            i_gate = self._sigmoid(x @ self.Wi + self.bi)
            f_gate = self._sigmoid(x @ self.Wf + self.bf)
            c_tilde = np.tanh(x @ self.Wc + self.bc)
            o_gate = self._sigmoid(x @ self.Wo + self.bo)
            c = f_gate * c + i_gate * c_tilde
            h = o_gate * np.tanh(c)
            row = self._sigmoid(h @ self.Wout + self.bout)
            img[i] = row
            prev_row = row
        return img


class MixtureOfExperts(BaseArchitecture):
    """Sparse mixture of experts with top-2 gating."""

    def __init__(self, sigma: float = 1.0, size: int = 64, n_experts: int = 4):
        super().__init__(sigma, size)
        self.n_experts = n_experts

    def sample_random_init(self) -> None:
        hidden = 64
        self.Wgate = np.random.randn(2, self.n_experts) * self.sigma
        self.bgate = np.random.randn(self.n_experts) * self.sigma
        self.W1_experts = np.random.randn(self.n_experts, 2, hidden) * self.sigma
        self.b1_experts = np.random.randn(self.n_experts, hidden) * self.sigma
        self.W2_experts = np.random.randn(self.n_experts, hidden, 1) * self.sigma
        self.b2_experts = np.random.randn(self.n_experts, 1) * self.sigma
        self._param_count = self.Wgate.size + self.bgate.size + self.W1_experts.size + self.b1_experts.size + self.W2_experts.size + self.b2_experts.size

    def generate(self) -> np.ndarray:
        coords = np.linspace(-1, 1, self.size)
        x, y = np.meshgrid(coords, coords)
        points = np.stack([x.flatten(), y.flatten()], axis=1)
        gate_logits = points @ self.Wgate + self.bgate
        top2_idx = np.argsort(gate_logits, axis=1)[:, -2:]
        gate_values = np.zeros_like(gate_logits)
        for i in range(len(points)):
            mask = np.zeros(self.n_experts)
            mask[top2_idx[i]] = 1
            gate_values[i] = np.exp(gate_logits[i]) * mask
        gate_weights = gate_values / (gate_values.sum(axis=1, keepdims=True) + 1e-8)

        output = np.zeros(len(points))
        for k in range(self.n_experts):
            h = np.maximum(0, points @ self.W1_experts[k] + self.b1_experts[k])
            expert_out = 1 / (1 + np.exp(-np.clip(h @ self.W2_experts[k] + self.b2_experts[k], -10, 10)))
            output += gate_weights[:, k] * expert_out.flatten()
        return output.reshape(self.size, self.size).astype(np.float32)


# Architecture registry
ARCHITECTURE_CLASSES = {
    'CPPN': CPPN, 'CoordMLP': CoordMLP, 'FourierMLP': FourierMLP,
    'MLPDecoder': MLPDecoder, 'ConvDecoder': ConvDecoder, 'ViTDecoder': ViTDecoder,
    'FourierBasis': FourierBasis, 'WalkCarver': WalkCarver, 'SpanningTreeMaze': SpanningTreeMaze,
    'LSTMDecoder': LSTMDecoder, 'MixtureOfExperts': MixtureOfExperts,
}


def get_architecture(name: str, size: int = 64) -> BaseArchitecture:
    """Create an architecture instance by name."""
    return ARCHITECTURE_CLASSES[name](sigma=1.0, size=size)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

@dataclass
class ArchitectureVolumeResult:
    """Results for one architecture."""
    name: str
    n_samples: int
    order_mean: float
    order_std: float
    order_min: float
    order_max: float
    order_median: float
    order_q25: float
    order_q75: float
    volume_by_threshold: Dict[str, float]
    runtime_seconds: float


def sample_architecture_orders(arch_name: str, n_samples: int, image_size: int, seed: int) -> Tuple[np.ndarray, float]:
    """Sample order values from random parameter configurations."""
    np.random.seed(seed)
    orders = np.zeros(n_samples)
    start_time = time.time()

    for i in range(n_samples):
        arch = get_architecture(arch_name, size=image_size)
        img = arch.sample()
        binary = (img > 0.5).astype(np.uint8)
        orders[i] = compute_order(binary)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"    {arch_name}: {i+1}/{n_samples} samples ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    return orders, time.time() - start_time


def main():
    """Main execution."""
    print("=" * 70)
    print("RES-292: Thermodynamic Volume Validation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples per architecture: {N_SAMPLES:,}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Order thresholds: {ORDER_THRESHOLDS}")
    print(f"  Architectures: {len(ARCHITECTURES)}")

    # Output paths
    output_dir = PROJECT_ROOT / 'results' / 'thermodynamic_volume_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"\nEstimated runtime: ~{N_SAMPLES * len(ARCHITECTURES) / 100:.0f} minutes")

    # Run analysis
    architecture_results = {}
    total_start = time.time()

    for arch_name in ARCHITECTURES:
        print(f"\n  Processing {arch_name}...")
        arch_seed = RANDOM_SEED + hash(arch_name) % 10000

        orders, runtime = sample_architecture_orders(arch_name, N_SAMPLES, IMAGE_SIZE, arch_seed)

        volumes = {f"threshold_{t}": float(np.mean(orders >= t)) for t in ORDER_THRESHOLDS}

        result = ArchitectureVolumeResult(
            name=arch_name,
            n_samples=N_SAMPLES,
            order_mean=float(np.mean(orders)),
            order_std=float(np.std(orders)),
            order_min=float(np.min(orders)),
            order_max=float(np.max(orders)),
            order_median=float(np.median(orders)),
            order_q25=float(np.percentile(orders, 25)),
            order_q75=float(np.percentile(orders, 75)),
            volume_by_threshold=volumes,
            runtime_seconds=runtime,
        )

        architecture_results[arch_name] = asdict(result)

        # Save individual order samples
        np.save(output_dir / f'{arch_name}_orders.npy', orders)

        print(f"    Order: {result.order_mean:.4f} ± {result.order_std:.4f}")
        print(f"    Volume (Order>0.1): {volumes['threshold_0.1']:.1%}")
        print(f"    Runtime: {runtime:.1f}s")

    total_runtime = time.time() - total_start

    # Compute summary statistics
    mean_volumes, std_volumes = {}, {}
    for t in ORDER_THRESHOLDS:
        key = f'threshold_{t}'
        values = [architecture_results[name]['volume_by_threshold'][key] for name in ARCHITECTURES]
        mean_volumes[key] = float(np.mean(values))
        std_volumes[key] = float(np.std(values))

    # Rankings
    rankings = {}
    for t in ORDER_THRESHOLDS:
        key = f'threshold_{t}'
        sorted_archs = sorted(ARCHITECTURES, key=lambda x: architecture_results[x]['volume_by_threshold'][key], reverse=True)
        rankings[key] = sorted_archs

    # Full results
    results = {
        'experiment_id': 'RES-292',
        'hypothesis': 'Thermodynamic volume varies by 5+ orders of magnitude across architectures',
        'n_architectures': len(ARCHITECTURES),
        'n_samples_per_arch': N_SAMPLES,
        'image_size': IMAGE_SIZE,
        'thresholds': ORDER_THRESHOLDS,
        'seed': RANDOM_SEED,
        'architecture_results': architecture_results,
        'mean_volume_by_threshold': mean_volumes,
        'std_volume_by_threshold': std_volumes,
        'volume_rankings': rankings,
        'total_runtime_seconds': total_runtime,
    }

    # Save results
    results_file = output_dir / 'res_292_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Volume at Order > 0.1")
    print("=" * 70)
    print(f"{'Architecture':<20} {'Volume':>10} {'Mean Order':>12}")
    print("-" * 42)
    for name in rankings['threshold_0.1']:
        vol = architecture_results[name]['volume_by_threshold']['threshold_0.1']
        mean_ord = architecture_results[name]['order_mean']
        print(f"{name:<20} {vol:>9.1%} {mean_ord:>12.4f}")

    print(f"\nTotal runtime: {total_runtime / 60:.1f} minutes")

    # Validate hypothesis
    volumes_01 = [architecture_results[name]['volume_by_threshold']['threshold_0.1'] for name in ARCHITECTURES]
    max_vol, min_vol = max(volumes_01), min([v for v in volumes_01 if v > 0] or [1e-10])

    if max_vol > 0 and min_vol > 0:
        ratio = max_vol / min_vol
        print(f"\nVolume ratio (max/min non-zero): {ratio:.1e}")
        if ratio > 1e5:
            print("✓ VALIDATED: Volume varies by >5 orders of magnitude")
        else:
            print(f"✗ NOT VALIDATED: Volume ratio = {ratio:.1e} (<10^5)")
    else:
        print("✓ VALIDATED: Some architectures have exactly 0 volume")

    return results


if __name__ == '__main__':
    main()
