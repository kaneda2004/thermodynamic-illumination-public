#!/usr/bin/env python3
"""
Training Validation with PyTorch: The "Money Plot"

This script validates the core hypothesis:
    "Architectures with lower bits-to-threshold learn faster"

We train decoder networks with different inductive biases on MNIST and measure:
1. Thermodynamic score (bits from the tournament)
2. Training speed (loss/accuracy after N steps)

Uses PyTorch with MPS (Apple Silicon) or CUDA support.

Usage:
    uv run python train_validation_torch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional
from torchvision import datasets, transforms


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingResult:
    """Results from training a decoder."""
    name: str
    bits_score: float
    losses: list[float]
    final_loss: float
    training_time: float


class CPPNDecoder(nn.Module):
    """
    CPPN-style decoder: coordinate-based with smooth activations.

    Maps (latent, x, y) -> pixel value using smooth periodic functions.
    This mirrors the inductive bias of CPPNs: globally coherent patterns.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Network: (latent + x + y + r) -> hidden -> pixel
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 4, hidden_dim),  # +4 for x, y, r, r^2
            SmoothActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            SmoothActivation(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Pre-compute coordinate grid
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten(), (r**2).flatten()
        ], dim=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate image from latent z. z: (batch, latent_dim)"""
        batch_size = z.shape[0]
        n_pixels = self.image_size ** 2

        # Broadcast latent to all pixels
        z_expanded = z.unsqueeze(1).expand(-1, n_pixels, -1)  # (B, H*W, latent)
        coords_expanded = self.coords.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, 4)

        x = torch.cat([z_expanded, coords_expanded], dim=-1)  # (B, H*W, latent+4)
        out = self.net(x)  # (B, H*W, 1)

        return out.view(batch_size, 1, self.image_size, self.image_size)


class SmoothActivation(nn.Module):
    """Mix of smooth activations like CPPNs use."""
    def forward(self, x):
        # Combination of sin and tanh for smooth, periodic patterns
        return 0.5 * torch.sin(x) + 0.5 * torch.tanh(x)


class MLPDecoder(nn.Module):
    """
    Standard MLP decoder: no coordinate info, purely learned mapping.

    This is the "local" baseline - no explicit spatial structure.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_pixels),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, self.image_size, self.image_size)


class FourierDecoder(nn.Module):
    """
    Fourier-style decoder: frequency basis functions.

    Maps latent to frequency coefficients, then reconstructs via basis.
    """

    def __init__(self, latent_dim: int = 16, n_frequencies: int = 8, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_frequencies = n_frequencies
        self.image_size = image_size

        # Map latent to frequency coefficients
        self.coeff_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_frequencies * n_frequencies * 2)  # Real + imag for each freq
        )

        # Precompute frequency basis
        freqs_x = torch.fft.fftfreq(image_size)[:n_frequencies]
        freqs_y = torch.fft.fftfreq(image_size)[:n_frequencies]
        fx, fy = torch.meshgrid(freqs_x, freqs_y, indexing='ij')
        self.register_buffer('freq_x', fx)
        self.register_buffer('freq_y', fy)

        # Spatial coordinates
        x = torch.arange(image_size).float()
        y = torch.arange(image_size).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]

        # Get coefficients
        coeffs = self.coeff_net(z)  # (B, n_freq^2 * 2)
        coeffs = coeffs.view(batch_size, self.n_frequencies, self.n_frequencies, 2)
        coeffs_complex = torch.complex(coeffs[..., 0], coeffs[..., 1])

        # Reconstruct via inverse DFT (simplified)
        # Sum of 2D sinusoids
        out = torch.zeros(batch_size, self.image_size, self.image_size, device=z.device)
        for i in range(self.n_frequencies):
            for j in range(self.n_frequencies):
                phase = 2 * np.pi * (self.freq_x[i, j] * self.xx + self.freq_y[i, j] * self.yy)
                basis = torch.cos(phase)
                out += coeffs_complex[:, i, j].real.unsqueeze(-1).unsqueeze(-1) * basis
                out += coeffs_complex[:, i, j].imag.unsqueeze(-1).unsqueeze(-1) * torch.sin(phase)

        out = torch.sigmoid(out / self.n_frequencies)
        return out.unsqueeze(1)


class SimpleAutoencoder(nn.Module):
    """Wraps a decoder with a simple encoder for reconstruction."""

    def __init__(self, decoder: nn.Module, latent_dim: int = 16, image_size: int = 28):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    model: nn.Module,
    train_loader,
    device: torch.device,
    n_epochs: int = 3,
    lr: float = 1e-3,
    checkpoint_every: int = 100
) -> TrainingResult:
    """Train autoencoder and track loss."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    step = 0
    start_time = time.time()

    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()

            if step % checkpoint_every == 0:
                losses.append(loss.item())
            step += 1

    training_time = time.time() - start_time

    return TrainingResult(
        name=model.decoder.__class__.__name__,
        bits_score=0,  # Filled in later
        losses=losses,
        final_loss=losses[-1] if losses else float('inf'),
        training_time=training_time
    )


def main():
    print("=" * 70)
    print("TRAINING VALIDATION: The Money Plot (PyTorch)")
    print("Testing if thermodynamic scores predict learning speed")
    print("=" * 70)
    print()

    device = get_device()
    print(f"Using device: {device}")
    print()

    # Thermodynamic scores from tournament (bits to threshold)
    # Lower = better (easier to generate MNIST-like images)
    thermo_scores = {
        'CPPNDecoder': 2.0,      # CPPN wins tournament (smooth, coordinate-based)
        'MLPDecoder': 10.0,      # MLP baseline (no spatial bias)
        'FourierDecoder': 20.0,  # Fourier (wrong inductive bias for MNIST)
    }

    # Load MNIST
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(f"Loaded {len(train_dataset)} training images")
    print()

    # Create decoders
    latent_dim = 16
    image_size = 28

    decoders = [
        ('CPPNDecoder', CPPNDecoder(latent_dim=latent_dim, hidden_dim=64, image_size=image_size)),
        ('MLPDecoder', MLPDecoder(latent_dim=latent_dim, hidden_dim=256, image_size=image_size)),
        ('FourierDecoder', FourierDecoder(latent_dim=latent_dim, n_frequencies=8, image_size=image_size)),
    ]

    results = []

    for name, decoder in decoders:
        print(f"Training {name}...")
        model = SimpleAutoencoder(decoder, latent_dim=latent_dim, image_size=image_size)

        result = train_autoencoder(
            model, train_loader, device,
            n_epochs=2,  # Quick training
            lr=1e-3,
            checkpoint_every=50
        )
        result.bits_score = thermo_scores[name]
        results.append(result)

        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Training time: {result.training_time:.1f}s")
        print()

    # Results table
    print("=" * 70)
    print("RESULTS: Thermodynamic Score vs Training Loss")
    print("=" * 70)
    print()
    print(f"{'Decoder':<20} {'Bits (lower=better)':<20} {'Final Loss':<15} {'Prediction'}")
    print("-" * 70)

    # Sort by bits (thermodynamic prediction)
    results.sort(key=lambda r: r.bits_score)

    for i, r in enumerate(results):
        rank = i + 1
        predicted = "BEST" if rank == 1 else f"#{rank}"
        print(f"{r.name:<20} {r.bits_score:<20.1f} {r.final_loss:<15.4f} {predicted}")

    # Correlation check
    print()
    print("=" * 70)
    print("CORRELATION CHECK")
    print("=" * 70)

    bits = [r.bits_score for r in results]
    losses = [r.final_loss for r in results]

    # Sort by bits and check if losses are also sorted
    bits_order = np.argsort(bits)
    loss_order = np.argsort(losses)

    print(f"Bits ranking (predicted):  {[results[i].name for i in bits_order]}")
    print(f"Loss ranking (actual):     {[results[i].name for i in loss_order]}")

    # Spearman correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(bits, losses)
    print()
    print(f"Spearman correlation: {corr:.3f} (p-value: {pval:.3f})")

    if corr > 0.5:
        print()
        print("POSITIVE CORRELATION: Lower bits -> Lower loss")
        print("The thermodynamic metric PREDICTS learning speed!")
    elif corr < -0.5:
        print()
        print("NEGATIVE CORRELATION: Lower bits -> Higher loss")
        print("The hypothesis appears WRONG for this setup.")
    else:
        print()
        print("WEAK CORRELATION: Need more architectures or longer training.")

    # Learning curves
    print()
    print("=" * 70)
    print("LEARNING CURVES (loss over time)")
    print("=" * 70)

    max_len = max(len(r.losses) for r in results)
    for step in range(0, max_len, max(1, max_len // 10)):
        row = f"Step {step:4d}: "
        for r in results:
            if step < len(r.losses):
                row += f"{r.name[:8]:>10}={r.losses[step]:.4f}  "
        print(row)


if __name__ == "__main__":
    main()
