#!/usr/bin/env python3
"""
GAN Training Validation: Fair Test of Thermodynamic Hypothesis

The thermodynamic metric measures: "How easily does random noise → structured image?"
GANs directly test this - the generator must produce realistic images from pure noise.

This is a fairer test than autoencoder reconstruction, where the encoder provides
structured input that bypasses the random→structured challenge.

Hypothesis: CPPN generators should produce coherent shapes faster than MLP generators,
especially in low-data regimes where inductive bias matters most.

Usage:
    uv run python train_validation_gan.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class GANResult:
    """Results from GAN training."""
    name: str
    bits_score: float
    g_losses: list[float]
    d_losses: list[float]
    fid_proxy: float  # Simple quality metric
    training_time: float


class SmoothActivation(nn.Module):
    """CPPN-style smooth activation."""
    def forward(self, x):
        return 0.5 * torch.sin(x) + 0.5 * torch.tanh(x)


class CPPNGenerator(nn.Module):
    """
    CPPN-style generator: coordinate-based structure.

    Maps (noise, x, y) → pixel. The coordinate conditioning is the key inductive bias,
    not the exotic activations. Uses LeakyReLU for stable GAN training.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Map noise to per-pixel features
        self.noise_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # CPPN: (features + coords) → pixel
        # Key insight: coordinate conditioning provides spatial coherence
        self.cppn = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for x, y, r, r²
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Coordinate grid
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten(), (r**2).flatten()
        ], dim=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        n_pixels = self.image_size ** 2

        # Get noise features
        features = self.noise_net(z)  # (B, hidden)

        # Broadcast to all pixels
        features = features.unsqueeze(1).expand(-1, n_pixels, -1)
        coords = self.coords.unsqueeze(0).expand(batch_size, -1, -1)

        # CPPN forward
        x = torch.cat([features, coords], dim=-1)
        out = self.cppn(x)

        return out.view(batch_size, 1, self.image_size, self.image_size)


class MLPGenerator(nn.Module):
    """
    Standard MLP generator: no coordinate structure.

    Direct noise → image mapping. Should produce less coherent early outputs.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_pixels),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, self.image_size, self.image_size)


class ConvGenerator(nn.Module):
    """
    Convolutional generator: standard DCGAN-style.

    Upsampling from noise. Has spatial structure but different from CPPN.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, image_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Project and reshape
        self.project = nn.Linear(latent_dim, hidden_dim * 7 * 7)

        # Upsample: 7 → 14 → 28
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),  # 7 → 14
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, 1, 4, 2, 1),  # 14 → 28
            nn.Sigmoid()
        )

        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(-1, self.hidden_dim, 7, 7)
        return self.conv(x)


class Discriminator(nn.Module):
    """Simple discriminator for all generators."""

    def __init__(self, image_size: int = 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_quality_score(generator, device, n_samples: int = 100) -> float:
    """
    Simple quality proxy: measure coherence of generated images.

    Uses spatial smoothness as proxy for "structured" output.
    Real FID would require Inception - this is a lightweight alternative.
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, generator.latent_dim, device=device)
        images = generator(z)

        # Spatial smoothness: low variation between adjacent pixels = coherent
        dx = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
        dy = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
        smoothness = 1.0 / (1.0 + (dx + dy).item())

        # Variance: should have some structure, not uniform
        variance = images.var().item()

        # Combined score: smooth but not uniform
        score = smoothness * min(1.0, variance * 10)

    generator.train()
    return score


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    device: torch.device,
    n_epochs: int = 5,
    lr: float = 2e-4,
    checkpoint_every: int = 100
) -> GANResult:
    """Train GAN and track metrics."""

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses = []
    d_losses = []
    step = 0
    start_time = time.time()

    for epoch in range(n_epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            opt_d.zero_grad()

            # Real
            d_real = discriminator(real_images)
            loss_real = criterion(d_real, real_labels)

            # Fake
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            loss_fake = criterion(d_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            d_fake = discriminator(fake_images)
            loss_g = criterion(d_fake, real_labels)  # Want discriminator to think fake is real
            loss_g.backward()
            opt_g.step()

            if step % checkpoint_every == 0:
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())

            step += 1

    training_time = time.time() - start_time
    quality = compute_quality_score(generator, device)

    return GANResult(
        name=generator.__class__.__name__,
        bits_score=0,
        g_losses=g_losses,
        d_losses=d_losses,
        fid_proxy=quality,
        training_time=training_time
    )


def save_samples(generator, device, output_path: str, n_samples: int = 16):
    """Save generated samples."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, generator.latent_dim, device=device)
        samples = generator(z)
        save_image(samples, output_path, nrow=4, normalize=True)
    generator.train()


def main():
    print("=" * 70)
    print("GAN TRAINING VALIDATION: Fair Test of Thermodynamic Hypothesis")
    print("=" * 70)
    print()
    print("Test: Can random noise → structured images?")
    print("CPPN's smooth, coordinate-based bias should help here.")
    print()

    device = get_device()
    print(f"Using device: {device}")

    # Thermodynamic scores (bits to threshold)
    thermo_scores = {
        'CPPNGenerator': 2.0,    # Smooth, coordinate-based (tournament winner)
        'ConvGenerator': 8.0,    # Spatial but different bias
        'MLPGenerator': 15.0,    # No spatial structure
    }

    # Load MNIST (subset for faster training)
    print("\nLoading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Use subset for low-data regime (where inductive bias matters most)
    n_train = 5000  # Low-data regime
    indices = np.random.choice(len(full_dataset), n_train, replace=False)
    subset = torch.utils.data.Subset(full_dataset, indices)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
    print(f"Using {n_train} training images (low-data regime)")

    # Create output directory
    os.makedirs('gan_samples', exist_ok=True)

    # Generators to test
    latent_dim = 64
    image_size = 28

    generators = [
        ('CPPNGenerator', CPPNGenerator(latent_dim=latent_dim, hidden_dim=128, image_size=image_size)),
        ('ConvGenerator', ConvGenerator(latent_dim=latent_dim, hidden_dim=128, image_size=image_size)),
        ('MLPGenerator', MLPGenerator(latent_dim=latent_dim, hidden_dim=256, image_size=image_size)),
    ]

    results = []

    for name, generator in generators:
        print(f"\nTraining {name}...")
        discriminator = Discriminator(image_size=image_size)

        result = train_gan(
            generator, discriminator, train_loader, device,
            n_epochs=10,  # More epochs for GAN training
            lr=2e-4,
            checkpoint_every=50
        )
        result.bits_score = thermo_scores[name]
        results.append(result)

        # Save samples
        save_samples(generator, device, f'gan_samples/{name}_samples.png')

        print(f"  Final G loss: {result.g_losses[-1]:.4f}")
        print(f"  Quality score: {result.fid_proxy:.4f}")
        print(f"  Training time: {result.training_time:.1f}s")
        print(f"  Samples saved to: gan_samples/{name}_samples.png")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Thermodynamic Score vs GAN Quality")
    print("=" * 70)
    print()
    print(f"{'Generator':<20} {'Bits':<10} {'Quality':<12} {'G Loss':<12} {'Prediction'}")
    print("-" * 70)

    # Sort by bits (thermodynamic prediction)
    results.sort(key=lambda r: r.bits_score)

    for i, r in enumerate(results):
        rank = i + 1
        predicted = "BEST" if rank == 1 else f"#{rank}"
        print(f"{r.name:<20} {r.bits_score:<10.1f} {r.fid_proxy:<12.4f} {r.g_losses[-1]:<12.4f} {predicted}")

    # Correlation
    print("\n" + "=" * 70)
    print("CORRELATION CHECK")
    print("=" * 70)

    bits = [r.bits_score for r in results]
    quality = [r.fid_proxy for r in results]

    # For quality, HIGHER is better (correlation should be negative with bits)
    from scipy.stats import spearmanr
    corr, pval = spearmanr(bits, quality)

    print(f"\nBits ranking (predicted best → worst): {[r.name for r in results]}")
    quality_sorted = sorted(results, key=lambda r: -r.fid_proxy)  # Higher quality = better
    print(f"Quality ranking (actual best → worst): {[r.name for r in quality_sorted]}")
    print(f"\nSpearman correlation (bits vs quality): {corr:.3f} (p={pval:.3f})")
    print("(Negative correlation = lower bits → higher quality = hypothesis supported)")

    if corr < -0.5:
        print("\n*** NEGATIVE CORRELATION: Lower bits → Higher quality ***")
        print("*** The thermodynamic metric PREDICTS generative quality! ***")
    elif corr > 0.5:
        print("\nPositive correlation: Hypothesis not supported for this task.")
    else:
        print("\nWeak correlation: Need more data or different setup.")

    print("\n" + "=" * 70)
    print("VISUAL CHECK")
    print("=" * 70)
    print("\nCheck generated samples in gan_samples/ directory:")
    print("- CPPNGenerator_samples.png - Should show smooth, coherent shapes")
    print("- ConvGenerator_samples.png - Should show spatial structure")
    print("- MLPGenerator_samples.png - May show less coherent patterns")
    print("\nThe CPPN should produce recognizable digit-like shapes faster.")


if __name__ == "__main__":
    main()
