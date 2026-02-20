#!/usr/bin/env python3
"""
Reconstruction Validation: Testing the Generative-Discriminative Trade-off

This tests whether thermodynamic bits predict RECONSTRUCTION quality
(as opposed to classification accuracy).

Hypothesis: Low-bits architectures (high structural alignment) should be
BETTER at reconstruction because they preserve spatial topology.

This validates the key insight:
- Low bits (CPPN) → Good for reconstruction (preserves topology)
- High bits (MLP) → Good for classification (high-dimensional projection)

Usage:
    uv run python reconstruction_validation.py --save_json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass
from torchvision import datasets, transforms
import argparse
import json
from pathlib import Path
import sys

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Feature Extractors (same as linear probe validation)
# =============================================================================

class RandomCPPNFeatures(nn.Module):
    """CPPN-style feature extractor - LOW BITS (high structure)."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        for param in self.parameters():
            param.requires_grad = False

        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten()
        ], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pixels = x.view(batch_size, -1)
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_values = pixels.unsqueeze(-1)
        cppn_input = torch.cat([coords_batch, pixel_values], dim=-1)
        return self.net(cppn_input)


class DeepCPPNFeatures(nn.Module):
    """Deeper CPPN - should have even lower bits."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        for param in self.parameters():
            param.requires_grad = False

        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten()
        ], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pixels = x.view(batch_size, -1)
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_values = pixels.unsqueeze(-1)
        cppn_input = torch.cat([coords_batch, pixel_values], dim=-1)
        return self.net(cppn_input)


class RandomMLPFeatures(nn.Module):
    """Random MLP - HIGH BITS (low structure, high entropy)."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        self.net = nn.Sequential(
            nn.Linear(n_pixels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * n_pixels),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2
        flat = x.view(batch_size, -1)
        out = self.net(flat)
        return torch.tanh(out.view(batch_size, n_pixels, self.feature_dim))


class RandomConvFeatures(nn.Module):
    """Random ConvNet - medium structure."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 1),
            nn.Tanh()
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        conv_out = self.conv(x)
        return conv_out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)


class RandomResNetFeatures(nn.Module):
    """ResNet-style with skip connections."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.block1_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.block2_conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block2_skip = nn.Conv2d(32, 64, 1)
        self.final = nn.Conv2d(64, feature_dim, 1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = torch.relu(self.conv1(x))
        identity = h
        h = torch.relu(self.block1_conv1(h))
        h = self.block1_conv2(h)
        h = torch.relu(h + identity)
        identity = self.block2_skip(h)
        h = torch.relu(self.block2_conv1(h))
        h = self.block2_conv2(h)
        h = torch.relu(h + identity)
        h = torch.tanh(self.final(h))
        return h.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)


class RandomFourierFeatures(nn.Module):
    """Random Fourier features - HIGH BITS."""

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        self.register_buffer('frequencies', torch.randn(n_pixels, feature_dim // 2) * 3)
        self.register_buffer('phases', torch.rand(feature_dim // 2) * 2 * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2
        flat = x.view(batch_size, -1)
        proj = flat @ self.frequencies
        features_global = torch.cat([
            torch.sin(proj + self.phases),
            torch.cos(proj + self.phases)
        ], dim=-1)
        return features_global.unsqueeze(1).expand(-1, n_pixels, -1)


# =============================================================================
# Reconstruction Model
# =============================================================================

class LinearReconstructor(nn.Module):
    """
    Linear reconstruction from frozen features.
    Features (B, H*W, D) -> Pool -> Linear -> Reconstruct (B, 1, H, W)
    """

    def __init__(self, feature_extractor: nn.Module, image_size: int = 28):
        super().__init__()
        self.features = feature_extractor
        self.image_size = image_size
        n_pixels = image_size ** 2

        # Linear decoder: per-pixel features -> pixel values
        # Option 1: Global pooling + expand
        self.decoder = nn.Linear(feature_extractor.feature_dim, n_pixels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get frozen features: (B, H*W, D)
        features = self.features(x)

        # Global average pooling: (B, D)
        pooled = features.mean(dim=1)

        # Decode to image: (B, H*W)
        decoded = torch.sigmoid(self.decoder(pooled))

        # Reshape: (B, 1, H, W)
        return decoded.view(-1, 1, self.image_size, self.image_size)


class PerPixelReconstructor(nn.Module):
    """
    Per-pixel linear reconstruction from frozen features.
    This is a stronger test: can the per-pixel features reconstruct the image?
    """

    def __init__(self, feature_extractor: nn.Module, image_size: int = 28):
        super().__init__()
        self.features = feature_extractor
        self.image_size = image_size

        # Per-pixel decoder: feature_dim -> 1 pixel value
        self.decoder = nn.Linear(feature_extractor.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Get frozen features: (B, H*W, D)
        features = self.features(x)

        # Decode each pixel: (B, H*W, D) -> (B, H*W, 1) -> (B, H*W)
        decoded = torch.sigmoid(self.decoder(features)).squeeze(-1)

        # Reshape: (B, 1, H, W)
        return decoded.view(batch_size, 1, self.image_size, self.image_size)


# =============================================================================
# Complexity Measurement
# =============================================================================

def compute_feature_complexity(
    extractor: nn.Module,
    device: torch.device,
    n_samples: int = 200,
    image_size: int = 28
) -> float:
    """Compute effective dimensionality as complexity proxy."""
    extractor = extractor.to(device)
    extractor.eval()

    random_images = torch.rand(n_samples, 1, image_size, image_size, device=device)

    with torch.no_grad():
        features = extractor(random_images)
        pooled = features.mean(dim=1)
        pooled_np = pooled.cpu().numpy()

    centered = pooled_np - pooled_np.mean(axis=0)
    cov = np.cov(centered.T)

    if cov.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues[::-1]

    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return 1.0

    normalized = eigenvalues / total_var
    participation_ratio = 1.0 / (normalized ** 2).sum()

    return float(np.log2(max(participation_ratio, 1.0)))


# =============================================================================
# Training
# =============================================================================

@dataclass
class ReconstructionResult:
    name: str
    bits: float
    train_losses: list[float]
    test_losses: list[float]
    final_loss: float


def train_reconstructor(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    n_epochs: int = 20,
    lr: float = 1e-2,
) -> ReconstructionResult:
    """Train linear reconstructor and track MSE loss."""

    model = model.to(device)
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0
        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # Evaluation
        model.eval()
        test_loss = 0
        n_test = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                test_loss += loss.item()
                n_test += 1

        test_losses.append(test_loss / n_test)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_losses[-1]:.4f}, test_loss={test_losses[-1]:.4f}")

    return ReconstructionResult(
        name=model.features.__class__.__name__,
        bits=0,
        train_losses=train_losses,
        test_losses=test_losses,
        final_loss=test_losses[-1]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reconstruction Validation")
    parser.add_argument("--quick", action="store_true", help="Run quick mode (fewer epochs/samples)")
    parser.add_argument("--save_json", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("RECONSTRUCTION VALIDATION: Testing Generative-Discriminative Trade-off")
    print("=" * 70)
    
    device = get_device()
    print(f"Using device: {device}")

    # Load MNIST
    print("\nLoading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])

    # Use a simpler data loading strategy if possible to avoid huge downloads if verified
    # But for reproducibility we stick to standard MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    n_train = 1000 if args.quick else 10000
    n_test = 200 if args.quick else 2000
    
    print(f"Using {n_train} training, {n_test} test images")
    
    train_indices = np.random.choice(len(train_dataset), n_train, replace=False)
    test_indices = np.random.choice(len(test_dataset), n_test, replace=False)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    # Feature extractors
    image_size = 28
    feature_dim = 64

    extractors = [
        ('DeepCPPN', DeepCPPNFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomCPPN', RandomCPPNFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomResNet', RandomResNetFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomConv', RandomConvFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomFourier', RandomFourierFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomMLP', RandomMLPFeatures(feature_dim=feature_dim, image_size=image_size)),
    ]

    # Compute complexity scores
    print("\nComputing feature complexity scores...")
    complexity_scores = {}
    for name, extractor in extractors:
        score = compute_feature_complexity(extractor, device, n_samples=200, image_size=image_size)
        complexity_scores[name] = score
        print(f"  {name}: {score:.2f} bits")

    all_results_data = {}

    # Test both reconstruction methods
    epochs = 2 if args.quick else 20
    
    for recon_type, ReconClass in [
        ("Global Pooling", LinearReconstructor),
        ("Per-Pixel", PerPixelReconstructor)
    ]:
        print(f"\n{'='*70}")
        print(f"RECONSTRUCTION TEST: {recon_type}")
        print(f"{'='*70}")

        results = []

        for name, extractor in extractors:
            print(f"\n--- {name} ---")

            model = ReconClass(extractor, image_size=image_size)
            result = train_reconstructor(model, train_loader, test_loader, device, n_epochs=epochs)
            result.bits = complexity_scores[name]
            results.append(result)

        # Correlation analysis
        bits = [r.bits for r in results]
        losses = [r.final_loss for r in results]

        corr, pval = spearmanr(bits, losses)

        print(f"\nSpearman correlation (bits vs loss): {corr:.3f} (p={pval:.3f})")
        
        # Collect data for JSON
        all_results_data[recon_type] = {
            "spearman_r": float(corr),
            "p_value": float(pval),
            "results": [
                {
                    "name": r.name,
                    "bits": float(r.bits),
                    "mse": float(r.final_loss)
                } for r in results
            ]
        }

    # Save Results
    if args.save_json:
        output_dir = Path("results/reconstruction_quality")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "reconstruction_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(all_results_data, f, indent=2)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()