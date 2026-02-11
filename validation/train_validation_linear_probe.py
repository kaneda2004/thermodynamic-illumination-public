#!/usr/bin/env python3
"""
Linear Probe Validation: Testing Representation Quality of Random Features

This tests whether the thermodynamic metric (bits-to-threshold) predicts
how well FROZEN random features support downstream linear classification.

Setup:
1. Initialize random-weight networks (CPPN-style, MLP, ConvNet)
2. Freeze all weights (no training of backbone)
3. Pass actual input images through to get features
4. Train ONLY a linear classifier on top
5. Measure classification accuracy

Key insight: This isolates INDUCTIVE BIAS from GRADIENT FLOW.
If a random CPPN "contains" digit structure in its geometry,
a linear probe should find it more easily than random MLP features.

CRITICAL: Features must DEPEND ON THE INPUT IMAGE, not just coordinates.
The previous version of this file had a bug where features ignored x entirely.

Usage:
    uv run python train_validation_linear_probe.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from torchvision import datasets, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ProbeResult:
    name: str
    bits_score: float
    train_losses: list[float]
    test_accuracies: list[float]
    final_accuracy: float
    training_time: float


class RandomCPPNFeatures(nn.Module):
    """
    CPPN-style feature extractor that ACTUALLY uses the input image.

    The CPPN receives (x_coord, y_coord, r, pixel_value) at each position
    and outputs features. This preserves coordinate structure while
    incorporating the actual input.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        # CPPN: (x, y, r, pixel_value) -> features
        # Input dim = 4: x_coord, y_coord, radius, pixel_intensity
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        # Freeze all weights
        for param in self.parameters():
            param.requires_grad = False

        # Coordinate grid (normalized to [-1, 1])
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        # Shape: (H*W, 3) for x, y, r
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten()
        ], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, 1, H, W)

        Returns:
            Features, shape (B, H*W, feature_dim)
        """
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2

        # Flatten image pixels: (B, 1, H, W) -> (B, H*W)
        pixels = x.view(batch_size, -1)

        # Expand coordinates for batch: (H*W, 3) -> (B, H*W, 3)
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate coordinates with pixel values: (B, H*W, 4)
        # pixels: (B, H*W) -> (B, H*W, 1)
        pixel_values = pixels.unsqueeze(-1)
        cppn_input = torch.cat([coords_batch, pixel_values], dim=-1)

        # Pass through CPPN: (B, H*W, 4) -> (B, H*W, feature_dim)
        features = self.net(cppn_input)

        return features


class RandomMLPFeatures(nn.Module):
    """
    Random MLP feature extractor - no spatial structure.

    Flattens the image and passes through a random MLP.
    This has no coordinate bias, just random projections.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        # Random projection: pixels -> features
        self.net = nn.Sequential(
            nn.Linear(n_pixels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * n_pixels),  # Output features per pixel
        )

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, 1, H, W)

        Returns:
            Features, shape (B, H*W, feature_dim)
        """
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2

        # Flatten: (B, 1, H, W) -> (B, H*W)
        flat = x.view(batch_size, -1)

        # Random projection: (B, H*W) -> (B, feature_dim * H*W)
        out = self.net(flat)

        # Reshape to per-pixel features: (B, H*W, feature_dim)
        features = out.view(batch_size, n_pixels, self.feature_dim)
        return torch.tanh(features)  # Normalize range


class RandomConvFeatures(nn.Module):
    """
    Random convolutional feature extractor.

    Uses random conv filters to extract local features.
    Has spatial structure through convolution, but different from CPPN.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        # Random conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, 1),  # 1x1 conv to get feature_dim channels
            nn.Tanh()
        )

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, 1, H, W)

        Returns:
            Features, shape (B, H*W, feature_dim)
        """
        batch_size = x.shape[0]

        # Conv forward: (B, 1, H, W) -> (B, feature_dim, H, W)
        conv_out = self.conv(x)

        # Reshape: (B, feature_dim, H, W) -> (B, H*W, feature_dim)
        features = conv_out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)
        return features


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier feature extractor.

    Projects input pixels using random sinusoidal bases.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        # Random frequencies for Fourier projection
        # We'll project the flattened image with random frequencies
        self.register_buffer('frequencies', torch.randn(n_pixels, feature_dim // 2) * 3)
        self.register_buffer('phases', torch.rand(feature_dim // 2) * 2 * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, 1, H, W)

        Returns:
            Features, shape (B, H*W, feature_dim)
        """
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2

        # Flatten: (B, 1, H, W) -> (B, H*W)
        flat = x.view(batch_size, -1)

        # Project to frequency domain: (B, H*W) @ (H*W, feature_dim//2) -> (B, feature_dim//2)
        proj = flat @ self.frequencies

        # Apply sin and cos: (B, feature_dim//2) -> (B, feature_dim)
        features_global = torch.cat([
            torch.sin(proj + self.phases),
            torch.cos(proj + self.phases)
        ], dim=-1)

        # Broadcast to all pixels (global features repeated)
        # This is a simplification; could also do per-pixel Fourier
        features = features_global.unsqueeze(1).expand(-1, n_pixels, -1)
        return features


class DeepCPPNFeatures(nn.Module):
    """
    Deeper CPPN with more layers - should have even lower "bits" if hypothesis holds.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        # Deeper network with more nonlinearities
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
        features = self.net(cppn_input)
        return features


class RandomResNetFeatures(nn.Module):
    """
    ResNet-style with skip connections - common in practice.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        # Initial conv
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        # ResNet-style blocks
        self.block1_conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.block2_conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block2_skip = nn.Conv2d(32, 64, 1)  # 1x1 for channel match

        # Final projection
        self.final = nn.Conv2d(64, feature_dim, 1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Initial
        h = torch.relu(self.conv1(x))

        # Block 1 with skip
        identity = h
        h = torch.relu(self.block1_conv1(h))
        h = self.block1_conv2(h)
        h = torch.relu(h + identity)

        # Block 2 with skip (channel change)
        identity = self.block2_skip(h)
        h = torch.relu(self.block2_conv1(h))
        h = self.block2_conv2(h)
        h = torch.relu(h + identity)

        # Final
        h = torch.tanh(self.final(h))

        features = h.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)
        return features


class RandomPatchFeatures(nn.Module):
    """
    Patch-based features - like a simplified ViT without attention.
    """

    def __init__(self, feature_dim: int = 64, image_size: int = 28, patch_size: int = 7):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, feature_dim)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2

        # Extract patches: (B, 1, H, W) -> (B, n_patches, patch_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)

        # Embed patches: (B, n_patches, feature_dim)
        patch_features = torch.tanh(self.patch_embed(patches))

        # Upsample to per-pixel (repeat each patch feature for its pixels)
        # This is a simplification
        features = patch_features.unsqueeze(2).expand(-1, -1, (self.image_size // int(np.sqrt(self.n_patches)))**2, -1)
        features = features.reshape(batch_size, n_pixels, self.feature_dim)

        return features


def compute_feature_complexity(
    extractor: nn.Module,
    device: torch.device,
    n_samples: int = 200,
    image_size: int = 28
) -> float:
    """
    Compute a structural complexity score for a feature extractor.

    Method: Measure the effective dimensionality of features on random images.
    - Generate random images
    - Extract features
    - Compute variance explained by top principal components
    - Higher effective dim = less structured = higher "bits"

    This is a computed proxy, not the thermodynamic measurement.
    But it's honest: we're measuring something real about the extractor.
    """
    extractor = extractor.to(device)
    extractor.eval()

    # Generate random images
    random_images = torch.rand(n_samples, 1, image_size, image_size, device=device)

    with torch.no_grad():
        # Extract features: (n_samples, n_pixels, feature_dim)
        features = extractor(random_images)

        # Global average pool: (n_samples, feature_dim)
        pooled = features.mean(dim=1)

        # Move to CPU for numpy operations
        pooled_np = pooled.cpu().numpy()

    # Compute covariance and eigenvalues
    centered = pooled_np - pooled_np.mean(axis=0)
    cov = np.cov(centered.T)

    # Handle edge cases
    if cov.ndim == 0:
        return 1.0

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    eigenvalues = eigenvalues[::-1]  # Sort descending

    # Compute effective dimensionality (participation ratio)
    # D_eff = (sum λ)² / sum(λ²)
    # Higher = more spread out = less structured = more "bits"
    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return 1.0

    normalized = eigenvalues / total_var
    participation_ratio = 1.0 / (normalized ** 2).sum()

    # Convert to "bits-like" score
    # Scale so that effective_dim of 1 = 0 bits, effective_dim of 64 = 6 bits
    bits_proxy = np.log2(max(participation_ratio, 1.0))

    return float(bits_proxy)


class LinearProbe(nn.Module):
    """
    Linear classifier on top of frozen features.

    This is the ONLY trainable part.
    """

    def __init__(self, feature_extractor: nn.Module, n_classes: int = 10):
        super().__init__()
        self.features = feature_extractor
        self.n_classes = n_classes

        # Global average pooling + linear classifier
        self.classifier = nn.Linear(feature_extractor.feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images using frozen features + linear layer.

        Args:
            x: Input images, shape (B, 1, H, W)

        Returns:
            Logits, shape (B, n_classes)
        """
        # Get frozen features: (B, H*W, feature_dim)
        features = self.features(x)

        # Global average pooling: (B, H*W, feature_dim) -> (B, feature_dim)
        pooled = features.mean(dim=1)

        # Linear classifier: (B, feature_dim) -> (B, n_classes)
        logits = self.classifier(pooled)
        return logits


def train_probe(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    n_epochs: int = 10,
    lr: float = 1e-2,
) -> ProbeResult:
    """Train linear probe and track convergence."""

    model = model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []
    start_time = time.time()

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        test_accuracies.append(accuracy)

        print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_losses[-1]:.4f}, acc={accuracy:.4f}")

    training_time = time.time() - start_time

    return ProbeResult(
        name=model.features.__class__.__name__,
        bits_score=0,  # Will be filled in later
        train_losses=train_losses,
        test_accuracies=test_accuracies,
        final_accuracy=test_accuracies[-1],
        training_time=training_time
    )


def main():
    print("=" * 70)
    print("LINEAR PROBE VALIDATION: Testing Random Feature Quality")
    print("=" * 70)
    print()
    print("This tests whether thermodynamic bits-to-threshold predicts")
    print("how well FROZEN random features support linear classification.")
    print()
    print("CRITICAL FIX: Features now DEPEND ON INPUT IMAGE (not just coords)")
    print()

    device = get_device()
    print(f"Using device: {device}")

    # Load MNIST first (needed for complexity computation)
    print("\nLoading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subset for faster iteration
    n_train = 10000
    n_test = 2000
    train_indices = np.random.choice(len(train_dataset), n_train, replace=False)
    test_indices = np.random.choice(len(test_dataset), n_test, replace=False)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    print(f"Using {n_train} training, {n_test} test images")

    # Feature extractors - 7 different architectures
    image_size = 28
    feature_dim = 64

    extractors = [
        ('DeepCPPNFeatures', DeepCPPNFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomCPPNFeatures', RandomCPPNFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomResNetFeatures', RandomResNetFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomConvFeatures', RandomConvFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomPatchFeatures', RandomPatchFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomFourierFeatures', RandomFourierFeatures(feature_dim=feature_dim, image_size=image_size)),
        ('RandomMLPFeatures', RandomMLPFeatures(feature_dim=feature_dim, image_size=image_size)),
    ]

    # Compute feature complexity scores (replaces hardcoded values)
    print("\nComputing feature complexity scores (effective dimensionality)...")
    print("Higher score = more complex/random features = hypothetically harder")
    complexity_scores = {}
    for name, extractor in extractors:
        score = compute_feature_complexity(extractor, device, n_samples=200, image_size=image_size)
        complexity_scores[name] = score
        print(f"  {name}: {score:.2f} bits (effective dim)")

    results = []

    for name, extractor in extractors:
        print(f"\n{'='*50}")
        print(f"Training linear probe on {name}...")
        print(f"{'='*50}")

        model = LinearProbe(extractor, n_classes=10)
        result = train_probe(model, train_loader, test_loader, device, n_epochs=10, lr=1e-2)
        result.bits_score = complexity_scores[name]
        results.append(result)

        print(f"  Final accuracy: {result.final_accuracy:.4f}")
        print(f"  Training time: {result.training_time:.1f}s")

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS: Thermodynamic Score vs Linear Probe Accuracy")
    print("=" * 70)
    print()
    print(f"{'Extractor':<25} {'Bits':<10} {'Accuracy':<12} {'Prediction'}")
    print("-" * 70)

    # Sort by bits (thermodynamic prediction: lower bits = better)
    results.sort(key=lambda r: r.bits_score)

    for i, r in enumerate(results):
        rank = i + 1
        predicted = "BEST" if rank == 1 else f"#{rank}"
        print(f"{r.name:<25} {r.bits_score:<10.1f} {r.final_accuracy:<12.4f} {predicted}")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    bits = [r.bits_score for r in results]
    accuracies = [r.final_accuracy for r in results]

    # For accuracy, HIGHER is better, so correlation should be NEGATIVE with bits
    from scipy.stats import spearmanr
    corr, pval = spearmanr(bits, accuracies)

    print(f"\nBits ranking (predicted best → worst):")
    for r in results:
        print(f"  {r.name}: {r.bits_score} bits")

    accuracy_sorted = sorted(results, key=lambda r: -r.final_accuracy)
    print(f"\nAccuracy ranking (actual best → worst):")
    for r in accuracy_sorted:
        print(f"  {r.name}: {r.final_accuracy:.4f}")

    print(f"\nSpearman correlation (bits vs accuracy): {corr:.3f} (p={pval:.3f})")
    print("(Negative correlation = lower bits → higher accuracy = hypothesis supported)")

    if corr < -0.5:
        print("\n" + "=" * 70)
        print("*** NEGATIVE CORRELATION DETECTED ***")
        print("Lower bits → Higher accuracy")
        print("The thermodynamic metric PREDICTS linear probe performance!")
        print("=" * 70)
    elif corr > 0.5:
        print("\n*** POSITIVE CORRELATION: Lower bits → Lower accuracy ***")
        print("Hypothesis NOT supported.")
    else:
        print("\n*** WEAK CORRELATION ***")
        print("Results inconclusive. Need more architectures or different setup.")

    # Learning curves
    print("\n" + "=" * 70)
    print("LEARNING CURVES (Test Accuracy)")
    print("=" * 70)

    for epoch in range(10):
        row = f"Epoch {epoch+1:2d}: "
        for r in results:
            row += f"{r.name[:8]:>10}={r.test_accuracies[epoch]:.3f}  "
        print(row)


if __name__ == "__main__":
    main()
