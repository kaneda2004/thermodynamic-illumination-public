#!/usr/bin/env python3
"""
Sample Efficiency Validation: Rigorous Test of Thermodynamic Hypothesis

This tests whether lower "bits" (better structural alignment) predicts
lower sample requirements for a given accuracy target.

Protocol (per reviewer recommendations):
1. Fix architecture, optimizer, training steps
2. Vary dataset size N
3. Measure final test accuracy at each N
4. Find N_required to reach target accuracy
5. Correlate N_required with computed bits
6. Run across 20+ architectures, 5 seeds each
7. Bootstrap CI for correlation
8. Replicate across two datasets

Usage:
    uv run python sample_efficiency_validation.py [--quick]
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Architecture Zoo: 20+ variants
# =============================================================================

class BaseCPPN(nn.Module):
    """Base CPPN that takes (x, y, r, pixel) -> features."""

    def __init__(self, feature_dim: int, image_size: int, hidden_dims: list[int], activation: str = 'tanh'):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        act_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]

        layers = []
        in_dim = 4  # x, y, r, pixel
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_fn()])
            in_dim = h
        layers.append(nn.Linear(in_dim, feature_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = False

        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([xx.flatten(), yy.flatten(), r.flatten()], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pixels = x.view(batch_size, -1)
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_values = pixels.unsqueeze(-1)
        cppn_input = torch.cat([coords_batch, pixel_values], dim=-1)
        return self.net(cppn_input)


class BaseConv(nn.Module):
    """Base ConvNet with configurable depth/width."""

    def __init__(self, feature_dim: int, image_size: int, channels: list[int], activation: str = 'relu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        act_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]

        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend([nn.Conv2d(in_ch, ch, 3, padding=1), act_fn()])
            in_ch = ch
        layers.append(nn.Conv2d(in_ch, feature_dim, 1))
        layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.conv(x)
        return out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)


class BaseMLP(nn.Module):
    """Base MLP with configurable depth/width."""

    def __init__(self, feature_dim: int, image_size: int, hidden_dims: list[int], activation: str = 'relu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        act_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]

        layers = []
        in_dim = n_pixels
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_fn()])
            in_dim = h
        layers.append(nn.Linear(in_dim, feature_dim * n_pixels))

        self.net = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        n_pixels = self.image_size ** 2
        flat = x.view(batch_size, -1)
        out = self.net(flat)
        return torch.tanh(out.view(batch_size, n_pixels, self.feature_dim))


class BaseFourier(nn.Module):
    """Fourier features with configurable frequency scale."""

    def __init__(self, feature_dim: int, image_size: int, freq_scale: float = 3.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        n_pixels = image_size ** 2

        self.register_buffer('frequencies', torch.randn(n_pixels, feature_dim // 2) * freq_scale)
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


class BaseResNet(nn.Module):
    """ResNet-style with configurable depth."""

    def __init__(self, feature_dim: int, image_size: int, n_blocks: int = 2, width: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv1 = nn.Conv2d(1, width, 3, padding=1)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1),
            ))

        self.final = nn.Conv2d(width, feature_dim, 1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = torch.relu(self.conv1(x))
        for block in self.blocks:
            h = torch.relu(block(h) + h)
        h = torch.tanh(self.final(h))
        return h.permute(0, 2, 3, 1).reshape(batch_size, -1, self.feature_dim)


def create_architecture_zoo(feature_dim: int = 64, image_size: int = 28) -> list[tuple[str, nn.Module]]:
    """Create 20+ architecture variants."""
    architectures = []

    # CPPNs with varying depth
    for depth, name_suffix in [(2, 'shallow'), (3, 'medium'), (5, 'deep')]:
        hidden = [128] * depth
        architectures.append((f'CPPN_{name_suffix}', BaseCPPN(feature_dim, image_size, hidden, 'tanh')))

    # CPPNs with varying width
    for width, name_suffix in [(64, 'narrow'), (256, 'wide')]:
        architectures.append((f'CPPN_{name_suffix}', BaseCPPN(feature_dim, image_size, [width, width], 'tanh')))

    # CPPNs with different activations
    for act in ['relu', 'gelu', 'silu']:
        architectures.append((f'CPPN_{act}', BaseCPPN(feature_dim, image_size, [128, 128], act)))

    # ConvNets with varying depth
    for depth, name_suffix in [(2, 'shallow'), (3, 'medium'), (5, 'deep')]:
        channels = [32] * depth
        architectures.append((f'Conv_{name_suffix}', BaseConv(feature_dim, image_size, channels, 'relu')))

    # ConvNets with varying width
    for width, name_suffix in [(16, 'narrow'), (64, 'wide')]:
        architectures.append((f'Conv_{name_suffix}', BaseConv(feature_dim, image_size, [width, width], 'relu')))

    # MLPs with varying depth
    for depth, name_suffix in [(2, 'shallow'), (3, 'medium'), (4, 'deep')]:
        hidden = [256] * depth
        architectures.append((f'MLP_{name_suffix}', BaseMLP(feature_dim, image_size, hidden, 'relu')))

    # MLPs with varying width
    for width, name_suffix in [(128, 'narrow'), (512, 'wide')]:
        architectures.append((f'MLP_{name_suffix}', BaseMLP(feature_dim, image_size, [width, width], 'relu')))

    # Fourier with varying frequency scales
    for scale, name_suffix in [(1.0, 'low_freq'), (3.0, 'med_freq'), (10.0, 'high_freq')]:
        architectures.append((f'Fourier_{name_suffix}', BaseFourier(feature_dim, image_size, scale)))

    # ResNets with varying depth
    for n_blocks, name_suffix in [(1, 'shallow'), (2, 'medium'), (4, 'deep')]:
        architectures.append((f'ResNet_{name_suffix}', BaseResNet(feature_dim, image_size, n_blocks, 32)))

    return architectures


# =============================================================================
# Complexity Measurement (same as before)
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
# Linear Probe Training
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, feature_extractor: nn.Module, n_classes: int = 10):
        super().__init__()
        self.features = feature_extractor
        self.classifier = nn.Linear(feature_extractor.feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


def train_and_evaluate(
    extractor: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-2,
) -> float:
    """Train linear probe and return final test accuracy."""
    model = LinearProbe(extractor, n_classes=10).to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_data)

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            data = train_data[batch_idx].to(device)
            target = train_labels[batch_idx].to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        correct = 0
        total = 0
        for i in range(0, len(test_data), batch_size):
            data = test_data[i:i+batch_size].to(device)
            target = test_labels[i:i+batch_size].to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


# =============================================================================
# Sample Efficiency Measurement
# =============================================================================

@dataclass
class SampleEfficiencyResult:
    name: str
    bits: float
    n_values: list[int]
    accuracies: list[float]  # accuracy at each N
    n_required: float  # N to reach target accuracy (interpolated)


def measure_sample_efficiency(
    extractor: nn.Module,
    name: str,
    bits: float,
    full_train_data: torch.Tensor,
    full_train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    n_values: list[int],
    target_accuracy: float = 0.20,  # Lowered from 0.30
    n_epochs: int = 10,
) -> SampleEfficiencyResult:
    """Measure accuracy at different dataset sizes."""
    accuracies = []

    for n in n_values:
        # Sample subset
        indices = torch.randperm(len(full_train_data))[:n]
        train_data = full_train_data[indices]
        train_labels = full_train_labels[indices]

        # Use the same extractor (frozen weights) - only classifier is reset
        acc = train_and_evaluate(
            extractor, train_data, train_labels,
            test_data, test_labels, device, n_epochs
        )
        accuracies.append(acc)

    # Interpolate to find N_required
    n_required = interpolate_n_required(n_values, accuracies, target_accuracy)

    return SampleEfficiencyResult(
        name=name,
        bits=bits,
        n_values=n_values,
        accuracies=accuracies,
        n_required=n_required
    )


def get_extractor_kwargs(extractor: nn.Module) -> dict:
    """Extract constructor kwargs from extractor."""
    if isinstance(extractor, BaseCPPN):
        # Reconstruct hidden_dims from network
        hidden_dims = []
        for module in extractor.net:
            if isinstance(module, nn.Linear) and module.out_features != extractor.feature_dim:
                hidden_dims.append(module.out_features)
        # Detect activation
        act = 'tanh'
        for module in extractor.net:
            if isinstance(module, nn.ReLU):
                act = 'relu'
                break
            elif isinstance(module, nn.GELU):
                act = 'gelu'
                break
            elif isinstance(module, nn.SiLU):
                act = 'silu'
                break
        return {'hidden_dims': hidden_dims, 'activation': act}

    elif isinstance(extractor, BaseConv):
        channels = []
        for module in extractor.conv:
            if isinstance(module, nn.Conv2d) and module.out_channels != extractor.feature_dim:
                channels.append(module.out_channels)
        act = 'relu'
        for module in extractor.conv:
            if isinstance(module, nn.Tanh) and module != extractor.conv[-1]:
                act = 'tanh'
                break
            elif isinstance(module, nn.GELU):
                act = 'gelu'
                break
        return {'channels': channels, 'activation': act}

    elif isinstance(extractor, BaseMLP):
        hidden_dims = []
        for module in extractor.net:
            if isinstance(module, nn.Linear):
                if module.out_features != extractor.feature_dim * (extractor.image_size ** 2):
                    hidden_dims.append(module.out_features)
        act = 'relu'
        for module in extractor.net:
            if isinstance(module, nn.Tanh):
                act = 'tanh'
                break
            elif isinstance(module, nn.GELU):
                act = 'gelu'
                break
        return {'hidden_dims': hidden_dims, 'activation': act}

    elif isinstance(extractor, BaseFourier):
        return {'freq_scale': float(extractor.frequencies.std().item() / 3.0) * 3.0}

    elif isinstance(extractor, BaseResNet):
        return {'n_blocks': len(extractor.blocks), 'width': extractor.conv1.out_channels}

    return {}


def interpolate_n_required(n_values: list[int], accuracies: list[float], target: float) -> float:
    """Linear interpolation to find N required for target accuracy."""
    for i in range(len(accuracies) - 1):
        if accuracies[i] <= target <= accuracies[i+1]:
            # Linear interpolation
            t = (target - accuracies[i]) / (accuracies[i+1] - accuracies[i] + 1e-10)
            return n_values[i] + t * (n_values[i+1] - n_values[i])
        elif accuracies[i] >= target >= accuracies[i+1]:
            # Decreasing (shouldn't happen normally)
            t = (target - accuracies[i]) / (accuracies[i+1] - accuracies[i] - 1e-10)
            return n_values[i] + t * (n_values[i+1] - n_values[i])

    # Target not reached
    if max(accuracies) < target:
        return float('inf')
    # Target exceeded at smallest N
    return float(n_values[0])


# =============================================================================
# Bootstrap Correlation Analysis
# =============================================================================

def bootstrap_correlation(
    bits: list[float],
    n_required: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float, float, float]:
    """
    Bootstrap Spearman correlation with confidence interval.

    Returns: (correlation, p_value, ci_low, ci_high)
    """
    bits = np.array(bits)
    n_required = np.array(n_required)

    # Filter out inf values
    valid = ~np.isinf(n_required)
    bits = bits[valid]
    n_required = n_required[valid]

    if len(bits) < 3:
        return 0.0, 1.0, -1.0, 1.0

    # Point estimate
    corr, pval = spearmanr(bits, n_required)

    # Bootstrap CI
    n = len(bits)
    bootstrap_corrs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        b_corr, _ = spearmanr(bits[idx], n_required[idx])
        if not np.isnan(b_corr):
            bootstrap_corrs.append(b_corr)

    if len(bootstrap_corrs) == 0:
        return corr, pval, -1.0, 1.0

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_corrs, alpha/2 * 100)
    ci_high = np.percentile(bootstrap_corrs, (1 - alpha/2) * 100)

    return corr, pval, ci_low, ci_high


# =============================================================================
# Main Experiment
# =============================================================================

def load_dataset(name: str, n_train: int, n_test: int):
    """Load MNIST or FashionMNIST."""
    transform = transforms.Compose([transforms.ToTensor()])

    if name == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif name == 'fashion':
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Convert to tensors
    train_indices = np.random.choice(len(train_dataset), n_train, replace=False)
    test_indices = np.random.choice(len(test_dataset), n_test, replace=False)

    train_data = torch.stack([train_dataset[i][0] for i in train_indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    test_data = torch.stack([test_dataset[i][0] for i in test_indices])
    test_labels = torch.tensor([test_dataset[i][1] for i in test_indices])

    return train_data, train_labels, test_data, test_labels


def run_experiment(
    dataset_name: str,
    device: torch.device,
    n_seeds: int = 5,
    quick: bool = False,
) -> dict:
    """Run full sample efficiency experiment on one dataset."""

    print(f"\n{'='*70}")
    print(f"SAMPLE EFFICIENCY EXPERIMENT: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Parameters
    n_train = 10000 if not quick else 5000
    n_test = 2000
    n_values = [100, 250, 500, 1000, 2500, 5000] if not quick else [100, 500, 2000]
    target_accuracy = 0.20  # Lowered - 30% too hard for random features
    n_epochs = 10 if not quick else 5
    image_size = 28
    feature_dim = 64

    print(f"Dataset sizes to test: {n_values}")
    print(f"Target accuracy: {target_accuracy}")
    print(f"Seeds per architecture: {n_seeds}")

    # Load data
    print(f"\nLoading {dataset_name}...")
    full_train_data, full_train_labels, test_data, test_labels = load_dataset(
        dataset_name, n_train, n_test
    )

    # Create architectures
    architectures = create_architecture_zoo(feature_dim, image_size)
    print(f"Testing {len(architectures)} architectures")

    # Compute complexity scores
    print("\nComputing complexity scores...")
    complexity_scores = {}
    for name, extractor in architectures:
        score = compute_feature_complexity(extractor, device, n_samples=200, image_size=image_size)
        complexity_scores[name] = score
        print(f"  {name}: {score:.2f} bits")

    # Run experiment with multiple seeds
    all_results = {name: [] for name, _ in architectures}

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        torch.manual_seed(seed * 1000)
        np.random.seed(seed * 1000)

        for i, (name, extractor) in enumerate(architectures):
            print(f"  [{i+1}/{len(architectures)}] {name}...", end=" ", flush=True)

            result = measure_sample_efficiency(
                extractor, name, complexity_scores[name],
                full_train_data, full_train_labels,
                test_data, test_labels, device,
                n_values, target_accuracy, n_epochs
            )
            all_results[name].append(result)

            if result.n_required == float('inf'):
                print(f"N_req=∞")
            else:
                print(f"N_req={result.n_required:.0f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    aggregated = []
    for name in complexity_scores:
        bits = complexity_scores[name]
        n_reqs = [r.n_required for r in all_results[name]]
        valid_n_reqs = [n for n in n_reqs if n != float('inf')]

        # Also track final accuracy at max N
        final_accs = [r.accuracies[-1] for r in all_results[name]]
        mean_final_acc = np.mean(final_accs)
        std_final_acc = np.std(final_accs)

        if valid_n_reqs:
            mean_n = np.mean(valid_n_reqs)
            std_n = np.std(valid_n_reqs)
        else:
            mean_n = float('inf')
            std_n = 0

        aggregated.append({
            'name': name,
            'bits': bits,
            'mean_n_required': mean_n,
            'std_n_required': std_n,
            'n_valid_seeds': len(valid_n_reqs),
            'mean_final_acc': mean_final_acc,
            'std_final_acc': std_final_acc,
        })

    # Sort by bits
    aggregated.sort(key=lambda x: x['bits'])

    print(f"\n{'Architecture':<18} {'Bits':<7} {'N_required':<16} {'Final Acc':<12} {'Seeds'}")
    print("-" * 70)
    for a in aggregated:
        if a['mean_n_required'] == float('inf'):
            n_str = "∞"
        else:
            n_str = f"{a['mean_n_required']:.0f} ± {a['std_n_required']:.0f}"
        acc_str = f"{a['mean_final_acc']:.3f} ± {a['std_final_acc']:.3f}"
        print(f"{a['name']:<18} {a['bits']:<7.2f} {n_str:<16} {acc_str:<12} {a['n_valid_seeds']}/{n_seeds}")

    # Correlation analysis with bootstrap
    bits_list = [a['bits'] for a in aggregated]
    n_required_list = [a['mean_n_required'] for a in aggregated]
    final_acc_list = [a['mean_final_acc'] for a in aggregated]

    # N_required correlation (hypothesis: positive - lower bits = lower N needed)
    corr_n, pval_n, ci_low_n, ci_high_n = bootstrap_correlation(bits_list, n_required_list)

    # Final accuracy correlation (hypothesis: negative - lower bits = higher accuracy)
    corr_acc, pval_acc = spearmanr(bits_list, final_acc_list)
    # Bootstrap for accuracy correlation
    n = len(bits_list)
    bootstrap_corrs = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        bc, _ = spearmanr(np.array(bits_list)[idx], np.array(final_acc_list)[idx])
        if not np.isnan(bc):
            bootstrap_corrs.append(bc)
    ci_low_acc = np.percentile(bootstrap_corrs, 2.5)
    ci_high_acc = np.percentile(bootstrap_corrs, 97.5)

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    print("\n--- Metric 1: Bits vs N_required ---")
    print(f"Spearman correlation: {corr_n:.3f} (p={pval_n:.4f})")
    print(f"95% Bootstrap CI: [{ci_low_n:.3f}, {ci_high_n:.3f}]")
    print("Hypothesis: POSITIVE (lower bits → lower N_required)")

    print("\n--- Metric 2: Bits vs Final Accuracy ---")
    print(f"Spearman correlation: {corr_acc:.3f} (p={pval_acc:.4f})")
    print(f"95% Bootstrap CI: [{ci_low_acc:.3f}, {ci_high_acc:.3f}]")
    print("Hypothesis: NEGATIVE (lower bits → higher accuracy)")

    # Combined verdict
    n_supported = (corr_n > 0.3 and pval_n < 0.05)
    acc_supported = (corr_acc < -0.3 and pval_acc < 0.05)
    n_contradicted = (corr_n < -0.3 and pval_n < 0.05)
    acc_contradicted = (corr_acc > 0.3 and pval_acc < 0.05)

    print("\n--- VERDICT ---")
    if n_supported or acc_supported:
        print("*** HYPOTHESIS SUPPORTED on at least one metric ***")
        verdict = "SUPPORTED"
    elif n_contradicted or acc_contradicted:
        print("*** HYPOTHESIS CONTRADICTED on at least one metric ***")
        verdict = "CONTRADICTED"
    else:
        print("*** INCONCLUSIVE: No significant correlation in hypothesized direction ***")
        verdict = "INCONCLUSIVE"

    return {
        'dataset': dataset_name,
        'corr_n_required': corr_n,
        'pval_n_required': pval_n,
        'ci_n_required': (ci_low_n, ci_high_n),
        'corr_accuracy': corr_acc,
        'pval_accuracy': pval_acc,
        'ci_accuracy': (ci_low_acc, ci_high_acc),
        'verdict': verdict,
        'results': aggregated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer samples')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    args = parser.parse_args()

    print("=" * 70)
    print("SAMPLE EFFICIENCY VALIDATION")
    print("Rigorous Test of Thermodynamic Hypothesis")
    print("=" * 70)
    print()
    print("Protocol:")
    print("  - 20+ architecture variants")
    print(f"  - {args.seeds} seeds each")
    print("  - Bootstrap CI for correlation")
    print("  - Replication across MNIST and FashionMNIST")
    print()

    device = get_device()
    print(f"Using device: {device}")

    # Run on both datasets
    results = {}

    for dataset in ['mnist', 'fashion']:
        results[dataset] = run_experiment(
            dataset, device,
            n_seeds=args.seeds,
            quick=args.quick,
        )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Replication Check")
    print("=" * 70)

    print("\n--- Bits vs N_required (hypothesis: positive) ---")
    print(f"{'Dataset':<12} {'Corr':<8} {'p-value':<10} {'95% CI':<20}")
    print("-" * 50)
    for dataset, r in results.items():
        ci_str = f"[{r['ci_n_required'][0]:.2f}, {r['ci_n_required'][1]:.2f}]"
        print(f"{dataset:<12} {r['corr_n_required']:<8.3f} {r['pval_n_required']:<10.4f} {ci_str:<20}")

    print("\n--- Bits vs Final Accuracy (hypothesis: negative) ---")
    print(f"{'Dataset':<12} {'Corr':<8} {'p-value':<10} {'95% CI':<20}")
    print("-" * 50)
    for dataset, r in results.items():
        ci_str = f"[{r['ci_accuracy'][0]:.2f}, {r['ci_accuracy'][1]:.2f}]"
        print(f"{dataset:<12} {r['corr_accuracy']:<8.3f} {r['pval_accuracy']:<10.4f} {ci_str:<20}")

    print("\n--- Verdicts ---")
    for dataset, r in results.items():
        print(f"{dataset}: {r['verdict']}")

    # Decision rule
    print("\n" + "=" * 70)
    print("DECISION RULE")
    print("=" * 70)

    both_supported = all(r['verdict'] == 'SUPPORTED' for r in results.values())
    both_contradicted = all(r['verdict'] == 'CONTRADICTED' for r in results.values())

    # Check accuracy correlation specifically (negative = supported)
    acc_supported = all(r['corr_accuracy'] < -0.3 and r['pval_accuracy'] < 0.05
                        for r in results.values())
    acc_contradicted = all(r['corr_accuracy'] > 0.3 and r['pval_accuracy'] < 0.05
                           for r in results.values())

    if acc_supported:
        print("*** HYPOTHESIS VALIDATED ***")
        print("Significant negative correlation (bits vs accuracy) on BOTH datasets.")
        print("Lower structural complexity → Higher accuracy with frozen random features.")
    elif both_contradicted or acc_contradicted:
        print("*** HYPOTHESIS FALSIFIED ***")
        print("The metric predicts the OPPOSITE of what was hypothesized.")
    elif both_supported:
        print("*** PARTIAL SUPPORT ***")
        print("Some metrics show support, but not the primary accuracy metric.")
    else:
        print("*** HYPOTHESIS NOT VALIDATED ***")
        print("Results do not replicate across datasets.")

    # Save results
    output_path = Path("sample_efficiency_results.json")
    with open(output_path, 'w') as f:
        # Convert inf to string for JSON
        def convert_inf(obj):
            if isinstance(obj, float) and np.isinf(obj):
                return "inf"
            elif isinstance(obj, dict):
                return {k: convert_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_inf(v) for v in obj]
            return obj

        json.dump(convert_inf(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
