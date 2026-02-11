#!/usr/bin/env python3
"""
Classification Trade-off Experiment: Testing the Generative-Discriminative Hypothesis
======================================================================================

Hypothesis: High-bits architectures (low structure) achieve better classification
accuracy than low-bits architectures (high structure) when capacity is matched.

This tests whether the "Generative-Discriminative Trade-off" claim is supported:
- Low-bits (high structure): Good at generation, bad at classification
- High-bits (low structure): Good at classification, bad at generation

Key Design: ALL architectures are capacity-matched (~250K params) to isolate
thermodynamic structure as the independent variable.

Usage:
    uv run python experiments/classification_tradeoff.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import io
import json
import math
from PIL import Image

# ==========================================
# CAPACITY-MATCHED ARCHITECTURE PAIRS
# Each pair: (Generator, Classifier, name, expected_structure)
# Target: ~250K parameters each
# ==========================================

# --- 1. DEEP CONVNET (Low bits / High structure) ---

class DeepConvGenerator(nn.Module):
    """Deep 6-layer ConvNet generator - ~250K params
    Expected: HIGH structure (strong spatial bias)
    """
    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 64, 2, 2))  # Small seed, many layers

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(64, 128),   # 4x4
            block(128, 128),  # 8x8
            block(128, 64),   # 16x16
            block(64, 32),    # 32x32
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self):
        return self.net(self.seed)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class DeepConvClassifier(nn.Module):
    """Deep ConvNet classifier - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- 2. SHALLOW CONVNET (Medium bits) ---

class ShallowConvGenerator(nn.Module):
    """Shallow 2-layer ConvNet generator - ~250K params
    Expected: MEDIUM structure (less smoothing)
    """
    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 256, 8, 8))

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self):
        return self.net(self.seed)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ShallowConvClassifier(nn.Module):
    """Shallow ConvNet classifier - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- 3. CPPN-STYLE (Low bits / High structure) ---

class CPPNGenerator(nn.Module):
    """Coordinate-based network - ~250K params
    Expected: HIGH structure (spectral bias toward smoothness)
    """
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size

        # Pre-compute coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        )
        r = torch.sqrt(x**2 + y**2)
        self.register_buffer('coords', torch.stack([x, y, r], dim=0).unsqueeze(0))  # 1, 3, H, W

        # Per-pixel MLP (implemented as 1x1 convs for efficiency)
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self):
        return self.net(self.coords)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class CPPNClassifier(nn.Module):
    """CPPN-style classifier with coordinate features - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.img_size = img_size

        # Coordinate features
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
            indexing='ij'
        )
        r = torch.sqrt(x**2 + y**2)
        self.register_buffer('coords', torch.stack([x, y, r], dim=0))  # 3, H, W

        # Per-pixel processing + image input
        self.per_pixel = nn.Sequential(
            nn.Conv2d(in_channels + 3, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, coords], dim=1)
        x = self.per_pixel(x)
        x = self.pool(x)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- 4. MLP (High bits / Low structure) ---

class MLPGenerator(nn.Module):
    """Pure MLP generator - ~250K params
    Expected: LOW structure (no spatial bias)
    """
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.seed = nn.Parameter(torch.randn(1, 64))

        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self):
        return self.net(self.seed).view(1, 1, self.img_size, self.img_size)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPClassifier(nn.Module):
    """Pure MLP classifier - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- 5. ViT-Tiny (High bits / Low structure) ---

class ViTGenerator(nn.Module):
    """Tiny ViT generator - ~250K params
    Expected: LOW structure (global attention scrambles)
    """
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 patches for 32x32 with 4x4

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_to_pixel = nn.Linear(embed_dim, patch_size * patch_size)

        self.img_size = img_size

    def forward(self):
        x = self.pos_embed  # 1, num_patches, embed_dim
        x = self.transformer(x)
        x = self.patch_to_pixel(x)  # 1, num_patches, patch_size^2

        # Reshape to image
        p = self.patch_size
        n = int(math.sqrt(self.num_patches))
        x = x.view(1, n, n, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(1, 1, n * p, n * p)
        return torch.sigmoid(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ViTClassifier(nn.Module):
    """Tiny ViT classifier - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28, patch_size=7, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        p = self.patch_size

        # Patchify
        x = x.unfold(2, p, p).unfold(3, p, p)  # B, C, H/p, W/p, p, p
        x = x.contiguous().view(B, -1, x.size(1) * p * p)  # B, num_patches, C*p*p
        x = self.patch_embed(x)  # B, num_patches, embed_dim

        # Add cls token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        return self.fc(x[:, 0])

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- 6. LOCAL ATTENTION (Medium-High bits) ---

class LocalAttentionGenerator(nn.Module):
    """Local attention generator (windowed) - ~250K params
    Expected: MEDIUM structure (restricted attention)
    """
    def __init__(self, img_size=32, patch_size=8, embed_dim=64, window_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.window_size = window_size
        self.patches_per_side = img_size // patch_size

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Simple windowed self-attention (3 layers)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            ) for _ in range(3)
        ])

        self.patch_to_pixel = nn.Linear(embed_dim, patch_size * patch_size)
        self.img_size = img_size

    def forward(self):
        x = self.pos_embed
        for layer in self.layers:
            x = x + layer(x)
        x = self.patch_to_pixel(x)

        p = self.patch_size
        n = self.patches_per_side
        x = x.view(1, n, n, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(1, 1, n * p, n * p)
        return torch.sigmoid(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LocalAttentionClassifier(nn.Module):
    """Local attention classifier - ~250K params"""
    def __init__(self, num_classes=10, in_channels=1, img_size=28, patch_size=7, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            ) for _ in range(4)
        ])

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, x.size(1) * p * p)
        x = self.patch_embed(x) + self.pos_embed

        for layer in self.layers:
            x = x + layer(x)

        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================
# STRUCTURE MEASUREMENT
# ==========================================

def get_structure_score(img_tensor):
    """Combined compression + smoothness metric (same as paper)"""
    # Handle different tensor shapes
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() == 3 and img_tensor.size(0) == 1:
        img_tensor = img_tensor.squeeze(0)

    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)

    # Compression score
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    compress_score = max(0, 1.0 - ratio)

    # Smoothness score (total variation)
    img_t = img_tensor.float()
    if img_t.dim() == 2:
        tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    else:
        tv_h = torch.mean(torch.abs(img_t[:, 1:, :] - img_t[:, :-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, :, 1:] - img_t[:, :, :-1]))

    tv_score = math.exp(-10 * (tv_h + tv_w).item())

    return compress_score * tv_score


def measure_structure(gen_class, n_samples=50):
    """Measure thermodynamic structure via random-weight outputs"""
    scores = []
    for _ in range(n_samples):
        model = gen_class()
        # Random weight initialization
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1.0)
            elif p.dim() == 1:
                nn.init.zeros_(p)

        with torch.no_grad():
            img = model()
            scores.append(get_structure_score(img))

    return np.mean(scores), np.std(scores)


# ==========================================
# CLASSIFICATION TRAINING
# ==========================================

def train_classifier(clf_class, train_loader, test_loader, num_classes=10,
                     in_channels=1, img_size=28, epochs=20, device='cpu'):
    """Train classifier and return test accuracy"""
    model = clf_class(num_classes=num_classes, in_channels=in_channels, img_size=img_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run_experiment(n_seeds=3):
    """Run the full classification trade-off experiment"""
    print("=" * 70)
    print("CLASSIFICATION TRADE-OFF EXPERIMENT")
    print("Testing: Do high-bits architectures achieve better classification?")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}\n")

    out_dir = Path(__file__).parent.parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Architecture pairs: (Generator, Classifier, name)
    architectures = [
        (DeepConvGenerator, DeepConvClassifier, 'DeepConv'),
        (ShallowConvGenerator, ShallowConvClassifier, 'ShallowConv'),
        (CPPNGenerator, CPPNClassifier, 'CPPN'),
        (LocalAttentionGenerator, LocalAttentionClassifier, 'LocalAttn'),
        (MLPGenerator, MLPClassifier, 'MLP'),
        (ViTGenerator, ViTClassifier, 'ViT'),
    ]

    # Verify parameter counts
    print("Architecture Parameter Counts:")
    print("-" * 50)
    for gen_cls, clf_cls, name in architectures:
        gen = gen_cls()
        clf = clf_cls()
        print(f"{name:15s}: Generator={gen.count_params():,}, Classifier={clf.count_params():,}")
    print()

    # Datasets
    datasets_config = [
        ('MNIST', datasets.MNIST, 1, 28, 10),
        ('FashionMNIST', datasets.FashionMNIST, 1, 28, 10),
        ('CIFAR10', datasets.CIFAR10, 3, 32, 10),
    ]

    results = {name: {'structure': None, 'structure_std': None, 'accuracy': {}}
               for _, _, name in architectures}

    # 1. Measure Structure (once per architecture)
    print("=" * 50)
    print("PHASE 1: Measuring Thermodynamic Structure")
    print("=" * 50)

    for gen_cls, _, name in architectures:
        struct_mean, struct_std = measure_structure(gen_cls, n_samples=30)
        results[name]['structure'] = struct_mean
        results[name]['structure_std'] = struct_std
        print(f"{name:15s}: Structure = {struct_mean:.4f} ± {struct_std:.4f}")

    print()

    # 2. Train Classifiers on Each Dataset
    for ds_name, ds_class, in_channels, img_size, num_classes in datasets_config:
        print("=" * 50)
        print(f"PHASE 2: Training on {ds_name}")
        print("=" * 50)

        # Load data
        if in_channels == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        train_data = ds_class('./data', train=True, download=True, transform=transform)
        test_data = ds_class('./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

        for _, clf_cls, name in architectures:
            accuracies = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    acc = train_classifier(
                        clf_cls, train_loader, test_loader,
                        num_classes=num_classes, in_channels=in_channels,
                        img_size=img_size, epochs=15, device=device
                    )
                    accuracies.append(acc)
                except Exception as e:
                    print(f"  {name} failed on {ds_name}: {e}")
                    accuracies.append(0.0)

            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies)
            results[name]['accuracy'][ds_name] = {'mean': acc_mean, 'std': acc_std}
            print(f"  {name:15s}: {acc_mean:.4f} ± {acc_std:.4f}")

    # 3. Compute Correlations
    print("\n" + "=" * 50)
    print("PHASE 3: Correlation Analysis")
    print("=" * 50)

    structures = [results[name]['structure'] for _, _, name in architectures]

    correlations = {}
    for ds_name, _, _, _, _ in datasets_config:
        accuracies = [results[name]['accuracy'][ds_name]['mean'] for _, _, name in architectures]

        r, p = stats.pearsonr(structures, accuracies)
        correlations[ds_name] = {'r': r, 'p': p}

        print(f"{ds_name:15s}: Pearson r = {r:.3f}, p = {p:.4f}")

        # Interpret
        if p < 0.05 and r > 0.5:
            print(f"  → POSITIVE correlation: High-bits architectures DO classify better")
        elif p < 0.05 and r < -0.5:
            print(f"  → NEGATIVE correlation: Low-bits architectures classify better (unexpected!)")
        else:
            print(f"  → NO significant correlation: Bits do not predict classification")

    # 4. Generate Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'DeepConv': '#2ecc71', 'ShallowConv': '#27ae60', 'CPPN': '#3498db',
              'LocalAttn': '#9b59b6', 'MLP': '#e74c3c', 'ViT': '#e67e22'}

    for idx, (ds_name, _, _, _, _) in enumerate(datasets_config):
        ax = axes[idx]

        for _, _, name in architectures:
            struct = results[name]['structure']
            acc = results[name]['accuracy'][ds_name]['mean']
            acc_err = results[name]['accuracy'][ds_name]['std']

            ax.errorbar(struct, acc, yerr=acc_err, fmt='o', markersize=10,
                       color=colors[name], label=name, capsize=5)

        # Regression line
        x = np.array(structures)
        y = np.array([results[name]['accuracy'][ds_name]['mean'] for _, _, name in architectures])
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 100)
        ax.plot(x_line, p_line(x_line), 'k--', alpha=0.5)

        r, p = correlations[ds_name]['r'], correlations[ds_name]['p']
        ax.set_title(f'{ds_name}\nr = {r:.3f}, p = {p:.3f}')
        ax.set_xlabel('Thermodynamic Structure')
        ax.set_ylabel('Classification Accuracy')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'classification_tradeoff.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(out_dir / 'classification_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Save Results
    with open(results_dir / 'classification_tradeoff.json', 'w') as f:
        json.dump({
            'results': results,
            'correlations': correlations,
            'architectures': [name for _, _, name in architectures]
        }, f, indent=2)

    print(f"\nFigure saved to {out_dir / 'classification_tradeoff.pdf'}")
    print(f"Results saved to {results_dir / 'classification_tradeoff.json'}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Generative-Discriminative Trade-off Hypothesis")
    print("=" * 70)

    avg_r = np.mean([correlations[ds]['r'] for ds in correlations])
    avg_p = np.mean([correlations[ds]['p'] for ds in correlations])

    print(f"Average correlation: r = {avg_r:.3f}")

    if avg_r > 0.3 and avg_p < 0.1:
        print("RESULT: Hypothesis SUPPORTED - High-bits → Better Classification")
        print("ACTION: STRENGTHEN claims in paper")
    elif avg_r < -0.3 and avg_p < 0.1:
        print("RESULT: OPPOSITE pattern - Low-bits → Better Classification")
        print("ACTION: REVISE claims (unexpected finding)")
    else:
        print("RESULT: Hypothesis NOT SUPPORTED - No correlation found")
        print("ACTION: REVISE paper to: 'Bits is a generative-only metric'")

    return results, correlations


if __name__ == "__main__":
    results, correlations = run_experiment(n_seeds=3)
