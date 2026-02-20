#!/usr/bin/env python3
"""
Quick Classification Probe: 1-epoch test of structure vs classification
Tests whether thermodynamic structure predicts classification ability.
Runtime: ~2-3 minutes total
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy import stats
import time

# Simple architectures matching our scaling law experiment
class ConvClassifier(nn.Module):
    """Deep ConvNet - HIGH structure expected"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

class ShallowConvClassifier(nn.Module):
    """Shallow ConvNet - MEDIUM structure expected"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(128*16, 10)
        )
    def forward(self, x): return self.net(x)

class MLPClassifier(nn.Module):
    """Pure MLP - LOW structure expected"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class ViTClassifier(nn.Module):
    """Tiny ViT - LOW structure expected (like MLP)"""
    def __init__(self, patch_size=7, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) ** 2
        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc = nn.Linear(embed_dim, 10)

    def forward(self, x):
        B, p = x.size(0), self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, -1, p*p)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.fc(x[:, 0])

class DepthwiseConvClassifier(nn.Module):
    """Depthwise separable - VERY HIGH structure expected"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, groups=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.Conv2d(32, 64, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 128, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# Known structure scores from our scaling law experiment (approximate)
STRUCTURE_SCORES = {
    'DeepConv': 0.89,      # ResNet-4 style
    'DepthwiseConv': 0.94, # Highest structure
    'ShallowConv': 0.50,   # Medium
    'MLP': 0.00,           # Zero structure
    'ViT': 0.00,           # Zero structure (like MLP)
}

def train_one_epoch(model, train_loader, device='cpu'):
    """Train for exactly 1 epoch, return accuracy"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, test_loader, device='cpu'):
    """Evaluate accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK CLASSIFICATION PROBE")
    print("1-epoch training to test structure vs classification")
    print("=" * 60)

    # Load MNIST (small subset for speed)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subset for faster training
    train_subset = torch.utils.data.Subset(train_data, range(10000))  # 10k samples
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

    architectures = [
        ('DeepConv', ConvClassifier),
        ('DepthwiseConv', DepthwiseConvClassifier),
        ('ShallowConv', ShallowConvClassifier),
        ('MLP', MLPClassifier),
        ('ViT', ViTClassifier),
    ]

    results = []
    print("\nTraining (1 epoch each on 10k samples)...")
    print("-" * 50)

    start = time.time()
    for name, cls in architectures:
        torch.manual_seed(42)
        model = cls()
        model = train_one_epoch(model, train_loader)
        acc = evaluate(model, test_loader)
        structure = STRUCTURE_SCORES[name]
        results.append((name, structure, acc))
        print(f"{name:15s}: Structure={structure:.2f}, Accuracy={acc:.4f}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    # Correlation analysis
    structures = [r[1] for r in results]
    accuracies = [r[2] for r in results]

    r, p = stats.pearsonr(structures, accuracies)

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Pearson r = {r:.3f}, p = {p:.4f}")
    print(f"n = {len(results)} architectures")

    if p < 0.05 and r > 0.5:
        print("\n→ SIGNIFICANT POSITIVE: High structure → High accuracy")
        print("  This would CONTRADICT the paper's finding!")
    elif p < 0.05 and r < -0.5:
        print("\n→ SIGNIFICANT NEGATIVE: High structure → Low accuracy")
        print("  This would SUPPORT the trade-off hypothesis!")
    else:
        print("\n→ NO SIGNIFICANT CORRELATION")
        print("  This CONFIRMS the paper: bits doesn't predict classification")

    print("\n" + "=" * 60)
    print("RAW DATA")
    print("=" * 60)
    print(f"{'Architecture':15s} {'Structure':>10s} {'Accuracy':>10s}")
    print("-" * 37)
    for name, struct, acc in sorted(results, key=lambda x: -x[1]):
        print(f"{name:15s} {struct:>10.2f} {acc:>10.4f}")
