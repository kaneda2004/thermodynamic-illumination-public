#!/usr/bin/env python3
"""Quick probe with 3 epochs to see if ConvNets catch up"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy import stats
import time

class ConvClassifier(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(128*16, 10)
        )
    def forward(self, x): return self.net(x)

class MLPClassifier(nn.Module):
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
        return self.fc(self.transformer(x)[:, 0])

STRUCTURE_SCORES = {
    'DeepConv': 0.89,
    'ShallowConv': 0.50,
    'MLP': 0.00,
    'ViT': 0.00,
}

def train_epochs(model, train_loader, epochs=3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()
    return model

def evaluate(model, test_loader, device='cpu'):
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
    print("QUICK PROBE: 3 EPOCHS")
    print("=" * 60)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

    architectures = [
        ('DeepConv', ConvClassifier),
        ('ShallowConv', ShallowConvClassifier),
        ('MLP', MLPClassifier),
        ('ViT', ViTClassifier),
    ]

    results = []
    print("\nTraining (3 epochs each on full 60k)...")
    print("-" * 50)

    start = time.time()
    for name, cls in architectures:
        torch.manual_seed(42)
        model = cls()
        model = train_epochs(model, train_loader, epochs=3)
        acc = evaluate(model, test_loader)
        structure = STRUCTURE_SCORES[name]
        results.append((name, structure, acc))
        print(f"{name:15s}: Structure={structure:.2f}, Accuracy={acc:.4f}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    structures = [r[1] for r in results]
    accuracies = [r[2] for r in results]
    r, p = stats.pearsonr(structures, accuracies)

    print("\n" + "=" * 60)
    print(f"CORRELATION: r = {r:.3f}, p = {p:.4f}")
    print("=" * 60)

    if abs(r) < 0.3 or p > 0.1:
        print("→ NO CORRELATION: Structure doesn't predict classification")
    elif r > 0.5 and p < 0.1:
        print("→ POSITIVE: High structure → High accuracy")
    elif r < -0.5 and p < 0.1:
        print("→ NEGATIVE: High structure → Low accuracy")
