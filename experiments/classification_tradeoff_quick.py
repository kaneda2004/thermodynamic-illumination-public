#!/usr/bin/env python3
"""
QUICK Classification Trade-off: MNIST only, 1 seed
Tests hypothesis with minimal compute to get preliminary results.
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
# CAPACITY-MATCHED ARCHITECTURES (~250K params)
# ==========================================

class DeepConvGenerator(nn.Module):
    """Deep ConvNet - HIGH structure"""
    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 64, 2, 2))
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
        self.net = nn.Sequential(
            block(64, 128), block(128, 128), block(128, 64), block(64, 32),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self):
        return self.net(self.seed)

class DeepConvClassifier(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class ShallowConvGenerator(nn.Module):
    """Shallow ConvNet - MEDIUM structure"""
    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 256, 8, 8))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self):
        return self.net(self.seed)

class ShallowConvClassifier(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256*16, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.fc(self.conv(x))

class MLPGenerator(nn.Module):
    """MLP - LOW structure"""
    def __init__(self, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.seed = nn.Parameter(torch.randn(1, 64))
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, img_size * img_size), nn.Sigmoid()
        )
    def forward(self):
        return self.net(self.seed).view(1, 1, self.img_size, self.img_size)

class MLPClassifier(nn.Module):
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

class ViTGenerator(nn.Module):
    """ViT - LOW structure"""
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_to_pixel = nn.Linear(embed_dim, patch_size * patch_size)
        self.img_size = img_size
    def forward(self):
        x = self.transformer(self.pos_embed)
        x = self.patch_to_pixel(x)
        p, n = self.patch_size, int(math.sqrt(self.num_patches))
        x = x.view(1, n, n, p, p).permute(0, 1, 3, 2, 4).contiguous().view(1, 1, n*p, n*p)
        return torch.sigmoid(x)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, img_size=28, patch_size=7, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        B, p = x.size(0), self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(B, -1, x.size(1)*p*p)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.fc(self.transformer(x)[:, 0])

# ==========================================
# MEASUREMENT FUNCTIONS
# ==========================================

def get_structure_score(img_tensor):
    if img_tensor.dim() == 4: img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() == 3 and img_tensor.size(0) == 1: img_tensor = img_tensor.squeeze(0)
    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    compress_score = max(0, 1.0 - len(buffer.getvalue()) / img_np.nbytes)
    img_t = img_tensor.float()
    if img_t.dim() == 2:
        tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    else:
        tv_h = torch.mean(torch.abs(img_t[:, 1:, :] - img_t[:, :-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, :, 1:] - img_t[:, :, :-1]))
    return compress_score * math.exp(-10 * (tv_h + tv_w).item())

def measure_structure(gen_class, n_samples=20):
    scores = []
    for i in range(n_samples):
        torch.manual_seed(i * 1000)  # Different seed for each sample
        model = gen_class()
        # Random weight initialization
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1.0)
        # Keep in train mode for BatchNorm to use batch stats
        model.train()
        with torch.no_grad():
            img = model()
            scores.append(get_structure_score(img))
    return np.mean(scores), np.std(scores)

def train_classifier(clf_class, train_loader, test_loader, epochs=10, device='cpu'):
    model = clf_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK CLASSIFICATION TRADE-OFF TEST")
    print("MNIST only, 1 seed - preliminary results")
    print("=" * 60)

    device = 'cpu'  # MPS has adaptive_avg_pool2d issues
    print(f"Device: {device}\n")

    architectures = [
        (DeepConvGenerator, DeepConvClassifier, 'DeepConv'),
        (ShallowConvGenerator, ShallowConvClassifier, 'ShallowConv'),
        (MLPGenerator, MLPClassifier, 'MLP'),
        (ViTGenerator, ViTClassifier, 'ViT'),
    ]

    # Measure structure
    print("Measuring Thermodynamic Structure...")
    print("-" * 40)
    structures = []
    for gen_cls, _, name in architectures:
        s_mean, s_std = measure_structure(gen_cls, n_samples=20)
        structures.append(s_mean)
        print(f"{name:15s}: {s_mean:.4f} ± {s_std:.4f}")

    # Load MNIST
    print("\nLoading MNIST...")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

    # Train classifiers
    print("\nTraining Classifiers (10 epochs each)...")
    print("-" * 40)
    accuracies = []
    for _, clf_cls, name in architectures:
        torch.manual_seed(42)
        acc = train_classifier(clf_cls, train_loader, test_loader, epochs=10, device=device)
        accuracies.append(acc)
        print(f"{name:15s}: {acc:.4f}")

    # Correlation
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    r, p = stats.pearsonr(structures, accuracies)
    print(f"Pearson r = {r:.3f}, p = {p:.4f}")

    if p < 0.1 and r > 0.3:
        print("\n→ POSITIVE trend: High structure → Higher accuracy")
        print("  This CONTRADICTS the hypothesis!")
        print("  (Expected: High structure → Lower accuracy)")
    elif p < 0.1 and r < -0.3:
        print("\n→ NEGATIVE trend: High structure → Lower accuracy")
        print("  This SUPPORTS the hypothesis!")
    else:
        print("\n→ NO significant correlation")
        print("  Hypothesis NOT SUPPORTED")
        print("  Bits may be a generative-only metric")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Architecture':15s} {'Structure':>10s} {'Accuracy':>10s}")
    print("-" * 37)
    for i, (_, _, name) in enumerate(architectures):
        print(f"{name:15s} {structures[i]:>10.4f} {accuracies[i]:>10.4f}")

    # Save results
    results = {
        'structures': {name: structures[i] for i, (_, _, name) in enumerate(architectures)},
        'accuracies': {name: accuracies[i] for i, (_, _, name) in enumerate(architectures)},
        'correlation': {'r': r, 'p': p}
    }
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'classification_tradeoff_quick.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'classification_tradeoff_quick.json'}")
