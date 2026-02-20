#!/usr/bin/env python3
"""
Thermodynamic Illumination: The Thermodynamic Scaling Law (v2)
Can we predict 'Test Accuracy' purely from 'Prior Volume' (Bits)?

Key Insight: The "slider" must vary ARCHITECTURE TYPE, not just kernel size.
- ConvNets have INHERENT structure from spatial connectivity
- MLPs have NO spatial structure
- Hybrids lie in between

We show: Test Accuracy correlates with Thermodynamic Volume across architecture types.

Usage:
    uv run python experiments/thermodynamic_scaling_law_v2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import io
from PIL import Image
from scipy import stats

# ==========================================
# 1. ARCHITECTURE TYPES (The True "Slider")
# ==========================================

class ConvGenerator(nn.Module):
    """Pure ConvNet: Strong spatial bias"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, seed):
        return self.net(seed)


class MLPGenerator(nn.Module):
    """Pure MLP: No spatial bias"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 32 * 32),
            nn.Sigmoid()
        )
    def forward(self, seed):
        return self.net(seed).view(-1, 1, 32, 32)


class HybridGenerator(nn.Module):
    """Hybrid: Conv + MLP mixing"""
    def __init__(self, mix_ratio=0.5):
        super().__init__()
        self.mix = mix_ratio
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),  # 32x32
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024), nn.ReLU(),
            nn.Linear(1024, 32 * 32),
            nn.Sigmoid()
        )
    def forward(self, seed):
        conv_out = self.conv(seed)
        mlp_out = self.mlp(seed).view(-1, 1, 32, 32)
        return self.mix * conv_out + (1 - self.mix) * mlp_out


class LocalConvGenerator(nn.Module):
    """1x1 Conv: Has conv structure but no spatial mixing"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),  # 32x32
            nn.Conv2d(32, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, seed):
        return self.net(seed)


class DepthwiseConvGenerator(nn.Module):
    """Depthwise separable: Weaker spatial mixing"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 32, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, seed):
        return self.net(seed)


# ==========================================
# 2. CLASSIFIER ARCHITECTURES (Matching)
# ==========================================

class ConvClassifier(nn.Module):
    """Pure ConvNet classifier"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    """Pure MLP classifier"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)


class HybridClassifier(nn.Module):
    """Hybrid classifier"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.mlp(self.conv(x))


class LocalConvClassifier(nn.Module):
    """1x1 Conv classifier (no spatial mixing)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. MEASURE STRUCTURE (Static)
# ==========================================

def get_structure_score(img_tensor):
    """Combined compression + smoothness metric"""
    img_np = (img_tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='PNG')
    ratio = len(buffer.getvalue()) / img_np.nbytes
    compress_score = max(0, 1.0 - ratio)

    img_t = img_tensor.squeeze()
    tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
    tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    tv_score = torch.exp(-5 * (tv_h + tv_w)).item()

    return compress_score * tv_score


def measure_structure(generator_class, n_live=15, max_iter=100):
    """Measure thermodynamic structure via simplified nested sampling"""
    model = generator_class()
    seed = torch.randn(1, 32, 8, 8)

    def get_w():
        return torch.nn.utils.parameters_to_vector(model.parameters())

    def set_w(v):
        torch.nn.utils.vector_to_parameters(v, model.parameters())

    live = []
    for _ in range(n_live):
        w = torch.randn_like(get_w())
        set_w(w)
        with torch.no_grad():
            s = get_structure_score(model(seed))
        live.append({'score': s, 'weights': w})

    live.sort(key=lambda x: x['score'])

    for i in range(max_iter):
        dead = live.pop(0)
        threshold = dead['score']

        survivor = live[np.random.randint(len(live))]
        nu = torch.randn_like(survivor['weights'])
        theta = np.random.rand() * 2 * math.pi
        theta_min, theta_max = theta - 2 * math.pi, theta

        found = False
        for _ in range(30):
            w_new = survivor['weights'] * math.cos(theta) + nu * math.sin(theta)
            set_w(w_new)
            with torch.no_grad():
                s_new = get_structure_score(model(seed))

            if s_new > threshold:
                live.append({'score': s_new, 'weights': w_new})
                live.sort(key=lambda x: x['score'])
                found = True
                break
            else:
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.rand() * (theta_max - theta_min) + theta_min

        if not found:
            live.append(dead)
            live.sort(key=lambda x: x['score'])

    return live[-1]['score']


# ==========================================
# 4. MEASURE ACCURACY (Dynamic)
# ==========================================

def measure_accuracy(classifier_class, train_loader, test_loader, epochs=10):
    """Train classifier and measure test accuracy"""
    model = classifier_class()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ==========================================
# 5. MAIN EXPERIMENT
# ==========================================

def run_scaling_law_experiment(n_runs=2):
    """Run the scaling law experiment across architecture types"""
    print("=" * 60)
    print("THE THERMODYNAMIC SCALING LAW v2")
    print("Architecture Types as the Inductive Bias Slider")
    print("=" * 60)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # Data: Few-shot MNIST
    transform = transforms.ToTensor()
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_subset = torch.utils.data.Subset(full_train, range(200))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"\nFew-shot regime: {len(train_subset)} training samples")

    # Architecture pairs (Generator, Classifier, Name)
    architectures = [
        (ConvGenerator, ConvClassifier, 'Conv 3x3'),
        (DepthwiseConvGenerator, ConvClassifier, 'Depthwise'),
        (HybridGenerator, HybridClassifier, 'Hybrid'),
        (LocalConvGenerator, LocalConvClassifier, '1x1 Conv'),
        (MLPGenerator, MLPClassifier, 'MLP'),
    ]

    results = {name: {'structures': [], 'accuracies': []} for _, _, name in architectures}

    for run in range(n_runs):
        print(f"\n{'='*40}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*40}")

        for gen_class, clf_class, name in architectures:
            print(f"\n{name}:")

            # Measure Structure
            struct = measure_structure(gen_class)
            results[name]['structures'].append(struct)
            print(f"  Structure: {struct:.4f}")

            # Measure Accuracy
            acc = measure_accuracy(clf_class, train_loader, test_loader)
            results[name]['accuracies'].append(acc)
            print(f"  Accuracy:  {acc:.3f}")

    # Aggregate
    summary = []
    for _, _, name in architectures:
        summary.append({
            'name': name,
            'structure_mean': np.mean(results[name]['structures']),
            'structure_std': np.std(results[name]['structures']),
            'accuracy_mean': np.mean(results[name]['accuracies']),
            'accuracy_std': np.std(results[name]['accuracies'])
        })

    structures = [s['structure_mean'] for s in summary]
    accuracies = [s['accuracy_mean'] for s in summary]
    names = [s['name'] for s in summary]

    correlation, p_value = stats.pearsonr(structures, accuracies)
    spearman_r, spearman_p = stats.spearmanr(structures, accuracies)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n| Architecture | Structure | Accuracy |")
    print("|--------------|-----------|----------|")
    for s in summary:
        print(f"| {s['name']:12} | {s['structure_mean']:.4f}±{s['structure_std']:.4f} | "
              f"{s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f} |")

    print(f"\nPearson correlation: r = {correlation:.3f}, p = {p_value:.4f}")
    print(f"Spearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.4f}")

    # ==========================================
    # VISUALIZATION
    # ==========================================

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    for i, (struct, acc, name) in enumerate(zip(structures, accuracies, names)):
        ax.scatter(struct, acc, s=300, c=[colors[i]], edgecolors='black',
                   linewidth=2, label=name, zorder=5)
        ax.annotate(name, (struct, acc), textcoords='offset points',
                    xytext=(10, 5), fontsize=10, fontweight='bold')

    # Fit line
    if len(structures) > 2:
        m, b = np.polyfit(structures, accuracies, 1)
        x_fit = np.linspace(min(structures) - 0.05, max(structures) + 0.05, 100)
        ax.plot(x_fit, m * x_fit + b, 'k--', alpha=0.7, linewidth=2,
                label=f'Linear fit (r={correlation:.3f})')

    ax.set_xlabel('Thermodynamic Structure (Static Prior Volume)', fontsize=12)
    ax.set_ylabel('Few-Shot Test Accuracy (Dynamic)', fontsize=12)
    ax.set_title(f'THE THERMODYNAMIC SCALING LAW\nr = {correlation:.3f}, p = {p_value:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'scaling_law_v2.png', dpi=150)
    plt.savefig(out_dir / 'scaling_law_v2.pdf')
    print(f"\nSaved: {out_dir / 'scaling_law_v2.png'}")
    plt.close()

    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)

    if correlation > 0.7:
        print(f"\n✓ STRONG CORRELATION (r = {correlation:.3f})")
        print("  Thermodynamic Structure PREDICTS Generalization!")
    elif correlation > 0.4:
        print(f"\n~ MODERATE CORRELATION (r = {correlation:.3f})")
        print("  Structure partially predicts accuracy.")
    else:
        print(f"\n✗ WEAK CORRELATION (r = {correlation:.3f})")
        print("  More architecture diversity needed.")

    return summary, correlation


if __name__ == "__main__":
    run_scaling_law_experiment(n_runs=2)
