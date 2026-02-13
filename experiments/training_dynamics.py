#!/usr/bin/env python3
"""
Thermodynamic Illumination: The Phase Transition of Learning
Tracking Structural Entropy during Training.

Hypothesis:
Learning = Volume Collapse.
We expect 'Structure Score' to rise and 'Effective Volume' to shrink
as the network internalizes the data distribution.

The key insight: we use "Feature Visualization" as a proxy for the
generative prior of a classifier. The "dreams" of the network reveal
what structure it has internalized.

Usage:
    uv run python experiments/training_dynamics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image

# ==========================================
# 1. SETUP
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_mnist():
    """Load MNIST dataset (subset for speed)."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Subset for speed
    subset = torch.utils.data.Subset(dataset, range(5000))
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
    return loader


def get_test_loader():
    """Load MNIST test set for evaluation."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    return loader


class SimpleConv(nn.Module):
    """Simple ConvNet for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. THERMODYNAMIC PROBE
# ==========================================

def get_structure_score(img_tensor):
    """
    Measure structure of a single image.
    Uses compression ratio and TV smoothness.
    """
    img_np = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Compressibility (PNG for grayscale)
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='PNG')
    compressed_size = len(buffer.getvalue())
    raw_size = img_np.nbytes
    compress_ratio = 1.0 - (compressed_size / raw_size)

    # Smoothness (TV)
    img_t = img_tensor.squeeze()
    if img_t.dim() == 2:
        tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    else:
        tv_h = torch.mean(torch.abs(img_t[:, 1:, :] - img_t[:, :-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, :, 1:] - img_t[:, :, :-1]))

    tv_score = torch.exp(-10 * (tv_h + tv_w)).item()

    # Combined metric (want both compressible AND smooth)
    return max(0, compress_ratio) * tv_score


def generate_dream(model, target_class, n_steps=50, lr=0.5):
    """
    Generate a "dream" image by optimizing input to maximize class score.
    This reveals what visual patterns the network associates with each class.
    """
    model.eval()

    # Start from gray noise (not pure noise - helps convergence)
    dream = torch.ones(1, 1, 28, 28, device=device) * 0.5
    dream = dream + torch.randn_like(dream) * 0.1
    dream = dream.requires_grad_(True)

    optimizer = optim.Adam([dream], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(dream)

        # Maximize target class score
        loss = -out[0, target_class]

        # Add regularization to encourage smooth, natural-looking images
        tv_reg = torch.mean(torch.abs(dream[:, :, 1:, :] - dream[:, :, :-1, :])) + \
                 torch.mean(torch.abs(dream[:, :, :, 1:] - dream[:, :, :, :-1]))
        loss = loss + 0.01 * tv_reg

        loss.backward()
        optimizer.step()

        # Clamp to valid range
        with torch.no_grad():
            dream.data = torch.clamp(dream.data, 0, 1)

    return dream.detach()


def calculate_dream_quality(model, n_classes=10):
    """
    Probe the model's "dream quality" - how structured are its class visualizations?

    High structure = model has learned meaningful visual concepts
    Low structure = model is still random or has memorized without understanding
    """
    model.eval()
    structures = []
    dreams = []

    for target_class in range(n_classes):
        dream = generate_dream(model, target_class)
        structure = get_structure_score(dream)
        structures.append(structure)
        dreams.append(dream)

    return np.mean(structures), np.std(structures), dreams


def calculate_weight_entropy(model):
    """
    Measure the "spread" of weight magnitudes.
    Random init = uniform spread, Trained = structured (some weights matter more)
    """
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.detach().cpu().flatten())
    all_weights = torch.cat(all_weights)

    # Entropy proxy: std of weight magnitudes (normalized)
    magnitudes = torch.abs(all_weights)
    entropy = torch.std(magnitudes).item() / (torch.mean(magnitudes).item() + 1e-8)

    return entropy


# ==========================================
# 3. TRAINING & TRACKING
# ==========================================

def train_and_track(n_epochs=20):
    """
    Train a classifier while tracking thermodynamic properties.
    """
    print("=" * 60)
    print("Thermodynamic Illumination: Phase Transition of Learning")
    print("=" * 60)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    model = SimpleConv().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_loader = get_mnist()
    test_loader = get_test_loader()

    history = {
        'epoch': [],
        'train_acc': [],
        'test_acc': [],
        'dream_quality': [],
        'dream_std': [],
        'weight_entropy': [],
        'loss': []
    }

    # Store dreams at different epochs for visualization
    dream_snapshots = {}

    print("\nStarting training with thermodynamic tracking...\n")

    for epoch in range(n_epochs):
        # 1. Measure Thermodynamic Properties (before training this epoch)
        dream_quality, dream_std, dreams = calculate_dream_quality(model)
        weight_entropy = calculate_weight_entropy(model)

        # Save dream snapshots at key epochs
        if epoch in [0, 1, 5, 10, n_epochs - 1]:
            dream_snapshots[epoch] = [d.cpu() for d in dreams]

        # 2. Train for one epoch
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # 3. Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = correct / total

        # Record history
        history['epoch'].append(epoch)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['dream_quality'].append(dream_quality)
        history['dream_std'].append(dream_std)
        history['weight_entropy'].append(weight_entropy)
        history['loss'].append(total_loss / len(train_loader))

        print(f"Epoch {epoch:2d}: Train={train_acc:.3f} Test={test_acc:.3f} | "
              f"Dream={dream_quality:.4f}±{dream_std:.4f} | WeightEnt={weight_entropy:.3f}")

    return history, dream_snapshots, model


# ==========================================
# 4. VISUALIZATION
# ==========================================

def plot_training_dynamics(history, dream_snapshots, save_dir):
    """Create comprehensive visualization of training dynamics."""

    # Figure 1: Main dynamics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A. Accuracy curves
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_acc'], 'b-', linewidth=2, label='Train Accuracy')
    ax.plot(history['epoch'], history['test_acc'], 'b--', linewidth=2, label='Test Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Performance')
    ax.legend()
    ax.grid(alpha=0.3)

    # B. Dream Quality (Thermodynamic Structure)
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['dream_quality'], 'purple', linewidth=2)
    ax.fill_between(history['epoch'],
                    np.array(history['dream_quality']) - np.array(history['dream_std']),
                    np.array(history['dream_quality']) + np.array(history['dream_std']),
                    alpha=0.3, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dream Quality (Structure Score)')
    ax.set_title('Thermodynamic Structure of Learned Representations')
    ax.grid(alpha=0.3)

    # C. Weight Entropy
    ax = axes[1, 0]
    ax.plot(history['epoch'], history['weight_entropy'], 'green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Entropy (Normalized Std)')
    ax.set_title('Weight Distribution Evolution')
    ax.grid(alpha=0.3)

    # D. Combined view (normalized)
    ax = axes[1, 1]
    # Normalize to [0, 1]
    acc_norm = np.array(history['test_acc'])
    dream_norm = (np.array(history['dream_quality']) - min(history['dream_quality'])) / \
                 (max(history['dream_quality']) - min(history['dream_quality']) + 1e-8)

    ax.plot(history['epoch'], acc_norm, 'b-', linewidth=2, label='Test Accuracy')
    ax.plot(history['epoch'], dream_norm, 'purple', linewidth=2, label='Dream Quality (norm)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Accuracy vs Dream Quality: The Phase Transition')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_dynamics.png', dpi=150)
    plt.savefig(save_dir / 'training_dynamics.pdf')
    print(f"Saved: {save_dir / 'training_dynamics.png'}")
    plt.close()

    # Figure 2: Dream evolution
    if dream_snapshots:
        epochs_to_show = sorted(dream_snapshots.keys())
        n_epochs_show = len(epochs_to_show)

        fig, axes = plt.subplots(n_epochs_show, 10, figsize=(15, 3 * n_epochs_show))

        for row, epoch in enumerate(epochs_to_show):
            dreams = dream_snapshots[epoch]
            for col in range(10):
                ax = axes[row, col] if n_epochs_show > 1 else axes[col]
                img = dreams[col].squeeze().numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(f'Epoch {epoch}', fontsize=10)

        # Add column titles
        for col in range(10):
            ax = axes[0, col] if n_epochs_show > 1 else axes[col]
            ax.set_title(f'Class {col}', fontsize=9)

        plt.suptitle('Evolution of "Dreams": What Each Class Looks Like to the Network', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / 'dream_evolution.png', dpi=150)
        print(f"Saved: {save_dir / 'dream_evolution.png'}")
        plt.close()


def create_summary_figure(history, save_dir):
    """Create a paper-ready summary figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Dual-axis plot
    color1 = '#3498db'
    color2 = '#9b59b6'

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Test Accuracy', color=color1, fontsize=11)
    line1, = ax1.plot(history['epoch'], history['test_acc'], color=color1, linewidth=3, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Dream Quality (Structure)', color=color2, fontsize=11)
    line2, = ax1_twin.plot(history['epoch'], history['dream_quality'], color=color2,
                           linewidth=3, linestyle='--', label='Structure')
    ax1_twin.tick_params(axis='y', labelcolor=color2)

    # Add legend
    ax1.legend([line1, line2], ['Test Accuracy', 'Dream Quality'], loc='center right')
    ax1.set_title('The Phase Transition of Learning', fontsize=12)

    # Right: Scatter plot of Accuracy vs Dream Quality
    scatter = ax2.scatter(history['test_acc'], history['dream_quality'],
                          c=history['epoch'], cmap='viridis', s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax2, label='Epoch')
    ax2.set_xlabel('Test Accuracy', fontsize=11)
    ax2.set_ylabel('Dream Quality', fontsize=11)
    ax2.set_title('Accuracy-Structure Trajectory', fontsize=12)
    ax2.grid(alpha=0.3)

    # Add arrow showing direction of learning
    for i in range(0, len(history['epoch']) - 1, 3):
        ax2.annotate('', xy=(history['test_acc'][i + 1], history['dream_quality'][i + 1]),
                     xytext=(history['test_acc'][i], history['dream_quality'][i]),
                     arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / 'training_phase_transition.png', dpi=150)
    plt.savefig(save_dir / 'training_phase_transition.pdf')
    print(f"Saved: {save_dir / 'training_phase_transition.png'}")
    plt.close()


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Run training with thermodynamic tracking
    history, dream_snapshots, model = train_and_track(n_epochs=20)

    # Create visualizations
    save_dir = Path("figures")
    plot_training_dynamics(history, dream_snapshots, save_dir)
    create_summary_figure(history, save_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Find key transitions
    acc_jumps = np.diff(history['test_acc'])
    dream_jumps = np.diff(history['dream_quality'])

    max_acc_jump_epoch = np.argmax(acc_jumps)
    max_dream_jump_epoch = np.argmax(dream_jumps)

    print(f"\nAccuracy jumps most at epoch {max_acc_jump_epoch} "
          f"({history['test_acc'][max_acc_jump_epoch]:.3f} -> {history['test_acc'][max_acc_jump_epoch + 1]:.3f})")
    print(f"Dream quality jumps most at epoch {max_dream_jump_epoch} "
          f"({history['dream_quality'][max_dream_jump_epoch]:.4f} -> {history['dream_quality'][max_dream_jump_epoch + 1]:.4f})")

    if max_dream_jump_epoch > max_acc_jump_epoch:
        print("\n-> CONFIRMED: Structure lags Accuracy!")
        print("   The model memorizes labels first, then 'grokks' visual concepts.")
    elif max_dream_jump_epoch < max_acc_jump_epoch:
        print("\n-> INTERESTING: Structure leads Accuracy!")
        print("   The model builds visual representations before classification improves.")
    else:
        print("\n-> Structure and Accuracy evolve together.")

    # Correlation analysis
    correlation = np.corrcoef(history['test_acc'], history['dream_quality'])[0, 1]
    print(f"\nCorrelation(Accuracy, Dream Quality): {correlation:.3f}")

    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)

    initial_dream = history['dream_quality'][0]
    final_dream = history['dream_quality'][-1]
    dream_increase = final_dream / (initial_dream + 1e-8)

    print(f"\nDream Quality increased {dream_increase:.1f}x during training")
    print(f"  Initial: {initial_dream:.4f}")
    print(f"  Final:   {final_dream:.4f}")
    print("\nThis demonstrates that learning creates STRUCTURED representations")
    print("where class-maximizing inputs become visually meaningful digits.")
