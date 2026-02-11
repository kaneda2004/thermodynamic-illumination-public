#!/usr/bin/env python3
"""
Thermodynamic Illumination: Frontier A
The Energy Barrier of Concepts in Diffusion Models

Hypothesis:
We can quantify the 'Bits of Separation' between classes by measuring
the Score Difference (Force) across the diffusion landscape.
High Barrier = Distinct Concepts. Low Barrier = Concept Bleed.

This experiment trains a toy conditional diffusion model on 2D data
(Ring vs Blob) and maps the thermodynamic landscape.

Usage:
    uv run python experiments/diffusion_thermodynamics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles, make_blobs

# ==========================================
# 1. DATA (The Concepts)
# ==========================================

def get_data(n_samples=3000):
    """
    Create two distinct 2D distributions:
    - Concept 0: Ring (outer circle)
    - Concept 1: Blob (central cluster)
    """
    # Concept 0: Ring (Outer)
    X_ring, _ = make_circles(n_samples=n_samples // 2, factor=0.5, noise=0.05)
    X_ring = X_ring[X_ring[:, 0] ** 2 + X_ring[:, 1] ** 2 > 0.3]  # Keep outer
    X_ring = X_ring[:n_samples // 2] * 2.0  # Scale up
    Y_ring = np.zeros(len(X_ring))

    # Concept 1: Blob (Center)
    X_blob = np.random.randn(n_samples // 2, 2) * 0.3
    Y_blob = np.ones(n_samples // 2)

    X = np.concatenate([X_ring, X_blob])
    Y = np.concatenate([Y_ring, Y_blob])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]

    return torch.FloatTensor(X), torch.LongTensor(Y)


# ==========================================
# 2. TOY DIFFUSION MODEL
# ==========================================

class ConditionalScoreNet(nn.Module):
    """
    Estimates the Score (Gradient of Log Density).
    Input: x (2D), t (noise level), y (class label)
    Output: score vector (2D) = ∇_x log p(x|y,t)
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Time embedding
        self.t_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        # Class embedding
        self.y_emb = nn.Embedding(2, 64)

        # Main network
        self.net = nn.Sequential(
            nn.Linear(2 + 64 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t, y):
        # Embed time and class
        t_vec = self.t_emb(t.unsqueeze(1))
        y_vec = self.y_emb(y)

        # Concatenate and process
        inp = torch.cat([x, t_vec, y_vec], dim=1)
        return self.net(inp)


# ==========================================
# 3. DIFFUSION TRAINING
# ==========================================

def get_diffusion_loss(model, x, y, sigma_max=5.0):
    """
    Denoising Score Matching loss.
    Train the model to predict the score at various noise levels.
    """
    batch_size = x.size(0)

    # Sample noise levels uniformly
    t = torch.rand(batch_size, device=x.device)

    # Sample noise
    noise = torch.randn_like(x)

    # Noise scale (variance exploding schedule)
    sigma = t.unsqueeze(1) * sigma_max

    # Perturb data
    x_noisy = x + noise * sigma

    # Predict score: the model should output -noise/sigma
    score_pred = model(x_noisy, t, y)
    target = -noise / (sigma + 1e-8)

    # Weighted loss (weight by sigma^2 for stability)
    loss = torch.mean(torch.sum((score_pred - target) ** 2, dim=1) * (sigma.squeeze() ** 2 + 1e-8))

    return loss


def train_diffusion(model, X, Y, n_iters=3000, lr=1e-3):
    """Train the diffusion model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    print("Training diffusion model...")
    for i in range(n_iters):
        # Random batch
        idx = torch.randint(0, len(X), (256,))
        x_batch = X[idx]
        y_batch = Y[idx]

        optimizer.zero_grad()
        loss = get_diffusion_loss(model, x_batch, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 500 == 0:
            print(f"  Iter {i}: Loss = {loss.item():.4f}")

    return losses


# ==========================================
# 4. SAMPLING (Reverse Diffusion)
# ==========================================

def sample_diffusion(model, n_samples, target_class, n_steps=100, sigma_max=5.0):
    """
    Generate samples via reverse diffusion (Langevin dynamics).
    """
    model.eval()

    # Start from noise
    x = torch.randn(n_samples, 2) * sigma_max
    y = torch.ones(n_samples, dtype=torch.long) * target_class

    # Time steps (from 1 to 0)
    dt = 1.0 / n_steps

    with torch.no_grad():
        for step in range(n_steps):
            t = 1.0 - step * dt
            t_tensor = torch.ones(n_samples) * t
            sigma = t * sigma_max

            # Get score
            score = model(x, t_tensor, y)

            # Reverse SDE step (Euler-Maruyama)
            # dx = sigma^2 * score * dt + sigma * sqrt(dt) * noise
            noise = torch.randn_like(x) if step < n_steps - 1 else 0
            x = x + (sigma ** 2) * score * dt + sigma * np.sqrt(dt) * noise * 0.5

    return x.numpy()


# ==========================================
# 5. THERMODYNAMIC ANALYSIS
# ==========================================

def calculate_energy_landscape(model, grid_size=60, t_probe=0.1):
    """
    Map the energy landscape at a given noise level.

    We compute:
    1. Score magnitude for each class (how strongly does the model "pull" toward that class?)
    2. Score difference (the "barrier" between classes)
    """
    model.eval()

    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    t = torch.ones(points.size(0)) * t_probe

    with torch.no_grad():
        # Score for Class 0 (Ring)
        s0 = model(points, t, torch.zeros(points.size(0), dtype=torch.long))
        # Score for Class 1 (Blob)
        s1 = model(points, t, torch.ones(points.size(0), dtype=torch.long))

        # Score magnitudes (how strongly does the model push here?)
        mag0 = torch.norm(s0, dim=1).reshape(grid_size, grid_size)
        mag1 = torch.norm(s1, dim=1).reshape(grid_size, grid_size)

        # Score difference (conceptual separation)
        diff = torch.norm(s0 - s1, dim=1).reshape(grid_size, grid_size)

        # "Confusion" metric: where do both classes have similar scores?
        confusion = 1.0 / (diff + 0.1)

    return {
        'xx': xx,
        'yy': yy,
        'score_mag_ring': mag0.numpy(),
        'score_mag_blob': mag1.numpy(),
        'separation': diff.numpy(),
        'confusion': confusion.numpy()
    }


def calculate_bits_of_separation(model, n_points=1000):
    """
    Calculate the average "bits of separation" between concepts.
    This is the log of the score difference integrated over the data manifold.
    """
    model.eval()

    # Sample points along the boundary region
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = 1.0  # Radius where ring and blob meet
    points = torch.FloatTensor(np.c_[r * np.cos(theta), r * np.sin(theta)])
    t = torch.ones(n_points) * 0.1  # Low noise

    with torch.no_grad():
        s0 = model(points, t, torch.zeros(n_points, dtype=torch.long))
        s1 = model(points, t, torch.ones(n_points, dtype=torch.long))

        diff = torch.norm(s0 - s1, dim=1)
        mean_diff = diff.mean().item()

        # Convert to "bits" (log2 scale)
        bits = np.log2(mean_diff + 1)

    return bits, mean_diff


# ==========================================
# 6. VISUALIZATION
# ==========================================

def plot_results(X, Y, samples, landscape, losses, save_dir):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # A. Training Data
    ax = axes[0, 0]
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=5, alpha=0.5, c='#e74c3c', label='Ring (Class 0)')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=5, alpha=0.5, c='#3498db', label='Blob (Class 1)')
    ax.set_title('Training Data: Two Concepts', fontsize=11)
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # B. Generated Samples
    ax = axes[0, 1]
    ax.scatter(samples[0][:, 0], samples[0][:, 1], s=10, alpha=0.6, c='#e74c3c', label='Generated Ring')
    ax.scatter(samples[1][:, 0], samples[1][:, 1], s=10, alpha=0.6, c='#3498db', label='Generated Blob')
    ax.set_title('Diffusion Generation', fontsize=11)
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # C. Training Loss
    ax = axes[0, 2]
    ax.plot(losses, alpha=0.3)
    ax.plot(np.convolve(losses, np.ones(50) / 50, mode='valid'), 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontsize=11)
    ax.grid(alpha=0.3)

    # D. Energy Landscape - Separation
    ax = axes[1, 0]
    c = ax.contourf(landscape['xx'], landscape['yy'], landscape['separation'],
                    levels=20, cmap='inferno')
    plt.colorbar(c, ax=ax, label='Score Difference')
    ax.set_title('Thermodynamic Barrier\n(Concept Separation)', fontsize=11)
    ax.set_xlabel('High values = Strong boundary')
    ax.set_aspect('equal')

    # E. Confusion Zone
    ax = axes[1, 1]
    c = ax.contourf(landscape['xx'], landscape['yy'], landscape['confusion'],
                    levels=20, cmap='coolwarm')
    plt.colorbar(c, ax=ax, label='Confusion (1/separation)')
    ax.set_title('Concept Bleed Zone\n(Where classes mix)', fontsize=11)
    ax.set_aspect('equal')

    # F. Score Field Visualization
    ax = axes[1, 2]
    # Downsample for quiver plot
    skip = 4
    xx_sub = landscape['xx'][::skip, ::skip]
    yy_sub = landscape['yy'][::skip, ::skip]

    # Show the "pull" direction for each class at selected points
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=2, alpha=0.2, c='#e74c3c')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=2, alpha=0.2, c='#3498db')

    # Mark high-separation boundary
    ax.contour(landscape['xx'], landscape['yy'], landscape['separation'],
               levels=[np.percentile(landscape['separation'], 75)],
               colors='black', linewidths=2, linestyles='--')

    ax.set_title('Concept Boundary\n(75th percentile separation)', fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_dir / 'diffusion_thermodynamics.png', dpi=150)
    plt.savefig(save_dir / 'diffusion_thermodynamics.pdf')
    print(f"Saved: {save_dir / 'diffusion_thermodynamics.png'}")
    plt.close()


def create_summary_figure(X, Y, samples, landscape, bits, save_dir):
    """Create paper-ready summary figure."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # A. Data and Generation
    ax = axes[0]
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=3, alpha=0.3, c='#e74c3c', label='Ring (data)')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=3, alpha=0.3, c='#3498db', label='Blob (data)')
    ax.scatter(samples[0][:, 0], samples[0][:, 1], s=20, alpha=0.8, c='#c0392b',
               marker='x', label='Ring (gen)')
    ax.scatter(samples[1][:, 0], samples[1][:, 1], s=20, alpha=0.8, c='#2980b9',
               marker='x', label='Blob (gen)')
    ax.set_title('Conditional Diffusion: Two Concepts', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # B. Thermodynamic Barrier
    ax = axes[1]
    c = ax.contourf(landscape['xx'], landscape['yy'], landscape['separation'],
                    levels=20, cmap='inferno')
    plt.colorbar(c, ax=ax, label='Separation Force')
    ax.contour(landscape['xx'], landscape['yy'], landscape['separation'],
               levels=[np.percentile(landscape['separation'], 70)],
               colors='white', linewidths=2, linestyles='--')
    ax.set_title(f'Energy Barrier Between Concepts\n(Bits of Separation: {bits:.2f})', fontsize=11)
    ax.set_aspect('equal')

    # C. Interpretation
    ax = axes[2]
    ax.axis('off')

    text = f"""
    THERMODYNAMIC DIFFUSION ANALYSIS

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Bits of Separation: {bits:.2f}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    INTERPRETATION:

    The diffusion model learns an
    energy landscape where:

    • High barrier (bright) =
      Strong concept distinction
      Model confidently separates
      Ring from Blob

    • Low barrier (dark) =
      Concept bleed zone
      Model may confuse classes

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IMPLICATION:

    This metric quantifies how
    "entangled" concepts are in
    the learned representation.
    """

    ax.text(0.1, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / 'diffusion_summary.png', dpi=150)
    plt.savefig(save_dir / 'diffusion_summary.pdf')
    print(f"Saved: {save_dir / 'diffusion_summary.png'}")
    plt.close()


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Thermodynamic Illumination: Frontier A")
    print("The Energy Barrier of Concepts in Diffusion Models")
    print("=" * 60)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    # 1. Generate data
    print("\nGenerating training data...")
    X, Y = get_data(n_samples=3000)
    print(f"  Ring samples: {(Y == 0).sum().item()}")
    print(f"  Blob samples: {(Y == 1).sum().item()}")

    # 2. Train diffusion model
    model = ConditionalScoreNet()
    losses = train_diffusion(model, X, Y, n_iters=3000)

    # 3. Generate samples
    print("\nGenerating samples via reverse diffusion...")
    samples = {
        0: sample_diffusion(model, n_samples=300, target_class=0),
        1: sample_diffusion(model, n_samples=300, target_class=1)
    }

    # 4. Thermodynamic analysis
    print("\nComputing thermodynamic landscape...")
    landscape = calculate_energy_landscape(model)
    bits, mean_sep = calculate_bits_of_separation(model)

    # 5. Visualize
    print("\nGenerating visualizations...")
    plot_results(X.numpy(), Y.numpy(), samples, landscape, losses, out_dir)
    create_summary_figure(X.numpy(), Y.numpy(), samples, landscape, bits, out_dir)

    # 6. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nBits of Separation: {bits:.3f}")
    print(f"Mean Score Difference: {mean_sep:.3f}")

    # Analyze the landscape
    separation = landscape['separation']
    print(f"\nLandscape Statistics:")
    print(f"  Min separation: {separation.min():.3f}")
    print(f"  Max separation: {separation.max():.3f}")
    print(f"  Mean separation: {separation.mean():.3f}")

    # Find high-confusion zones
    confusion = landscape['confusion']
    high_conf_mask = confusion > np.percentile(confusion, 90)
    high_conf_area = high_conf_mask.sum() / confusion.size * 100

    print(f"\nHigh-confusion area: {high_conf_area:.1f}% of space")

    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)

    if bits > 2.0:
        print(f"\nHIGH SEPARATION ({bits:.2f} bits)")
        print("  The model has learned distinct concept boundaries.")
        print("  Low risk of concept bleed / hallucination.")
    elif bits > 1.0:
        print(f"\nMODERATE SEPARATION ({bits:.2f} bits)")
        print("  Concepts are reasonably distinct but some overlap exists.")
        print("  May see occasional mixing at boundaries.")
    else:
        print(f"\nLOW SEPARATION ({bits:.2f} bits)")
        print("  WARNING: High concept entanglement!")
        print("  Model may confuse/hallucinate between classes.")

    print("\nThis framework enables quantifying 'concept bleed'")
    print("in diffusion models before deployment.")
