#!/usr/bin/env python3
"""
RES-219: Weight Space Collapse is General, Not CPPN-Specific

Tests whether weight space effective dimensionality drops at order thresholds in other
architectures (ResNet, MLP). If collapse is general, it validates that this is a fundamental
property of expressive networks under order constraints, not a CPPN quirk.

Hypothesis: Both ResNet and MLP architectures will show effective dimensionality drops
≥2.5x when reconstruction order crosses 0.5, mirroring the CPPN collapse pattern.

Simplified approach:
- Train 5 instances each (vs 30) to save time
- Use 5 epochs with snapshots at: 1, 2, 3, 4, 5
- Minimal architectures to enable quick training
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.decomposition import PCA

# Ensure we're in project root
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from research_system.log_manager import ResearchLogManager


class TinyResNet(nn.Module):
    """Minimal ResNet for 32x32 (2 blocks, ~8k params)."""

    def __init__(self, channels=1):
        super().__init__()
        self.initial = nn.Conv2d(channels, 16, 3, padding=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1)
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial(x)
        identity = x
        x = self.block1(x)
        x = x + identity
        identity = x
        x = self.block2(x)
        x = x + identity
        x = self.final(x)
        return x

    def get_weights(self):
        """Flatten all weights into a single vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)


class TinyMLP(nn.Module):
    """Minimal MLP for 32x32 (~8k params)."""

    def __init__(self, channels=1):
        super().__init__()
        input_size = 32 * 32 * channels
        output_size = 32 * 32 * channels

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        x = x.view(batch_size, 1, 32, 32)
        return x

    def get_weights(self):
        """Flatten all weights into a single vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)


def compute_order_metric(outputs):
    """
    Compute order metric from network outputs.
    Order ~ spatial coherence (inverse of edge density).
    """
    batch_np = outputs.detach().cpu().numpy()
    orders = []

    for img in batch_np:
        img_flat = img.flatten()
        # Edge density proxy: percentage of pixels near middle gray (0.5)
        near_middle = np.sum(np.abs(img_flat - 0.5) < 0.1) / len(img_flat)
        # Order inversely proportional to edge density
        order = near_middle / (1.0 + 0.5 * (1 - near_middle))
        orders.append(order)

    return np.mean(orders)


def compute_effective_dimension(weights):
    """Compute effective dimensionality using PCA."""
    if len(weights.shape) == 1:
        weights = weights.reshape(1, -1)

    # Handle case with very few samples
    if weights.shape[0] == 1:
        return {
            'n_components_90': 1,
            'first_pc_var': 1.0,
            'effective_dim': 1.0,
            'eigenvalue_ratio': 1.0,
        }

    pca = PCA()
    pca.fit(weights)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumsum >= 0.90) + 1 if np.any(cumsum >= 0.90) else len(cumsum)

    var = pca.explained_variance_
    var_norm = var / var.sum()
    entropy = -np.sum(var_norm * np.log(var_norm + 1e-10))

    return {
        'n_components_90': float(n_components_90),
        'first_pc_var': float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else 0,
        'effective_dim': float(entropy / np.log(len(var))),
        'eigenvalue_ratio': float(var[0] / var[-1]) if len(var) > 1 else 1.0,
    }


def train_architecture(model_class, name, n_models=5, n_epochs=5, device='cpu'):
    """Train multiple instances, measure weight space collapse."""

    results = {
        'architecture': name,
        'n_models': n_models,
        'n_epochs': n_epochs,
        'weight_snapshots': {},
        'order_snapshots': {},
        'dimensionality': {},
    }

    # Simple training data
    np.random.seed(42)
    train_data = torch.randn(50, 1, 32, 32).float().to(device)
    train_data = torch.clamp(train_data * 0.2 + 0.5, 0, 1)

    for model_idx in range(n_models):
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_data)
            loss.backward()
            optimizer.step()

            # Record at every epoch
            if epoch not in results['weight_snapshots']:
                results['weight_snapshots'][epoch] = []
                results['order_snapshots'][epoch] = []

            weights = model.get_weights()
            order = compute_order_metric(outputs)

            results['weight_snapshots'][epoch].append(weights)
            results['order_snapshots'][epoch].append(order)

        print(f"  {name} model {model_idx + 1}/{n_models} trained")

    # Analyze dimensionality
    for epoch in range(1, n_epochs + 1):
        weights_list = results['weight_snapshots'][epoch]
        orders_list = results['order_snapshots'][epoch]

        weights_matrix = np.array(weights_list)
        dim_stats = compute_effective_dimension(weights_matrix)

        results['dimensionality'][epoch] = {
            'mean_order': float(np.mean(orders_list)),
            'std_order': float(np.std(orders_list)),
            'effective_dim': dim_stats['effective_dim'],
            'n_components_90': dim_stats['n_components_90'],
            'first_pc_var': dim_stats['first_pc_var'],
            'eigenvalue_ratio': dim_stats['eigenvalue_ratio'],
        }

    return results


def analyze_collapse(all_results):
    """Analyze if architectures show collapse pattern."""

    analysis = {}

    for arch_name, results in all_results.items():
        dims = results['dimensionality']
        epochs = sorted([int(e) for e in dims.keys()])

        if len(epochs) < 2:
            continue

        # Compare first and last epoch
        first_epoch = epochs[0]
        last_epoch = epochs[-1]

        initial_dim = dims[first_epoch]['effective_dim']
        final_dim = dims[last_epoch]['effective_dim']
        ratio = initial_dim / (final_dim + 1e-8)

        analysis[arch_name] = {
            'initial_eff_dim': initial_dim,
            'final_eff_dim': final_dim,
            'collapse_ratio': ratio,
            'initial_order': dims[first_epoch]['mean_order'],
            'final_order': dims[last_epoch]['mean_order'],
            'shows_collapse': ratio >= 2.5,
        }

    return analysis


def main():
    print("=" * 80)
    print("RES-219: Weight Space Collapse Across Architectures")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Train models
    print("Training ResNet...")
    resnet_results = train_architecture(TinyResNet, 'ResNet', n_models=5, n_epochs=5, device=device)

    print("Training MLP...")
    mlp_results = train_architecture(TinyMLP, 'MLP', n_models=5, n_epochs=5, device=device)

    # CPPN baseline from RES-218
    cppn_results = {
        'architecture': 'CPPN',
        'dimensionality': {
            1: {'mean_order': 0.005, 'effective_dim': 4.12},
            100: {'mean_order': 0.465, 'effective_dim': 1.45}
        }
    }

    # Analyze
    all_arch_results = {
        'ResNet': resnet_results,
        'MLP': mlp_results,
        'CPPN': cppn_results
    }

    collapse_analysis = analyze_collapse(all_arch_results)

    # Results
    all_show = (
        collapse_analysis.get('ResNet', {}).get('shows_collapse', False) and
        collapse_analysis.get('MLP', {}).get('shows_collapse', False)
    )

    final_results = {
        'method': 'Effective dimensionality measurement across architectures',
        'hypothesis': 'Weight space collapse (eff_dim drop ≥2.5x) is general',
        'architectures': ['CPPN', 'ResNet', 'MLP'],
        'test_threshold': 2.5,
        'cppn_collapse': {
            'initial_dim': 4.12,
            'final_dim': 1.45,
            'ratio': 2.84
        },
        'resnet_collapse': {
            'initial_dim': collapse_analysis['ResNet']['initial_eff_dim'],
            'final_dim': collapse_analysis['ResNet']['final_eff_dim'],
            'ratio': collapse_analysis['ResNet']['collapse_ratio']
        },
        'mlp_collapse': {
            'initial_dim': collapse_analysis['MLP']['initial_eff_dim'],
            'final_dim': collapse_analysis['MLP']['final_eff_dim'],
            'ratio': collapse_analysis['MLP']['collapse_ratio']
        },
        'all_show_collapse': all_show,
        'conclusion': 'validate' if all_show else 'refute'
    }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for arch, stats in collapse_analysis.items():
        print(f"\n{arch}:")
        print(f"  Initial eff_dim: {stats['initial_eff_dim']:.3f}")
        print(f"  Final eff_dim: {stats['final_eff_dim']:.3f}")
        print(f"  Collapse ratio: {stats['collapse_ratio']:.3f}x")
        print(f"  Shows collapse (≥2.5x): {stats['shows_collapse']}")

    print(f"\nAll show collapse: {all_show}")
    print(f"Conclusion: {final_results['conclusion'].upper()}")

    # Save results
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/cross_architecture_mechanism', exist_ok=True)
    results_path = '/Users/matt/Development/monochrome_noise_converger/results/cross_architecture_mechanism/res_219_results.json'

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Update log
    print("\nUpdating research log...")
    log_manager = ResearchLogManager()

    summary = f"Weight space collapse is {'GENERAL' if all_show else 'NOT GENERAL'} across architectures. "
    summary += f"ResNet {collapse_analysis['ResNet']['collapse_ratio']:.2f}x, "
    summary += f"MLP {collapse_analysis['MLP']['collapse_ratio']:.2f}x, "
    summary += f"CPPN {final_results['cppn_collapse']['ratio']:.2f}x. "

    if all_show:
        summary += "Validates weight space collapse as fundamental to expressive networks."
    else:
        summary += "Collapse pattern appears architecture-specific."

    result_dict = {
        'conclusion': final_results['conclusion'],
        'summary': summary,
        'metrics': {
            'cppn_ratio': final_results['cppn_collapse']['ratio'],
            'resnet_ratio': collapse_analysis['ResNet']['collapse_ratio'],
            'mlp_ratio': collapse_analysis['MLP']['collapse_ratio'],
            'effect_size': collapse_analysis['ResNet']['collapse_ratio']
        }
    }

    log_manager.complete_experiment(
        entry_id='RES-219',
        status='validate' if all_show else 'refute',
        result=result_dict,
        results_file='/Users/matt/Development/monochrome_noise_converger/results/cross_architecture_mechanism/res_219_results.json'
    )

    print("Research log updated!")
    return final_results


if __name__ == '__main__':
    results = main()
    print("\n✓ Complete")
