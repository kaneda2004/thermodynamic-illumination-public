#!/usr/bin/env python3
"""
Thermodynamic Illumination: Ground Truth Calibration
Where do real images fall on the Structure Spectrum?

This validates that the metric isn't arbitrary and reveals the
"optimal bias" point for natural images.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import io
import math
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import json

def get_structure_score(img_tensor):
    """Same metric used throughout the paper"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_np = (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format='JPEG', quality=85)
    ratio = len(buffer.getvalue()) / img_np.nbytes
    comp_score = max(0, 1.0 - ratio)

    tv_h = torch.mean(torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]))
    tv_val = (tv_h + tv_w).item()
    tv_score = math.exp(-10 * tv_val)

    return comp_score * tv_score


def get_dip_target():
    """The synthetic test image used in DIP experiments"""
    img = Image.new('RGB', (64, 64), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 30, 50], fill=(255, 50, 50))
    draw.ellipse([30, 20, 55, 45], fill=(50, 100, 255))
    draw.line([0, 60, 64, 0], fill=(255, 255, 0), width=3)
    draw.rectangle([45, 45, 58, 58], fill=(50, 200, 50))
    return torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0


def measure_cifar(n_samples=100):
    """Measure structure of CIFAR-10 natural images"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    scores = []
    for i in range(n_samples):
        img = dataset[i][0]
        scores.append(get_structure_score(img))

    return scores


def measure_imagenet_style():
    """Measure a few ImageNet-style images if available, or use varied CIFAR classes"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    # Sample from different classes
    class_scores = {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for class_idx in range(10):
        class_images = [i for i, (_, label) in enumerate(dataset) if label == class_idx][:10]
        scores = [get_structure_score(dataset[i][0]) for i in class_images]
        class_scores[class_names[class_idx]] = {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }

    return class_scores


if __name__ == "__main__":
    print("=" * 60)
    print("GROUND TRUTH CALIBRATION")
    print("=" * 60)

    # 1. DIP Synthetic Target
    print("\n1. DIP Synthetic Target (geometric shapes):")
    dip_img = get_dip_target()
    dip_score = get_structure_score(dip_img)
    print(f"   Structure Score: {dip_score:.4f}")

    # 2. CIFAR-10 Natural Images
    print("\n2. Natural Images (CIFAR-10, n=100):")
    cifar_scores = measure_cifar(100)
    cifar_mean = np.mean(cifar_scores)
    cifar_std = np.std(cifar_scores)
    print(f"   Mean: {cifar_mean:.4f} ± {cifar_std:.4f}")
    print(f"   Min:  {min(cifar_scores):.4f}")
    print(f"   Max:  {max(cifar_scores):.4f}")

    # 3. Per-class breakdown
    print("\n3. Per-Class Structure Scores:")
    class_scores = measure_imagenet_style()
    for name, scores in sorted(class_scores.items(), key=lambda x: -x[1]['mean']):
        print(f"   {name:12s}: {scores['mean']:.4f} ± {scores['std']:.4f}")

    # 4. Context with architecture scores
    print("\n" + "=" * 60)
    print("CONTEXTUALIZATION")
    print("=" * 60)

    arch_scores = {
        'Depthwise': 0.94,
        'ResNet-6': 0.92,
        'ResNet-4': 0.89,
        'ResNet-9x9': 0.81,
        'U-Net': 0.79,
        'CPPN': 0.73,
        'Fourier': 0.54,
        'ViT': 0.0001,
        'MLP': 0.0,
    }

    print(f"\nNatural Image Baseline: {cifar_mean:.4f}")
    print(f"DIP Target Baseline:    {dip_score:.4f}")
    print()

    for name, score in sorted(arch_scores.items(), key=lambda x: -x[1]):
        gap = score - cifar_mean
        status = "OVERSMOOTHED" if gap > 0.3 else ("UNDERSMOOTHED" if gap < -0.3 else "MATCHED")
        print(f"   {name:12s}: {score:.4f}  (gap: {gap:+.3f})  [{status}]")

    # 5. Save results
    results = {
        'dip_target': dip_score,
        'cifar_mean': cifar_mean,
        'cifar_std': cifar_std,
        'class_scores': class_scores
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'ground_truth_calibration.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 6. Create calibration figure
    fig_dir = Path(__file__).parent.parent / 'figures'

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot architecture scores as bars
    names = list(arch_scores.keys())
    scores = [arch_scores[n] for n in names]
    colors = ['#2ecc71' if s > cifar_mean else '#e74c3c' for s in scores]

    bars = ax.barh(names, scores, color=colors, alpha=0.7)

    # Add baselines
    ax.axvline(x=cifar_mean, color='blue', linestyle='--', linewidth=2,
               label=f'Natural Images ({cifar_mean:.3f})')
    ax.axvline(x=dip_score, color='orange', linestyle=':', linewidth=2,
               label=f'DIP Target ({dip_score:.3f})')

    # Shade the "matched" zone
    ax.axvspan(cifar_mean - 0.15, cifar_mean + 0.15, alpha=0.1, color='blue',
               label='Optimal Bias Zone')

    ax.set_xlabel('Structure Score', fontsize=12)
    ax.set_title('Ground Truth Calibration: Where Do Real Images Fall?', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(fig_dir / 'ground_truth_calibration.png', dpi=150)
    plt.savefig(fig_dir / 'ground_truth_calibration.pdf', dpi=150)
    plt.close()

    print(f"\nFigure saved to {fig_dir / 'ground_truth_calibration.pdf'}")
    print(f"Results saved to {results_dir / 'ground_truth_calibration.json'}")
