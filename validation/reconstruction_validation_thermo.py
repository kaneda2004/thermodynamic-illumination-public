#!/usr/bin/env python3
"""
Reconstruction Thermodynamic Validation (Gold Standard).

Compute thermodynamic bits via nested sampling in weight space for generator
analogs of the reconstruction feature extractors, then correlate with per-pixel
reconstruction MSE. Supports threshold sweeps and MNIST-calibrated thresholds.

Usage:
    uv run python validation/reconstruction_validation_thermo.py
    uv run python validation/reconstruction_validation_thermo.py --runs 5 --max-iter 300
    uv run python validation/reconstruction_validation_thermo.py --thresholds 0.05,0.1,0.2
    uv run python validation/reconstruction_validation_thermo.py --mnist-percentiles 10,50,90
    uv run python validation/reconstruction_validation_thermo.py --mse-json results/reconstruction_mse.json
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.thermo_sampler_v3 import order_multiplicative, order_multiplicative_v2


DEFAULT_MSE = {
    # Per-pixel reconstruction results from validation/reconstruction_validation.py
    "RandomCPPN": 0.0012,
    "DeepCPPN": 0.0013,
    "RandomResNet": 0.0020,
    "RandomConv": 0.0039,
    "RandomMLP": 0.0941,
    "RandomFourier": 0.0968,
}


# =============================================================================
# Generator analogs (weights -> image)
# =============================================================================

class RandomCPPNGen(nn.Module):
    """Shallow CPPN generator (coordinate MLP)."""
    def __init__(self, image_size: int = 28, hidden: int = 128):
        super().__init__()
        self.image_size = image_size
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, image_size),
            torch.linspace(-1, 1, image_size),
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        coords = torch.stack([x, y, r], dim=-1)
        self.register_buffer("coords", coords)

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        x = self.coords.view(-1, 3)
        out = self.net(x)
        return out.view(1, 1, self.image_size, self.image_size)


class DeepCPPNGen(nn.Module):
    """Deeper CPPN generator (more layers, stronger spectral bias)."""
    def __init__(self, image_size: int = 28, hidden: int = 128):
        super().__init__()
        self.image_size = image_size
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, image_size),
            torch.linspace(-1, 1, image_size),
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        coords = torch.stack([x, y, r], dim=-1)
        self.register_buffer("coords", coords)

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.Tanh(),
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        x = self.coords.view(-1, 3)
        out = self.net(x)
        return out.view(1, 1, self.image_size, self.image_size)


class RandomFourierGen(nn.Module):
    """Random Fourier features generator (global mixing, no spatial bias)."""
    def __init__(
        self,
        image_size: int = 28,
        latent_dim: int = 64,
        n_freqs: int = 64,
        proj_scale: float = 3.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.seed = nn.Parameter(torch.randn(1, latent_dim))
        self.proj = nn.Parameter(torch.randn(latent_dim, n_freqs))
        self.proj_scale = proj_scale
        self.phases = nn.Parameter(torch.rand(n_freqs) * 2 * math.pi)
        self.to_pixels = nn.Linear(n_freqs * 2, image_size * image_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self) -> torch.Tensor:
        proj = self.seed @ (self.proj * self.proj_scale)
        feats = torch.cat(
            [torch.sin(proj + self.phases), torch.cos(proj + self.phases)],
            dim=-1,
        )
        out = self.sigmoid(self.to_pixels(feats))
        return out.view(1, 1, self.image_size, self.image_size)


class RandomMLPGen(nn.Module):
    """Random MLP generator (no spatial bias)."""
    def __init__(self, image_size: int = 28, latent_dim: int = 64):
        super().__init__()
        self.image_size = image_size
        self.seed = nn.Parameter(torch.randn(1, latent_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        out = self.net(self.seed)
        return out.view(1, 1, self.image_size, self.image_size)


class RandomConvGen(nn.Module):
    """Shallow Conv generator (local bias)."""
    def __init__(self, image_size: int = 28, base_channels: int = 64):
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4 for RandomConvGen")
        base = image_size // 4
        self.seed = nn.Parameter(torch.randn(1, base_channels, base, base))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 4, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        return self.net(self.seed)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = self.conv2(h)
        return torch.relu(h + x)


class RandomResNetGen(nn.Module):
    """ResNet-style generator with skip connections."""
    def __init__(self, image_size: int = 28, base_channels: int = 64):
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4 for RandomResNetGen")
        base = image_size // 4
        self.seed = nn.Parameter(torch.randn(1, base_channels, base, base))
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block1 = ResidualBlock(base_channels)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block2 = ResidualBlock(base_channels)
        self.to_img = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self) -> torch.Tensor:
        x = self.up1(self.seed)
        x = self.block1(x)
        x = self.up2(x)
        x = self.block2(x)
        return self.to_img(x)


# =============================================================================
# Order metrics
# =============================================================================

def _squeeze_image(img_tensor: torch.Tensor) -> torch.Tensor:
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() == 3 and img_tensor.size(0) == 1:
        img_tensor = img_tensor.squeeze(0)
    return img_tensor


def order_jpeg_tv(img_tensor: torch.Tensor) -> float:
    img_tensor = _squeeze_image(img_tensor)
    img_np = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(img_np).save(buffer, format="JPEG", quality=85)
    ratio = len(buffer.getvalue()) / max(1, img_np.nbytes)
    compress_score = max(0.0, 1.0 - ratio)

    img_t = img_tensor.float()
    if img_t.dim() == 2:
        tv_h = torch.mean(torch.abs(img_t[1:, :] - img_t[:-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, 1:] - img_t[:, :-1]))
    else:
        tv_h = torch.mean(torch.abs(img_t[:, 1:, :] - img_t[:, :-1, :]))
        tv_w = torch.mean(torch.abs(img_t[:, :, 1:] - img_t[:, :, :-1]))
    tv_score = math.exp(-10 * (tv_h + tv_w).item())

    return compress_score * tv_score


def tensor_to_binary(img_tensor: torch.Tensor, threshold: float) -> np.ndarray:
    img_tensor = _squeeze_image(img_tensor)
    img_np = img_tensor.detach().cpu().numpy()
    return (img_np >= threshold).astype(np.uint8)


def order_multiplicative_tensor(img_tensor: torch.Tensor, threshold: float) -> float:
    img_bin = tensor_to_binary(img_tensor, threshold)
    return float(order_multiplicative(img_bin))


def order_multiplicative_v2_tensor(
    img_tensor: torch.Tensor, threshold: float, resolution_ref: int
) -> float:
    img_bin = tensor_to_binary(img_tensor, threshold)
    return float(order_multiplicative_v2(img_bin, resolution_ref=resolution_ref))


def get_order_fn(metric: str, binarize_threshold: float, resolution_ref: int):
    if metric == "jpeg_tv":
        return order_jpeg_tv
    if metric == "multiplicative":
        return lambda img: order_multiplicative_tensor(img, binarize_threshold)
    if metric == "multiplicative_v2":
        return lambda img: order_multiplicative_v2_tensor(img, binarize_threshold, resolution_ref)
    raise ValueError(f"Unknown order metric: {metric}")


# =============================================================================
# Threshold helpers
# =============================================================================

def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_percentiles(percentiles: list[float]) -> list[float]:
    if percentiles and all(0 < p <= 1 for p in percentiles):
        return [p * 100 for p in percentiles]
    return percentiles


def compute_mnist_thresholds(
    order_fn,
    percentiles: list[float],
    n_samples: int,
    seed: int,
    image_size: int,
    data_root: str,
    download: bool,
) -> list[float]:
    rng = np.random.default_rng(seed)
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(data_root, train=True, download=download, transform=transform)
    if n_samples > len(dataset):
        n_samples = len(dataset)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    values = []
    for idx in indices:
        img, _ = dataset[idx]
        img = img.unsqueeze(0)  # (1, 1, H, W)
        if img.shape[-1] != image_size:
            img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
        values.append(order_fn(img))

    values = np.array(values)
    thresholds = np.percentile(values, percentiles).tolist()
    return thresholds


# =============================================================================
# Simplified nested sampling in weight space
# =============================================================================

def get_weights_vec(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters())


def set_weights_vec(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def elliptical_slice_sampling(
    model: nn.Module,
    current_weights: torch.Tensor,
    threshold: float,
    rng: np.random.Generator,
    order_fn,
    max_attempts: int = 30,
) -> tuple[torch.Tensor, float, bool]:
    nu = torch.randn_like(current_weights)
    theta = rng.uniform(0, 2 * math.pi)
    theta_min, theta_max = theta - 2 * math.pi, theta

    for _ in range(max_attempts):
        new_weights = current_weights * math.cos(theta) + nu * math.sin(theta)
        set_weights_vec(model, new_weights)
        with torch.no_grad():
            score = order_fn(model())
        if score > threshold:
            return new_weights, score, True
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)

    return current_weights, threshold, False


def nested_bits_multi_threshold(
    model_class: type[nn.Module],
    image_size: int,
    n_live: int,
    max_iter: int,
    thresholds: list[float],
    seed: int,
    device: torch.device,
    order_fn,
    model_kwargs: dict | None = None,
) -> tuple[dict[float, float], dict[float, bool], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model_kwargs = model_kwargs or {}
    model = model_class(image_size=image_size, **model_kwargs).to(device)
    model.eval()

    param_count = len(get_weights_vec(model))
    live = []
    for _ in range(n_live):
        w = torch.randn(param_count, device=device)
        set_weights_vec(model, w)
        with torch.no_grad():
            score = order_fn(model())
        live.append({"w": w, "score": score})
    live.sort(key=lambda x: x["score"])

    thresholds_sorted = sorted(thresholds)
    bits_map = {t: 0.0 for t in thresholds_sorted}
    reached_map = {t: False for t in thresholds_sorted}
    next_idx = 0

    for iteration in range(max_iter):
        dead = live.pop(0)
        bits = (iteration + 1) / (n_live * math.log(2))
        while next_idx < len(thresholds_sorted) and dead["score"] >= thresholds_sorted[next_idx]:
            tau = thresholds_sorted[next_idx]
            bits_map[tau] = bits
            reached_map[tau] = True
            next_idx += 1

        donor = live[rng.integers(0, len(live))]
        new_w, new_score, _ = elliptical_slice_sampling(
            model, donor["w"], dead["score"], rng=rng, order_fn=order_fn
        )
        live.append({"w": new_w, "score": new_score})
        live.sort(key=lambda x: x["score"])

    max_bits = max_iter / (n_live * math.log(2))
    for tau in thresholds_sorted:
        if not reached_map[tau]:
            bits_map[tau] = max_bits
    return bits_map, reached_map, param_count


# =============================================================================
# Main
# =============================================================================

def get_architectures(args):
    return [
        ("DeepCPPN", DeepCPPNGen, {}),
        ("RandomCPPN", RandomCPPNGen, {}),
        ("RandomResNet", RandomResNetGen, {}),
        ("RandomConv", RandomConvGen, {}),
        ("RandomFourier", RandomFourierGen, {
            "latent_dim": args.fourier_latent_dim,
            "n_freqs": args.fourier_n_freqs,
            "proj_scale": args.fourier_proj_scale,
        }),
        ("RandomMLP", RandomMLPGen, {}),
    ]


def load_mse(path: str | None) -> dict[str, float]:
    if not path:
        return dict(DEFAULT_MSE)
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "mse" in data:
        return data["mse"]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermodynamic bits vs reconstruction MSE")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--mnist-percentiles", type=str, default=None)
    parser.add_argument("--mnist-samples", type=int, default=1000)
    parser.add_argument("--mnist-seed", type=int, default=123)
    parser.add_argument("--mnist-root", type=str, default="./data")
    parser.add_argument("--mnist-download", action="store_true")
    parser.add_argument("--order-metric", type=str, default="jpeg_tv",
                        choices=["jpeg_tv", "multiplicative", "multiplicative_v2"])
    parser.add_argument("--binarize-threshold", type=float, default=0.5)
    parser.add_argument("--resolution-ref", type=int, default=32)
    parser.add_argument("--n-live", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fourier-latent-dim", type=int, default=64)
    parser.add_argument("--fourier-n-freqs", type=int, default=64)
    parser.add_argument("--fourier-proj-scale", type=float, default=3.0)
    parser.add_argument("--mse-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    mse_map = load_mse(args.mse_json)
    order_fn = get_order_fn(args.order_metric, args.binarize_threshold, args.resolution_ref)

    if args.thresholds and args.mnist_percentiles:
        raise ValueError("Use either --thresholds or --mnist-percentiles, not both.")

    thresholds = None
    mnist_percentiles = None
    if args.thresholds:
        thresholds = parse_float_list(args.thresholds)
    elif args.mnist_percentiles:
        mnist_percentiles = normalize_percentiles(parse_float_list(args.mnist_percentiles))
        thresholds = compute_mnist_thresholds(
            order_fn,
            mnist_percentiles,
            n_samples=args.mnist_samples,
            seed=args.mnist_seed,
            image_size=args.image_size,
            data_root=args.mnist_root,
            download=args.mnist_download,
        )
    else:
        thresholds = [args.threshold]

    if not thresholds:
        raise ValueError("No thresholds provided.")

    print("=" * 70)
    print("RECONSTRUCTION VALIDATION (THERMODYNAMIC BITS)")
    print("=" * 70)
    print(f"image_size={args.image_size} order_metric={args.order_metric}")
    if args.order_metric != "jpeg_tv":
        print(f"binarize_threshold={args.binarize_threshold} resolution_ref={args.resolution_ref}")
    if mnist_percentiles is not None:
        thresholds_str = ", ".join(f"{t:.4f}" for t in thresholds)
        percentiles_str = ", ".join(f"{p:.1f}" for p in mnist_percentiles)
        print(f"mnist_percentiles=[{percentiles_str}] -> thresholds=[{thresholds_str}]")
    else:
        thresholds_str = ", ".join(f"{t:.4f}" for t in thresholds)
        print(f"thresholds=[{thresholds_str}]")
    print(f"n_live={args.n_live} max_iter={args.max_iter} runs={args.runs}")
    print(f"device={device}")

    thresholds_sorted = sorted(set(thresholds))
    results_by_threshold = {t: [] for t in thresholds_sorted}

    def format_run(bits_map, reached_map):
        parts = []
        for tau in thresholds_sorted:
            mark = "" if reached_map[tau] else ">="
            parts.append(f"{tau:.2f}:{mark}{bits_map[tau]:.2f}")
        return " ".join(parts)

    for name, gen_cls, gen_kwargs in get_architectures(args):
        per_tau_bits = {t: [] for t in thresholds_sorted}
        per_tau_reached = {t: [] for t in thresholds_sorted}
        param_count = 0
        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            bits_map, reached_map, param_count = nested_bits_multi_threshold(
                gen_cls,
                image_size=args.image_size,
                n_live=args.n_live,
                max_iter=args.max_iter,
                thresholds=thresholds_sorted,
                seed=seed,
                device=device,
                order_fn=order_fn,
                model_kwargs=gen_kwargs,
            )
            for tau in thresholds_sorted:
                per_tau_bits[tau].append(bits_map[tau])
                per_tau_reached[tau].append(reached_map[tau])
            print(f"  {name:<14} run {run_idx+1}: {format_run(bits_map, reached_map)}")

        for tau in thresholds_sorted:
            bits_mean = float(np.mean(per_tau_bits[tau]))
            bits_std = float(np.std(per_tau_bits[tau]))
            reached_count = int(sum(per_tau_reached[tau]))
            results_by_threshold[tau].append({
                "name": name,
                "bits_mean": bits_mean,
                "bits_std": bits_std,
                "reached_count": reached_count,
                "runs": args.runs,
                "param_count": param_count,
                "mse": mse_map.get(name),
            })

    for tau in thresholds_sorted:
        print("\n" + "=" * 70)
        print(f"SUMMARY (tau={tau:.4f})")
        print("=" * 70)
        print(f"{'Architecture':<15} {'Bits':<12} {'MSE':<10} {'Status'}")
        print("-" * 70)
        lower_bound = False
        for r in results_by_threshold[tau]:
            status = "reached" if r["reached_count"] == r["runs"] else f">= ( {r['reached_count']}/{r['runs']} )"
            if r["reached_count"] < r["runs"]:
                lower_bound = True
            mse_val = r["mse"]
            mse_str = f"{mse_val:.4f}" if mse_val is not None else "n/a"
            bits_str = f"{r['bits_mean']:.2f}±{r['bits_std']:.2f}"
            print(f"{r['name']:<15} {bits_str:<12} {mse_str:<10} {status}")

        paired = [(r["bits_mean"], r["mse"]) for r in results_by_threshold[tau] if r["mse"] is not None]
        if len(paired) >= 2:
            bits_vals = [p[0] for p in paired]
            mse_vals = [p[1] for p in paired]
            corr, pval = spearmanr(bits_vals, mse_vals)
            print("\nSpearman correlation (bits vs MSE): "
                  f"{corr:.3f} (p={pval:.3f})")
            if lower_bound:
                print("Note: at least one architecture did not reach threshold; bits are lower bounds.")
        else:
            print("\nNot enough MSE values to compute correlation.")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "image_size": args.image_size,
                "thresholds": thresholds_sorted,
                "mnist_percentiles": mnist_percentiles,
                "mnist_samples": args.mnist_samples if mnist_percentiles is not None else None,
                "mnist_seed": args.mnist_seed if mnist_percentiles is not None else None,
                "mnist_root": args.mnist_root if mnist_percentiles is not None else None,
                "order_metric": args.order_metric,
                "binarize_threshold": args.binarize_threshold,
                "resolution_ref": args.resolution_ref,
                "n_live": args.n_live,
                "max_iter": args.max_iter,
                "runs": args.runs,
                "seed": args.seed,
                "device": str(device),
                "fourier_latent_dim": args.fourier_latent_dim,
                "fourier_n_freqs": args.fourier_n_freqs,
                "fourier_proj_scale": args.fourier_proj_scale,
            },
            "results": {str(t): results_by_threshold[t] for t in thresholds_sorted},
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
