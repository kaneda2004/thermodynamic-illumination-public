#!/usr/bin/env python3
"""
DIP Alignment Validation Experiment
==================================

Reliability-focused rewrite of the alignment sanity check:
- Parallel execution across CPU workers
- Crash-safe JSONL streaming output
- Resume support (skip completed jobs)
- OUTPUT_DIR-aware path handling for GCP runners

Default workload:
- Architectures: CPPN, Fourier, ResNet-6, Depthwise
- Noise levels: 0.05, 0.10, 0.15, 0.20
- Targets: 5 CIFAR-10 images (default: resized to 64x64)
- Seeds: 5 per condition
- Total jobs: 400

Usage:
    uv run python submission_staging/experiments/dip_alignment_validation.py

GCP usage (runner injects OUTPUT_DIR automatically):
    ./run_on_gcp_live.sh \
      -e submission_staging/experiments/dip_alignment_validation.py \
      -r results/dip_alignment_sanity \
      -c 32 -t 21600
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import warnings
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

# Headless backend for remote workers.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Keep worker-level threading under control to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False


def flush_print(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================
# 1. ARCHITECTURES
# ==========================================


class CPPNGen(nn.Module):
    """CPPN: coordinate-based MLP with spectral bias."""

    def __init__(self, img_size: int = 64, hidden: int = 256):
        super().__init__()
        self.img_size = img_size

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, img_size),
                torch.linspace(-1, 1, img_size),
                indexing="ij",
            ),
            dim=-1,
        )

        r = torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2).unsqueeze(-1)
        self.register_buffer("coords", torch.cat([coords, r], dim=-1))

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self):
        x = self.coords.view(-1, 3)
        out = self.net(x)
        return out.view(1, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)


class FourierFeaturesGen(nn.Module):
    """Fourier features with low-frequency bias."""

    def __init__(self, img_size: int = 64, n_freqs: int = 8, hidden: int = 256):
        super().__init__()
        self.img_size = img_size

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, img_size),
                torch.linspace(-1, 1, img_size),
                indexing="ij",
            ),
            dim=-1,
        )
        self.register_buffer("coords", coords)

        freqs = 2 ** torch.linspace(0, n_freqs - 1, n_freqs) * math.pi
        self.register_buffer("freqs", freqs)

        input_dim = 2 + 4 * n_freqs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self):
        coords = self.coords.view(-1, 2)
        features = [coords]
        for freq in self.freqs:
            features.append(torch.sin(freq * coords))
            features.append(torch.cos(freq * coords))

        x = torch.cat(features, dim=-1)
        out = self.net(x)
        return out.view(1, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)


class ResNet6Gen(nn.Module):
    """ResNet-style ConvNet generator with upsampling blocks.

    For img_size=64, this matches the original 6x upsampling design.
    For smaller powers-of-two (e.g. 32), we truncate the block schedule.
    """

    def __init__(self, img_size: int = 64):
        super().__init__()
        if img_size <= 0 or (img_size & (img_size - 1)) != 0:
            raise ValueError(f"img_size must be a power of two, got {img_size}")
        if img_size < 8:
            raise ValueError(f"img_size too small for this generator, got {img_size}")
        self.img_size = int(img_size)
        self.input_shape = (1, 256, 1, 1)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        n_upsamples = int(round(math.log2(self.img_size)))
        schedule = [256, 128, 128, 64, 32, 16]  # out channels per upsampling stage
        if n_upsamples > len(schedule):
            raise ValueError(
                f"img_size={self.img_size} requires {n_upsamples} upsampling stages, "
                f"but only {len(schedule)} are defined"
            )

        layers: List[nn.Module] = []
        in_c = 256
        for out_c in schedule[:n_upsamples]:
            layers.append(block(in_c, out_c))
            in_c = out_c
        layers.append(nn.Conv2d(in_c, 3, 3, padding=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class DepthwiseSepGen(nn.Module):
    """Depthwise-separable convolutional generator.

    For img_size=64, this matches the original 4x upsampling design from a 4x4 seed.
    For smaller powers-of-two (e.g. 32), we truncate the block schedule.
    """

    def __init__(self, img_size: int = 64):
        super().__init__()
        if img_size <= 0 or (img_size & (img_size - 1)) != 0:
            raise ValueError(f"img_size must be a power of two, got {img_size}")
        if img_size % 4 != 0:
            raise ValueError(f"img_size must be divisible by 4, got {img_size}")
        self.img_size = int(img_size)
        self.input_shape = (1, 128, 4, 4)

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c),
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        n_upsamples = int(round(math.log2(self.img_size // 4)))
        schedule = [128, 64, 32, 16]  # out channels per upsampling stage
        if n_upsamples > len(schedule):
            raise ValueError(
                f"img_size={self.img_size} requires {n_upsamples} upsampling stages, "
                f"but only {len(schedule)} are defined"
            )

        layers: List[nn.Module] = []
        in_c = 128
        for out_c in schedule[:n_upsamples]:
            layers.append(block(in_c, out_c))
            in_c = out_c
        layers.append(nn.Conv2d(in_c, 3, 3, padding=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.seed = nn.Parameter(torch.randn(self.input_shape))

    def forward(self):
        return self.net(self.seed)


class ViTGen(nn.Module):
    """Patch-wise Vision Transformer generator.

    This intentionally uses a patch decoder (linear to pixels per patch) rather than a
    convolutional decoder, matching the regime analyzed in the paper.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        if img_size <= 0 or (img_size & (img_size - 1)) != 0:
            raise ValueError(f"img_size must be a power of two, got {img_size}")
        if patch_size <= 0 or (img_size % patch_size) != 0:
            raise ValueError(f"patch_size must divide img_size, got {patch_size} vs {img_size}")

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)

        num_patches = (self.img_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, int(dim)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(heads),
            dim_feedforward=int(dim) * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.to_pixels = nn.Linear(int(dim), 3 * self.patch_size * self.patch_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        x = self.transformer(self.pos_embed)
        x = self.to_pixels(x)

        b, n, p = x.shape
        h = w = int(math.sqrt(n))
        x = x.view(b, h, w, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return self.sigmoid(x)


class MLPGen(nn.Module):
    """Fully-connected generator baseline (no spatial inductive bias)."""

    def __init__(self, img_size: int = 64):
        super().__init__()
        if img_size <= 0 or (img_size & (img_size - 1)) != 0:
            raise ValueError(f"img_size must be a power of two, got {img_size}")
        self.img_size = int(img_size)
        self.latent = nn.Parameter(torch.randn(1, 256))
        self.net = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * self.img_size * self.img_size),
            nn.Sigmoid(),
        )

    def forward(self):
        out = self.net(self.latent)
        return out.view(1, 3, self.img_size, self.img_size)


ARCHITECTURES_DEFAULT = ("CPPN", "Fourier", "ResNet-6", "Depthwise")
ARCH_EXPECTED_STRUCTURE = {
    "CPPN": 0.73,
    "Fourier": 0.54,
    "ResNet-6": 0.92,
    "Depthwise": 0.94,
}


def make_architecture(name: str, img_size: int = 64) -> nn.Module:
    if name == "CPPN":
        return CPPNGen(img_size=img_size)
    if name == "Fourier":
        return FourierFeaturesGen(img_size=img_size)
    if name == "ResNet-6":
        return ResNet6Gen(img_size=img_size)
    if name == "Depthwise":
        return DepthwiseSepGen(img_size=img_size)
    if name == "ViT":
        patch_size = 4 if int(img_size) <= 32 else 8
        return ViTGen(img_size=img_size, patch_size=patch_size)
    if name == "MLP":
        return MLPGen(img_size=img_size)
    raise ValueError(f"Unknown architecture: {name}")


# ==========================================
# 2. TARGET LOADING
# ==========================================


def load_cifar10_targets(
    n_images: int,
    seed: int,
    img_size: int,
    data_root: Path,
    require_dataset: bool,
) -> List[dict]:
    """Load CIFAR-10 images as RGB targets (optional resize, fallback to synthetic)."""
    try:
        from torchvision import datasets, transforms
    except ImportError:
        if require_dataset:
            raise
        flush_print("torchvision unavailable, using synthetic targets")
        return create_synthetic_targets(n_images=n_images, seed=seed, img_size=img_size)

    rs = np.random.RandomState(seed)
    tforms = []
    if int(img_size) != 32:
        tforms.append(transforms.Resize((int(img_size), int(img_size))))
    tforms.append(transforms.ToTensor())
    transform = transforms.Compose(tforms)

    try:
        dataset = datasets.CIFAR10(
            root=str(data_root / "cifar10"),
            train=False,
            download=True,
            transform=transform,
        )
    except Exception as exc:
        if require_dataset:
            raise RuntimeError(f"Failed to load CIFAR-10: {exc}") from exc
        flush_print(f"Failed to load CIFAR-10 ({exc}), using synthetic targets")
        return create_synthetic_targets(n_images=n_images, seed=seed, img_size=img_size)

    n_images = min(int(n_images), int(len(dataset)))
    # Use a permutation prefix so increasing n_images preserves the earlier subset (resume-friendly).
    indices = rs.permutation(len(dataset))[:n_images]
    targets = []
    for idx in indices:
        img, label = dataset[idx]
        targets.append({"name": f"cifar_{idx}", "image": img.unsqueeze(0), "label": int(label)})

    return targets


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        total = r.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        downloaded = 0
        last_print = time.time()
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_int and (time.time() - last_print) > 5:
                pct = 100.0 * downloaded / max(total_int, 1)
                flush_print(f"  download: {downloaded/1e6:.1f}MB / {total_int/1e6:.1f}MB ({pct:.1f}%)")
                last_print = time.time()

    tmp.replace(dest)


def ensure_tiny_imagenet(data_root: Path, require_dataset: bool) -> Path:
    """Ensure Tiny ImageNet is present under data_root, downloading if needed."""
    root = data_root / "tiny-imagenet-200"
    if root.exists():
        return root

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_root / "tiny-imagenet-200.zip"
    try:
        flush_print(f"Downloading Tiny ImageNet from {url} ...")
        _download_file(url, zip_path)
        flush_print("Extracting Tiny ImageNet ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
    except Exception as exc:
        if require_dataset:
            raise RuntimeError(f"Failed to download/extract Tiny ImageNet: {exc}") from exc
        flush_print(f"Failed to download Tiny ImageNet ({exc}), using synthetic targets")
        return Path()

    if not root.exists():
        if require_dataset:
            raise RuntimeError(f"Tiny ImageNet extraction did not create {root}")
        return Path()
    return root


def load_tiny_imagenet_val_targets(
    n_images: int,
    seed: int,
    img_size: int,
    data_root: Path,
    require_dataset: bool,
) -> List[dict]:
    """Load Tiny ImageNet validation images as RGB targets (native 64x64, optional resize)."""
    root = ensure_tiny_imagenet(data_root, require_dataset=require_dataset)
    if not root:
        return create_synthetic_targets(n_images=n_images, seed=seed, img_size=img_size)

    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    if not images_dir.exists() or not ann_path.exists():
        if require_dataset:
            raise RuntimeError("Tiny ImageNet val split not found after extraction")
        return create_synthetic_targets(n_images=n_images, seed=seed, img_size=img_size)

    mapping: Dict[str, str] = {}
    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    filenames = sorted(mapping.keys())
    wnids = sorted(set(mapping.values()))
    wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

    n_images = min(int(n_images), len(filenames))
    # Deterministic permutation prefix so increasing n_images preserves the earlier subset.
    rs = random.Random(seed)
    filenames_shuffled = filenames[:]
    rs.shuffle(filenames_shuffled)
    chosen = filenames_shuffled[:n_images]

    # PIL is a torch/torchvision dependency in our environment; import locally to keep startup cheap.
    from PIL import Image as PILImage

    targets: List[dict] = []
    for fname in chosen:
        img_path = images_dir / fname
        wnid = mapping.get(fname, "")
        label = int(wnid_to_idx.get(wnid, -1))
        img = PILImage.open(img_path).convert("RGB")
        if img.size != (int(img_size), int(img_size)):
            img = img.resize((int(img_size), int(img_size)), resample=PILImage.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        targets.append({"name": f"tinyval_{fname}", "image": tensor, "label": label})

    return targets


def load_targets(
    dataset: str,
    n_images: int,
    seed: int,
    img_size: int,
    data_root: Path,
    require_dataset: bool,
) -> List[dict]:
    dataset = str(dataset).strip().lower()
    if dataset == "cifar10":
        return load_cifar10_targets(
            n_images=n_images,
            seed=seed,
            img_size=img_size,
            data_root=data_root,
            require_dataset=require_dataset,
        )
    if dataset in {"tiny_imagenet", "tinyimagenet"}:
        return load_tiny_imagenet_val_targets(
            n_images=n_images,
            seed=seed,
            img_size=img_size,
            data_root=data_root,
            require_dataset=require_dataset,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def create_synthetic_targets(n_images: int, seed: int, img_size: int) -> List[dict]:
    """Fallback synthetic targets if dataset retrieval is unavailable."""
    from PIL import Image as PILImage

    np.random.seed(seed)
    targets = []
    for i in range(n_images):
        low_res = np.random.rand(8, 8, 3).astype(np.float32)
        img = np.zeros((int(img_size), int(img_size), 3), dtype=np.float32)
        for c in range(3):
            pil_low = PILImage.fromarray((low_res[:, :, c] * 255).astype(np.uint8))
            pil_high = pil_low.resize((int(img_size), int(img_size)), PILImage.BILINEAR)
            img[:, :, c] = np.array(pil_high).astype(np.float32) / 255.0

        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        targets.append({"name": f"synthetic_{i}", "image": tensor, "label": -1})

    return targets


# ==========================================
# 3. METRICS AND TRAINING
# ==========================================


def compute_structure_score(img_tensor: torch.Tensor) -> float:
    """Simplified multiplicative structure proxy in [0, 1]."""
    if img_tensor.dim() == 4:
        img = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    elif img_tensor.dim() == 3:
        img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    else:
        img = img_tensor.detach().cpu().numpy()

    img_gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
    img_bin = (img_gray > 0.5).astype(np.uint8)

    img_flat = img_bin.flatten()
    n_total = len(img_flat)
    n_ones = np.sum(img_flat)
    balance = min(n_ones, n_total - n_ones) / max(n_ones, n_total - n_ones + 1e-6)

    h, w = img_bin.shape
    h_match = np.sum(img_bin[:, :-1] == img_bin[:, 1:]) / ((h * (w - 1)) + 1e-6)
    v_match = np.sum(img_bin[:-1, :] == img_bin[1:, :]) / (((h - 1) * w) + 1e-6)
    coherence = (h_match + v_match) / 2

    return float(np.clip(balance * coherence, 0.0, 1.0))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def train_single_run(
    model: nn.Module,
    target_clean: torch.Tensor,
    target_noisy: torch.Tensor,
    steps: int,
    lr: float,
) -> Tuple[float, int]:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_psnr = -float("inf")
    best_iter = 0

    for i in range(steps):
        optimizer.zero_grad()
        out = model()
        loss = nn.MSELoss()(out, target_noisy)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            score = psnr(out, target_clean)
            if score > best_psnr:
                best_psnr = score
                best_iter = i

    return best_psnr, best_iter


# ==========================================
# 4. PARALLEL JOB EXECUTION
# ==========================================


WORKER_TARGETS: List[np.ndarray] = []


@dataclass(frozen=True)
class Config:
    noise_levels: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    steps: int = 2000
    lr: float = 0.01
    n_targets: int = 5
    n_seeds: int = 5


def make_job_key(target_idx: int, noise_level: float, seed: int, architecture: str) -> str:
    return f"{target_idx}|{noise_level:.4f}|{seed}|{architecture}"


def init_worker(targets: List[np.ndarray]) -> None:
    global WORKER_TARGETS
    WORKER_TARGETS = targets
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_single_job(job: dict) -> dict:
    target_idx = int(job["target_idx"])
    noise_level = float(job["noise_level"])
    seed = int(job["seed"])
    arch_name = str(job["architecture"])
    arch_id = int(job.get("arch_id", zlib.crc32(arch_name.encode("utf-8")) & 0xFFFFFFFF))
    steps = int(job["steps"])
    lr = float(job["lr"])

    record = {
        "target_idx": target_idx,
        "target": str(job["target"]),
        "noise_level": noise_level,
        "seed": seed,
        "architecture": arch_name,
        "job_key": make_job_key(target_idx, noise_level, seed, arch_name),
    }

    try:
        target_clean = torch.from_numpy(WORKER_TARGETS[target_idx]).unsqueeze(0)
        img_size = int(target_clean.shape[-1])

        noise_gen = torch.Generator().manual_seed(seed + target_idx * 1000)
        noise = torch.randn(target_clean.shape, generator=noise_gen)
        target_noisy = torch.clamp(target_clean + noise * noise_level, 0, 1)

        arch_seed = seed * 10000 + target_idx * 100 + int(round(noise_level * 1000)) + arch_id
        torch.manual_seed(arch_seed)
        np.random.seed(arch_seed % (2**32 - 1))

        model = make_architecture(arch_name, img_size=img_size)
        best_psnr, best_iter = train_single_run(
            model=model,
            target_clean=target_clean,
            target_noisy=target_noisy,
            steps=steps,
            lr=lr,
        )

        record.update(
            {
                "status": "success",
                "best_psnr": float(best_psnr),
                "best_iter": int(best_iter),
            }
        )
    except Exception as exc:
        record.update({"status": "error", "error": str(exc)})

    return record


# ==========================================
# 5. IO, RESUME, SUMMARY
# ==========================================


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def load_existing_results(path: Path) -> List[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_success_key_set(rows: List[dict]) -> set:
    completed = set()
    for row in rows:
        if row.get("status") == "success":
            completed.add(
                make_job_key(
                    int(row["target_idx"]),
                    float(row["noise_level"]),
                    int(row["seed"]),
                    str(row["architecture"]),
                )
            )
    return completed


def write_status(status_path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = now_str()
    with status_path.open("w") as f:
        json.dump(payload, f, indent=2)


def summarize_results(
    runs: List[dict],
    noise_levels: Tuple[float, ...],
    architectures: Tuple[str, ...],
) -> dict:
    successful = [r for r in runs if r.get("status") == "success"]

    by_arch = {}
    for arch in architectures:
        arch_runs = [r for r in successful if r["architecture"] == arch]
        if not arch_runs:
            continue
        psnrs = np.array([r["best_psnr"] for r in arch_runs], dtype=float)
        by_arch[arch] = {
            "n": int(len(psnrs)),
            "mean": float(np.mean(psnrs)),
            "std": float(np.std(psnrs)),
            "min": float(np.min(psnrs)),
            "max": float(np.max(psnrs)),
        }

    by_noise = {}
    for nl in noise_levels:
        nl_key = f"{nl:.2f}"
        by_noise[nl_key] = {}
        for arch in architectures:
            runs_nl = [
                r
                for r in successful
                if r["architecture"] == arch and abs(float(r["noise_level"]) - nl) < 1e-9
            ]
            if not runs_nl:
                continue
            psnrs = np.array([r["best_psnr"] for r in runs_nl], dtype=float)
            by_noise[nl_key][arch] = {
                "n": int(len(psnrs)),
                "mean": float(np.mean(psnrs)),
                "std": float(np.std(psnrs)),
            }

    pair_map: Dict[Tuple[int, float, int], Dict[str, float]] = {}
    for row in successful:
        key = (int(row["target_idx"]), float(row["noise_level"]), int(row["seed"]))
        pair_map.setdefault(key, {})[row["architecture"]] = float(row["best_psnr"])

    cppn_vs_fourier = {}
    have_cppn_fourier = ("CPPN" in architectures) and ("Fourier" in architectures)
    for nl in noise_levels:
        cppn_vals = []
        fourier_vals = []
        for (target_idx, noise_level, seed), vals in pair_map.items():
            del target_idx, seed
            if abs(noise_level - nl) > 1e-9:
                continue
            if have_cppn_fourier and "CPPN" in vals and "Fourier" in vals:
                cppn_vals.append(vals["CPPN"])
                fourier_vals.append(vals["Fourier"])

        nl_key = f"{nl:.2f}"
        if cppn_vals and fourier_vals:
            cppn_arr = np.array(cppn_vals, dtype=float)
            fourier_arr = np.array(fourier_vals, dtype=float)
            try:
                t_stat, p_val = stats.ttest_rel(cppn_arr, fourier_arr)
            except Exception:
                t_stat, p_val = float("nan"), float("nan")
            cppn_vs_fourier[nl_key] = {
                "n_pairs": int(len(cppn_arr)),
                "cppn_mean": float(np.mean(cppn_arr)),
                "fourier_mean": float(np.mean(fourier_arr)),
                "delta_cppn_minus_fourier": float(np.mean(cppn_arr - fourier_arr)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }

    return {
        "successful_runs": int(len(successful)),
        "total_runs": int(len(runs)),
        "by_architecture": by_arch,
        "by_noise_level": by_noise,
        "cppn_vs_fourier": cppn_vs_fourier,
    }


def create_visualization(
    summary: dict,
    output_dir: Path,
    architectures: Tuple[str, ...],
    dataset: str,
    img_size: int,
) -> None:
    by_noise = summary.get("by_noise_level", {})
    if not by_noise:
        return

    noise_levels = sorted(float(k) for k in by_noise.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors = {
        "CPPN": "#1f77b4",
        "Fourier": "#ff7f0e",
        "ResNet-6": "#2ca02c",
        "Depthwise": "#d62728",
        "ViT": "#9467bd",
        "MLP": "#7f7f7f",
    }

    for arch in architectures:
        means = []
        stds = []
        valid_noise = []
        for nl in noise_levels:
            item = by_noise.get(f"{nl:.2f}", {}).get(arch)
            if item is None:
                continue
            valid_noise.append(nl)
            means.append(float(item["mean"]))
            stds.append(float(item["std"]))

        if valid_noise:
            ax1.errorbar(
                valid_noise,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                label=arch,
                color=colors.get(arch, "gray"),
            )

    ax1.set_xlabel("Noise Level (sigma)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"DIP Performance vs Noise Level ({dataset}, {int(img_size)}x{int(img_size)})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    cppn_fourier = summary.get("cppn_vs_fourier", {})
    deltas = []
    bars_x = []
    bar_colors = []
    for nl in noise_levels:
        row = cppn_fourier.get(f"{nl:.2f}")
        if row is None:
            continue
        delta = float(row["delta_cppn_minus_fourier"])
        bars_x.append(nl)
        deltas.append(delta)
        bar_colors.append("#1f77b4" if delta >= 0 else "#d62728")

    if bars_x:
        ax2.bar(bars_x, deltas, width=0.03, color=bar_colors)
    ax2.axhline(y=0.0, color="black", linewidth=0.7)
    ax2.set_xlabel("Noise Level (sigma)")
    ax2.set_ylabel("CPPN - Fourier (dB)")
    ax2.set_title("Alignment Test (positive means CPPN wins)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "alignment_validation.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "alignment_validation.pdf", dpi=150, bbox_inches="tight")
    plt.close()


# ==========================================
# 6. MAIN
# ==========================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIP alignment validation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset for clean targets: cifar10 or tiny_imagenet",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="Target image size (CIFAR-10 is native 32; Tiny ImageNet is native 64)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("DATA_ROOT", "~/.cache/mnc_datasets"),
        help="Dataset cache directory (kept outside output-dir to avoid uploading data to GCS)",
    )
    parser.add_argument(
        "--require-dataset",
        action="store_true",
        help="Fail if dataset cannot be loaded (no synthetic fallback)",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        default=42,
        help="Seed for selecting dataset target images",
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker count (default: cpu_count)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=list(ARCHITECTURES_DEFAULT),
        help="Architectures to evaluate (default: CPPN Fourier ResNet-6 Depthwise)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "results/dip_alignment"),
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior JSONL progress")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit jobs (smoke test)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = str(args.dataset).strip().lower()
    img_size = int(args.img_size)
    data_root = Path(str(args.data_root)).expanduser().resolve()

    config = Config(
        noise_levels=tuple(args.noise_levels),
        steps=args.steps,
        lr=args.lr,
        n_targets=args.targets,
        n_seeds=args.seeds,
    )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "run_results.jsonl"
    summary_json = output_dir / "validation_results.json"
    status_json = output_dir / "status.json"
    manifest_json = output_dir / "run_manifest.json"

    workers = args.workers or cpu_count() or 1
    workers = max(1, min(int(workers), cpu_count() or int(workers)))

    flush_print("=" * 72)
    flush_print("DIP ALIGNMENT VALIDATION")
    flush_print("=" * 72)
    flush_print(f"Started: {now_str()}")
    flush_print(f"Output dir: {output_dir}")
    flush_print(f"Workers: {workers}")
    flush_print()

    flush_print("Loading targets...")
    targets = load_targets(
        dataset=dataset,
        n_images=config.n_targets,
        seed=int(args.target_seed),
        img_size=img_size,
        data_root=data_root,
        require_dataset=bool(args.require_dataset),
    )
    target_arrays = [
        target["image"].squeeze(0).detach().cpu().numpy().astype(np.float32) for target in targets
    ]

    architectures = tuple(args.architectures)

    arch_stats = {}
    flush_print("Measuring architecture priors...")
    for arch_name in architectures:
        model = make_architecture(arch_name, img_size=img_size)
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            sample = model()
            measured = compute_structure_score(sample)
        expected = ARCH_EXPECTED_STRUCTURE.get(arch_name)
        arch_stats[arch_name] = {
            "expected_structure": None if expected is None else float(expected),
            "measured_structure": float(measured),
            "params": int(params),
        }
        if expected is None:
            flush_print(f"  {arch_name:<10} measured={measured:.3f} params={params:,}")
        else:
            flush_print(
                f"  {arch_name:<10} expected={float(expected):.2f} measured={measured:.3f} params={params:,}"
            )

    jobs = []
    for target_idx, target in enumerate(targets):
        for noise_level in config.noise_levels:
            for seed in range(config.n_seeds):
                for arch_name in architectures:
                    jobs.append(
                        {
                            "target_idx": int(target_idx),
                            "target": str(target["name"]),
                            "noise_level": float(noise_level),
                            "seed": int(seed),
                            "architecture": arch_name,
                            "arch_id": int(
                                zlib.crc32(str(arch_name).encode("utf-8")) & 0xFFFFFFFF
                            ),
                            "steps": int(config.steps),
                            "lr": float(config.lr),
                        }
                    )

    existing_rows = []
    completed_success_keys = set()
    if not args.no_resume and results_jsonl.exists():
        existing_rows = load_existing_results(results_jsonl)
        completed_success_keys = build_success_key_set(existing_rows)
        flush_print(
            f"Resume mode: {len(completed_success_keys)} successful jobs found in {results_jsonl.name}"
        )

    pending_jobs = [
        j
        for j in jobs
        if make_job_key(j["target_idx"], j["noise_level"], j["seed"], j["architecture"])
        not in completed_success_keys
    ]

    if args.max_jobs is not None:
        pending_jobs = pending_jobs[: max(0, int(args.max_jobs))]

    total_jobs = len(jobs)
    flush_print(f"Total jobs: {total_jobs}")
    flush_print(f"Pending jobs: {len(pending_jobs)}")
    flush_print()

    manifest = {
        "created_at": now_str(),
        "config": {
            "dataset": dataset,
            "img_size": int(img_size),
            "data_root": str(data_root),
            "require_dataset": bool(args.require_dataset),
            "target_seed": int(args.target_seed),
            "architectures": list(architectures),
            "noise_levels": list(config.noise_levels),
            "steps": int(config.steps),
            "lr": float(config.lr),
            "n_targets": int(config.n_targets),
            "n_seeds": int(config.n_seeds),
            "workers": int(workers),
        },
        "targets": [
            {
                "idx": int(i),
                "name": str(t["name"]),
                "label": int(t["label"]),
            }
            for i, t in enumerate(targets)
        ],
        "architectures": arch_stats,
        "total_jobs": int(total_jobs),
        "pending_jobs": int(len(pending_jobs)),
    }
    with manifest_json.open("w") as f:
        json.dump(manifest, f, indent=2)

    if not pending_jobs:
        flush_print("No pending jobs. Recomputing summary from existing results.")
        all_rows = existing_rows
        summary = summarize_results(all_rows, config.noise_levels, architectures=architectures)
        payload = {
            "config": manifest["config"],
            "architectures": arch_stats,
            "runs": all_rows,
            "summary": summary,
            "generated_at": now_str(),
        }
        with summary_json.open("w") as f:
            json.dump(payload, f, indent=2)
        create_visualization(
            summary,
            output_dir,
            architectures=architectures,
            dataset=dataset,
            img_size=img_size,
        )
        write_status(
            status_json,
            {
                "state": "complete",
                "total_jobs": int(total_jobs),
                "completed_success": int(len(completed_success_keys)),
                "errors": int(len([r for r in all_rows if r.get("status") == "error"])),
            },
        )
        flush_print("Complete.")
        return 0

    interrupted = False
    start_ts = time.time()
    produced_rows = []

    write_status(
        status_json,
        {
            "state": "running",
            "total_jobs": int(total_jobs),
            "completed_success": int(len(completed_success_keys)),
            "pending_jobs": int(len(pending_jobs)),
            "workers": int(workers),
            "started_at": now_str(),
        },
    )

    try:
        with Pool(processes=workers, initializer=init_worker, initargs=(target_arrays,)) as pool:
            for idx, result in enumerate(pool.imap_unordered(run_single_job, pending_jobs), start=1):
                append_jsonl(results_jsonl, result)
                produced_rows.append(result)

                if idx % 10 == 0 or idx == len(pending_jobs):
                    elapsed = max(1e-6, time.time() - start_ts)
                    rate = idx / elapsed
                    remaining = len(pending_jobs) - idx
                    eta_min = (remaining / rate) / 60.0 if rate > 0 else float("inf")
                    last_psnr = result.get("best_psnr")
                    last_psnr_str = "n/a" if last_psnr is None else f"{last_psnr:.2f}dB"
                    flush_print(
                        f"[{now_str()}] {idx}/{len(pending_jobs)} pending done "
                        f"({100.0 * idx / max(len(pending_jobs), 1):.1f}%), "
                        f"{rate:.2f} jobs/s, ETA {eta_min:.1f} min, "
                        f"last={result.get('architecture')} {last_psnr_str} [{result.get('status')}]"
                    )

                    current_rows = existing_rows + produced_rows
                    current_success = len(
                        [r for r in current_rows if r.get("status") == "success"]
                    )
                    current_errors = len([r for r in current_rows if r.get("status") == "error"])
                    write_status(
                        status_json,
                        {
                            "state": "running",
                            "total_jobs": int(total_jobs),
                            "completed_success": int(current_success),
                            "errors": int(current_errors),
                            "pending_jobs": int(len(pending_jobs) - idx),
                            "workers": int(workers),
                            "started_at": datetime.fromtimestamp(start_ts).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        },
                    )
    except KeyboardInterrupt:
        interrupted = True
        flush_print("Interrupted by user.")
    except Exception as exc:
        interrupted = True
        flush_print(f"Run interrupted by error: {exc}")

    all_rows = existing_rows + produced_rows
    summary = summarize_results(all_rows, config.noise_levels, architectures=architectures)

    payload = {
        "config": manifest["config"],
        "architectures": arch_stats,
        "runs": all_rows,
        "summary": summary,
        "generated_at": now_str(),
    }
    with summary_json.open("w") as f:
        json.dump(payload, f, indent=2)

    create_visualization(
        summary,
        output_dir,
        architectures=architectures,
        dataset=dataset,
        img_size=img_size,
    )

    final_success = len([r for r in all_rows if r.get("status") == "success"])
    final_errors = len([r for r in all_rows if r.get("status") == "error"])
    final_state = "interrupted" if interrupted else "complete"
    write_status(
        status_json,
        {
            "state": final_state,
            "total_jobs": int(total_jobs),
            "completed_success": int(final_success),
            "errors": int(final_errors),
            "workers": int(workers),
            "finished_at": now_str(),
        },
    )

    flush_print()
    flush_print("=" * 72)
    flush_print("RESULTS SUMMARY")
    flush_print("=" * 72)
    flush_print(f"Successful runs: {summary['successful_runs']} / {summary['total_runs']}")

    for noise_key, row in summary.get("cppn_vs_fourier", {}).items():
        delta = row["delta_cppn_minus_fourier"]
        winner = "CPPN" if delta > 0 else "Fourier"
        flush_print(
            f"sigma={noise_key}: CPPN={row['cppn_mean']:.2f}dB, "
            f"Fourier={row['fourier_mean']:.2f}dB, "
            f"delta={delta:+.2f}dB -> {winner} (p={row['p_value']:.4f})"
        )

    flush_print()
    flush_print(f"Saved summary: {summary_json}")
    flush_print(f"Saved figure: {output_dir / 'alignment_validation.png'}")
    flush_print(f"Status file: {status_json}")

    return 1 if interrupted else 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    raise SystemExit(main())
