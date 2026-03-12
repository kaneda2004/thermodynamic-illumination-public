#!/usr/bin/env python3
"""
RES-382: Fan-in-scaled Coord-Trios Rerun

Purpose:
Re-evaluate the matched 32x32 coordinate-conditioned ConvNet/MLP/ViT trio
under exact Monte Carlo sampling from three priors:

1. Fixed isotropic Gaussian N(0, 0.3)  (matches RES-358)
2. Xavier-normal fan-in/out scaled prior
3. He-normal fan-in scaled prior

Why Monte Carlo instead of nested sampling?
For the scaled priors, the clean question is simply whether the pass-rate /
tail-mass ranking changes under exact prior sampling. MC answers that directly
without introducing a second approximation in the proposal kernel.

Outputs:
- run_manifest.json
- status.json
- results_partial.json         (atomically updated after every completed block)
- result_<scheme>_<arch>.json  (per-block atomically saved summary)
- res_382_results.json         (final summary)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from core.thermo_sampler_v3 import order_multiplicative


def atomic_json_save(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def get_output_dir(output_dir_arg: Optional[str]) -> Path:
    env_dir = output_dir_arg or os.environ.get("OUTPUT_DIR")
    if env_dir:
        out = Path(env_dir)
        return out if out.is_absolute() else REPO_ROOT / out
    return REPO_ROOT / "results" / "fanin_scaled_coord_rerun"


def update_status(
    output_dir: Path,
    phase: str,
    scheme: str,
    architecture: str,
    step: int,
    total: int,
    message: str = "",
) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "scheme": scheme,
        "architecture": architecture,
        "step": step,
        "total": total,
        "progress_pct": round(100.0 * step / max(total, 1), 1),
        "message": message,
    }
    atomic_json_save(payload, output_dir / "status.json")


def stable_seed(base_seed: int, scheme_idx: int, arch_idx: int) -> int:
    return int(base_seed + 1000 * scheme_idx + 17 * arch_idx)


def make_coord_grid(size: int) -> torch.Tensor:
    c = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x, y], dim=0).unsqueeze(0)


def make_coord_pairs(size: int) -> torch.Tensor:
    c = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], dim=1)


class CoordConvNet(nn.Module):
    def __init__(self, hidden: int = 32, image_size: int = 32):
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def render(self) -> np.ndarray:
        coords = make_coord_grid(self.image_size)
        with torch.no_grad():
            out = self.net(coords).squeeze()
        return (out.numpy() > 0.5).astype(np.uint8)


class CoordMLP(nn.Module):
    def __init__(self, hidden: int = 64, image_size: int = 32):
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def render(self) -> np.ndarray:
        coords = make_coord_pairs(self.image_size)
        with torch.no_grad():
            out = self.net(coords).view(self.image_size, self.image_size)
        return (out.numpy() > 0.5).astype(np.uint8)


class CoordViT(nn.Module):
    def __init__(self, embed_dim: int = 32, n_heads: int = 4, n_layers: int = 2, image_size: int = 32):
        super().__init__()
        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)

        self.coord_embed = nn.Linear(2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 1)
        self.register_buffer("pos_encoding", self._make_pos_encoding(self.image_size))

    def _make_pos_encoding(self, size: int) -> torch.Tensor:
        n_positions = size * size
        pe = torch.zeros(n_positions, self.embed_dim)
        position = torch.arange(0, n_positions).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def render(self) -> np.ndarray:
        coords = make_coord_pairs(self.image_size).unsqueeze(0)
        with torch.no_grad():
            x = self.coord_embed(coords) + self.pos_encoding
            x = self.transformer(x)
            out = torch.sigmoid(self.head(x)).view(self.image_size, self.image_size)
        return (out.numpy() > 0.5).astype(np.uint8)


def init_tensor_normal_(tensor: torch.Tensor, std: float) -> None:
    with torch.no_grad():
        nn.init.normal_(tensor, 0.0, std)


def init_tensor_xavier_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        nn.init.xavier_normal_(tensor)


def init_tensor_he_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        nn.init.kaiming_normal_(tensor, nonlinearity="relu")


def init_model(model: nn.Module, scheme: str, gaussian_std: float) -> None:
    if scheme == "gaussian_0p3":
        for p in model.parameters():
            init_tensor_normal_(p, gaussian_std)
        return

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            if scheme == "xavier":
                init_tensor_xavier_(module.in_proj_weight)
            elif scheme == "he":
                init_tensor_he_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            if scheme == "xavier":
                init_tensor_xavier_(module.weight)
            elif scheme == "he":
                init_tensor_he_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def attention_diagnostics(model: CoordViT) -> Dict[str, float]:
    coords = make_coord_pairs(model.image_size).unsqueeze(0)
    with torch.no_grad():
        x = model.coord_embed(coords) + model.pos_encoding
        layer = model.transformer.layers[0]
        mha = layer.self_attn
        qkv = torch.nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, _ = qkv.split(mha.embed_dim, dim=-1)
        n_heads = mha.num_heads
        head_dim = mha.embed_dim // n_heads
        q = q.view(1, -1, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, -1, n_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        probs = torch.softmax(logits, dim=-1)
        nonzero = probs[probs > 0]
        return {
            "logit_abs_max": float(logits.abs().max()),
            "logit_std": float(logits.std()),
            "attn_max": float(probs.max()),
            "attn_min_nonzero": float(nonzero.min()) if nonzero.numel() else 0.0,
            "zero_frac": float((probs == 0).float().mean()),
        }


def summarize_attention(diags: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["logit_abs_max", "logit_std", "attn_max", "attn_min_nonzero", "zero_frac"]
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([d[key] for d in diags], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "max": float(vals.max()),
            "min": float(vals.min()),
        }
    return summary


def exact_tail_bits(pass_count: int, n_samples: int) -> Optional[float]:
    if pass_count <= 0:
        return None
    return float(-math.log2(pass_count / n_samples))


def mc_block(
    model_factory,
    architecture: str,
    scheme: str,
    output_dir: Path,
    n_samples: int,
    tau: float,
    gaussian_std: float,
    logit_samples: int,
    status_every: int,
) -> Dict[str, Any]:
    orders: List[float] = []
    vit_diags: List[Dict[str, float]] = []
    start = time.time()

    for sample_idx in range(n_samples):
        model = model_factory()
        init_model(model, scheme=scheme, gaussian_std=gaussian_std)
        img = model.render()
        order = float(order_multiplicative(img))
        orders.append(order)

        if architecture == "ViT" and len(vit_diags) < logit_samples:
            vit_diags.append(attention_diagnostics(model))

        if (sample_idx + 1) % status_every == 0 or (sample_idx + 1) == n_samples:
            update_status(
                output_dir,
                phase="mc",
                scheme=scheme,
                architecture=architecture,
                step=sample_idx + 1,
                total=n_samples,
                message="sampling prior draws",
            )

    orders_arr = np.array(orders, dtype=float)
    pass_count = int(np.sum(orders_arr >= tau))
    result: Dict[str, Any] = {
        "architecture": architecture,
        "scheme": scheme,
        "n_samples": int(n_samples),
        "tau": float(tau),
        "mean_order": float(orders_arr.mean()),
        "std_order": float(orders_arr.std()),
        "median_order": float(np.median(orders_arr)),
        "max_order": float(orders_arr.max()),
        "pass_count": pass_count,
        "pass_rate": float(pass_count / n_samples),
        "tail_bits_estimate": exact_tail_bits(pass_count, n_samples),
        "elapsed_seconds": float(time.time() - start),
    }
    if architecture == "ViT":
        result["attention_diagnostics"] = summarize_attention(vit_diags)
    return result


def build_rankings(results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, List[str]]]:
    rankings: Dict[str, Dict[str, List[str]]] = {}
    for scheme, scheme_results in results.items():
        pass_rank = sorted(
            scheme_results.keys(),
            key=lambda arch: scheme_results[arch]["pass_rate"],
            reverse=True,
        )
        mean_rank = sorted(
            scheme_results.keys(),
            key=lambda arch: scheme_results[arch]["mean_order"],
            reverse=True,
        )
        rankings[scheme] = {
            "by_pass_rate": pass_rank,
            "by_mean_order": mean_rank,
        }
    return rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RES-382 fan-in-scaled coord-trio rerun")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", 32)))
    parser.add_argument("--mc-samples", type=int, default=int(os.environ.get("MC_SAMPLES", 5000)))
    parser.add_argument("--tau", type=float, default=float(os.environ.get("TAU", 0.1)))
    parser.add_argument("--gaussian-std", type=float, default=float(os.environ.get("GAUSSIAN_STD", 0.3)))
    parser.add_argument("--logit-samples", type=int, default=int(os.environ.get("LOGIT_SAMPLES", 128)))
    parser.add_argument("--status-every", type=int, default=int(os.environ.get("STATUS_EVERY", 250)))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = get_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "RES-382",
        "description": "Fan-in-scaled coord-trio rerun with exact MC under Gaussian/Xavier/He priors",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "config": {
            "seed": args.seed,
            "image_size": args.image_size,
            "mc_samples": args.mc_samples,
            "tau": args.tau,
            "gaussian_std": args.gaussian_std,
            "logit_samples": args.logit_samples,
            "status_every": args.status_every,
            "schemes": ["gaussian_0p3", "xavier", "he"],
            "architectures": ["ConvNet", "MLP", "ViT"],
        },
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    atomic_json_save(manifest, output_dir / "run_manifest.json")

    partial_path = output_dir / "results_partial.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        results = partial.get("results", {})
    else:
        results = {}

    architectures = [
        ("ConvNet", lambda: CoordConvNet(hidden=32, image_size=args.image_size)),
        ("MLP", lambda: CoordMLP(hidden=64, image_size=args.image_size)),
        ("ViT", lambda: CoordViT(embed_dim=32, n_heads=4, n_layers=2, image_size=args.image_size)),
    ]
    schemes = ["gaussian_0p3", "xavier", "he"]

    total_blocks = len(schemes) * len(architectures)
    completed_blocks = sum(
        1 for scheme in results.values() for block in scheme.values() if isinstance(block, dict) and block.get("completed")
    )

    for scheme_idx, scheme in enumerate(schemes):
        results.setdefault(scheme, {})
        for arch_idx, (arch_name, model_factory) in enumerate(architectures):
            if results[scheme].get(arch_name, {}).get("completed"):
                continue

            block_seed = stable_seed(args.seed, scheme_idx, arch_idx)
            np.random.seed(block_seed)
            torch.manual_seed(block_seed)

            update_status(
                output_dir,
                phase="mc",
                scheme=scheme,
                architecture=arch_name,
                step=0,
                total=args.mc_samples,
                message="starting block",
            )

            result = mc_block(
                model_factory=model_factory,
                architecture=arch_name,
                scheme=scheme,
                output_dir=output_dir,
                n_samples=args.mc_samples,
                tau=args.tau,
                gaussian_std=args.gaussian_std,
                logit_samples=args.logit_samples,
                status_every=args.status_every,
            )
            result["completed"] = True
            result["block_seed"] = block_seed

            results[scheme][arch_name] = result
            completed_blocks += 1

            atomic_json_save(result, output_dir / f"result_{scheme}_{arch_name}.json")
            atomic_json_save(
                {
                    "experiment": "RES-382",
                    "completed_blocks": completed_blocks,
                    "total_blocks": total_blocks,
                    "results": results,
                },
                partial_path,
            )

    final_output = {
        "experiment": "RES-382",
        "description": manifest["description"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": manifest["config"],
        "results": results,
        "rankings": build_rankings(results),
    }
    atomic_json_save(final_output, output_dir / "res_382_results.json")
    update_status(
        output_dir,
        phase="done",
        scheme="all",
        architecture="all",
        step=total_blocks,
        total=total_blocks,
        message="experiment complete",
    )

    print("RES-382 complete")
    for scheme, ranking in final_output["rankings"].items():
        print(f"{scheme}: pass-rate ranking = {' > '.join(ranking['by_pass_rate'])}")


if __name__ == "__main__":
    main()
