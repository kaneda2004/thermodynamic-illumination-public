"""
Create the ConvNet-vs-ViT teaser visual used in the paper.

Design requirement: keep dedicated vertical bands for:
1) main title
2) per-panel headers
3) image panels
4) per-panel metrics
5) bottom comparison subtitle

This avoids title/subtext overlap in all export targets.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

IMAGE_SIZE = 32
GRID_SIDE = 4
NUM_SAMPLES = GRID_SIDE * GRID_SIDE
SEED_BASE = 42

GREEN = "#228B22"
RED = "#B22222"

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = OUTPUT_DIR / "killer_visual.pdf"
OUTPUT_PNG = OUTPUT_DIR / "killer_visual.png"


class CoordConvNet(nn.Module):
    """Coordinate-conditioned ConvNet with local receptive fields."""

    def __init__(self, hidden: int = 32, n_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        layers = []
        in_channels = 2
        for _ in range(n_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            in_channels = hidden
        layers.append(nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.5)
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0.0, 0.1)

    @staticmethod
    def _coord_grid(size: int) -> torch.Tensor:
        coord = torch.linspace(-1, 1, size)
        x_grid, y_grid = torch.meshgrid(coord, coord, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    def render(self, size: int = IMAGE_SIZE) -> np.ndarray:
        with torch.no_grad():
            img = self.net(self._coord_grid(size)).squeeze().numpy()
        return (img > 0.5).astype(np.uint8)


class CoordViT(nn.Module):
    """Coordinate-conditioned ViT with global token mixing."""

    def __init__(self, embed_dim: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.coord_embed = nn.Linear(2, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 1)
        self.register_buffer("pos_encoding", self._make_pos_encoding(IMAGE_SIZE))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.3)
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0.0, 0.1)

    def _make_pos_encoding(self, size: int) -> torch.Tensor:
        n_positions = size * size
        pe = torch.zeros(n_positions, self.embed_dim)
        pos = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / self.embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def render(self, size: int = IMAGE_SIZE) -> np.ndarray:
        coord = torch.linspace(-1, 1, size)
        x_grid, y_grid = torch.meshgrid(coord, coord, indexing="ij")
        xy = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1).unsqueeze(0)
        with torch.no_grad():
            tokens = self.coord_embed(xy)
            tokens = tokens + self.pos_encoding
            tokens = self.transformer(tokens)
            img = torch.sigmoid(self.head(tokens)).view(size, size).numpy()
        return (img > 0.5).astype(np.uint8)


def _image_grid(images: list[np.ndarray], grid_side: int = GRID_SIDE, gap: int = 2, gap_value: float = 0.5) -> np.ndarray:
    cell_h, cell_w = images[0].shape
    canvas_h = grid_side * cell_h + (grid_side - 1) * gap
    canvas_w = grid_side * cell_w + (grid_side - 1) * gap
    canvas = np.full((canvas_h, canvas_w), gap_value, dtype=np.float32)
    for idx, image in enumerate(images):
        row = idx // grid_side
        col = idx % grid_side
        r0 = row * (cell_h + gap)
        c0 = col * (cell_w + gap)
        canvas[r0 : r0 + cell_h, c0 : c0 + cell_w] = image
    return canvas


def _sample_grid(model_cls: type[nn.Module]) -> np.ndarray:
    images = []
    for i in range(NUM_SAMPLES):
        torch.manual_seed(SEED_BASE + i * 1000)
        model = model_cls()
        images.append(model.render(IMAGE_SIZE))
    return _image_grid(images)


def create_killer_visual() -> None:
    conv_grid = _sample_grid(CoordConvNet)
    vit_grid = _sample_grid(CoordViT)

    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 0.22, 1.0],
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.22,
        wspace=0.06,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    ax_left.imshow(conv_grid, cmap="gray", vmin=0, vmax=1)
    ax_right.imshow(vit_grid, cmap="gray", vmin=0, vmax=1)
    for axis, color in ((ax_left, GREEN), (ax_right, RED)):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3.5)

    ax_mid.axis("off")
    ax_mid.text(0.5, 0.62, "x", fontsize=56, ha="center", va="center", transform=ax_mid.transAxes)
    ax_mid.text(0.5, 0.44, "~21x", fontsize=42, fontweight="bold", ha="center", va="center", transform=ax_mid.transAxes)
    ax_mid.text(0.5, 0.27, "pass-fraction\ngap", fontsize=20, style="italic", ha="center", va="center", transform=ax_mid.transAxes)

    left_bbox = ax_left.get_position()
    right_bbox = ax_right.get_position()
    left_x = 0.5 * (left_bbox.x0 + left_bbox.x1)
    right_x = 0.5 * (right_bbox.x0 + right_bbox.x1)

    fig.text(
        0.5,
        0.965,
        "The ConvNet-ViT Gap: Same Coordinates, Different Mixing",
        ha="center",
        va="top",
        fontsize=30,
        fontweight="bold",
    )

    fig.text(left_x, 0.885, "CoordConvNet Prior", color=GREEN, ha="center", va="top", fontsize=28, fontweight="bold")
    fig.text(left_x, 0.84, "16 random samples", color=GREEN, ha="center", va="top", fontsize=28, fontweight="bold")
    fig.text(right_x, 0.885, "CoordViT Prior", color=RED, ha="center", va="top", fontsize=28, fontweight="bold")
    fig.text(right_x, 0.84, "16 random samples", color=RED, ha="center", va="top", fontsize=28, fontweight="bold")

    fig.text(left_x, 0.155, "0.72 NS crossing bits at tau=0.1\n64% initial pass fraction", ha="center", va="center", fontsize=23)
    fig.text(right_x, 0.155, "2.67 NS crossing bits at tau=0.1\n3% initial pass fraction", ha="center", va="center", fontsize=23)

    fig.text(
        0.5,
        0.06,
        "~2 bit NS crossing-cost gap",
        ha="center",
        va="center",
        fontsize=24,
        style="italic",
    )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    create_killer_visual()
