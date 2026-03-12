"""
Create a compact schematic of the Thermodynamic Illumination pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
OUT_PDF = OUT_DIR / "prior_pipeline_schematic.pdf"
OUT_PNG = OUT_DIR / "prior_pipeline_schematic.png"


def box(ax, x, y, w, h, text, fc="#f5f5f5", ec="#444444", fs=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#333333"),
    )


def create():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Thermodynamic Illumination Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    box(ax, 0.03, 0.58, 0.17, 0.22, "Architecture\n+ weight prior\n$\\theta \\sim p(\\theta)$", fc="#e8f3ff")
    box(ax, 0.24, 0.58, 0.17, 0.22, "Generate image\n$x=f_\\theta(u)$\n$u$ fixed by family", fc="#eef9ee")
    box(ax, 0.45, 0.58, 0.17, 0.22, "Order metric\n$O(x)$\nthreshold $\\tau$", fc="#fff6e8")
    box(ax, 0.66, 0.70, 0.15, 0.14, "MC path:\n$p(\\tau)=\\Pr[O\\geq\\tau]$\n$B_{MC}=-\\log_2 p$", fc="#f2ecff", fs=11)
    box(ax, 0.66, 0.50, 0.15, 0.14, "NS path:\n$B_{NS}^{cross}$\n$\\hat p_{live}$", fc="#ffeef0", fs=11)
    box(ax, 0.84, 0.58, 0.13, 0.22, "Compare\narchitectures:\nrankings,\nregimes,\nDIP alignment", fc="#f0f0f0")

    arrow(ax, 0.20, 0.69, 0.24, 0.69)
    arrow(ax, 0.41, 0.69, 0.45, 0.69)
    arrow(ax, 0.62, 0.69, 0.66, 0.77)
    arrow(ax, 0.62, 0.69, 0.66, 0.57)
    arrow(ax, 0.81, 0.77, 0.84, 0.69)
    arrow(ax, 0.81, 0.57, 0.84, 0.69)

    ax.text(
        0.5,
        0.12,
        "Interpretation rule: probability ratios come from $B_{MC}$; "
        "$B_{NS}^{cross}$ is an operational threshold-crossing cost under fixed NS protocol.",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    create()
