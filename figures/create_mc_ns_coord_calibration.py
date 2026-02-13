#!/usr/bin/env python3
"""
Create MC-vs-NS calibration scatter for the matched coordinate trio.

Data source (reported in manuscript Table tab:mc_ns_coord_calibration):
- CoordConvNet: B_NS=0.72, B_MC=0.67
- CoordMLP:     B_NS=2.11, B_MC=2.06
- CoordViT:     B_NS=2.67, B_MC=5.38
"""

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    points = [
        ("CoordConvNet", 0.72, 0.67, "#2ca02c"),
        ("CoordMLP", 2.11, 2.06, "#ff7f0e"),
        ("CoordViT", 2.67, 5.38, "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    colors = [p[3] for p in points]

    ax.scatter(xs, ys, c=colors, s=120, edgecolors="black", linewidths=0.6, zorder=3)

    x_min, x_max = 0.0, 6.0
    ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", color="gray", linewidth=1.5, label="y = x")

    for name, x, y, _ in points:
        dx = 0.08
        dy = 0.10 if name != "CoordViT" else 0.15
        ax.text(x + dx, y + dy, name, fontsize=12)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(r"NS crossing bits $B_{\mathrm{NS}}^{\mathrm{cross}}(\tau{=}0.1)$", fontsize=12)
    ax.set_ylabel(r"MC tail-surprisal bits $B_{\mathrm{MC}}(\tau{=}0.1)$", fontsize=12)
    ax.set_title("MC vs NS Calibration (Matched Coordinate Trio)", fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    this_dir = Path(__file__).resolve().parent
    out_pdf = this_dir / "mc_ns_coord_calibration_scatter.pdf"
    out_png = this_dir / "mc_ns_coord_calibration_scatter.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")

    arxiv_fig_dir = this_dir.parent / "arxiv_submission" / "figures"
    arxiv_fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(arxiv_fig_dir / out_pdf.name, dpi=300, bbox_inches="tight")
    fig.savefig(arxiv_fig_dir / out_png.name, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print(f"Wrote {arxiv_fig_dir / out_pdf.name}")
    print(f"Wrote {arxiv_fig_dir / out_png.name}")


if __name__ == "__main__":
    main()
