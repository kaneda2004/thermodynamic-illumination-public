#!/usr/bin/env python3
"""
Generate submission figures from canonical result artifacts.

Usage:
    uv run python generate_figures.py
    uv run python generate_figures.py --figure 1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats

# Configure matplotlib for publication quality
rcParams["font.family"] = "serif"
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 11
rcParams["axes.titlesize"] = 12
rcParams["legend.fontsize"] = 9
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
FIGURE_DIR = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RES358_PATH = RESULTS_DIR / "clean_comparison" / "res_358_results.json"
RES336_PATH = RESULTS_DIR / "reconstruction_13arch" / "res_336_results.json"
PRIOR_CURVE_CSV = RESULTS_DIR / "prior_comparison_multiplicative" / "curve_summary_all.csv"
SAMPLE_EFFICIENCY_JSON = RESULTS_DIR / "sample_efficiency_results.json"

COLORS = {
    "CPPN": "#2ecc71",
    "Uniform": "#7f8c8d",
    "CoordConvNet": "#228B22",
    "CoordMLP": "#e67e22",
    "CoordViT": "#B22222",
}


def setup_figure_dir():
    FIGURE_DIR.mkdir(exist_ok=True)


def load_prior_curves():
    if not PRIOR_CURVE_CSV.exists():
        raise FileNotFoundError(f"Missing prior curve CSV: {PRIOR_CURVE_CSV}")

    df = pd.read_csv(PRIOR_CURVE_CSV)
    curves = {}
    for prior_key, label in (("cppn", "CPPN"), ("uniform", "Uniform")):
        prior_df = df[df["prior"] == prior_key]
        if prior_df.empty:
            continue

        bits = -prior_df["log_X"].to_numpy() / np.log(2)
        order = prior_df["order_mean"].to_numpy()
        q16 = prior_df["order_q16"].to_numpy() if "order_q16" in prior_df else None
        q84 = prior_df["order_q84"].to_numpy() if "order_q84" in prior_df else None

        sort_idx = np.argsort(bits)
        curves[label] = {
            "bits": bits[sort_idx],
            "order": order[sort_idx],
            "order_q16": q16[sort_idx] if q16 is not None else None,
            "order_q84": q84[sort_idx] if q84 is not None else None,
        }

    return curves


def load_res358_coord_curves():
    if not RES358_PATH.exists():
        raise FileNotFoundError(f"Missing RES-358 JSON: {RES358_PATH}")

    with open(RES358_PATH) as f:
        data = json.load(f)

    name_map = {
        "ConvNet": "CoordConvNet",
        "MLP": "CoordMLP",
        "ViT": "CoordViT",
    }
    curves = {}
    for src_name, display_name in name_map.items():
        src = data["results"][src_name]
        pairs = []
        for tau_str, bits_val in src["bits_results"].items():
            if isinstance(bits_val, (int, float)):
                pairs.append((float(tau_str), float(bits_val)))
        pairs.sort(key=lambda x: x[0])

        thresholds = np.array([p[0] for p in pairs], dtype=float)
        bits = np.array([p[1] for p in pairs], dtype=float)
        curves[display_name] = {
            "thresholds": thresholds,
            "bits": bits,
            "pass_rate": float(src["initial_pass_rate"]),
            "bits_at_01": float(src["bits_at_01"]),
        }

    return curves


def bits_to_threshold(curve, tau):
    mask = curve["order"] >= tau
    if np.any(mask):
        idx = int(np.argmax(mask))
        return float(curve["bits"][idx]), True
    return float(curve["bits"].max()), False


def find_bits_at_tau(coord_curve, tau):
    idx = np.argmin(np.abs(coord_curve["thresholds"] - tau))
    return float(coord_curve["bits"][idx])


def fig1_volume_comparison():
    print("Generating Figure 1: Volume Comparison...")

    prior_curves = load_prior_curves()
    coord_curves = load_res358_coord_curves()

    fig, ax = plt.subplots(figsize=(8, 5))

    for label in ("CPPN", "Uniform"):
        if label not in prior_curves:
            continue
        c = prior_curves[label]
        ax.plot(c["bits"], c["order"], label=label, color=COLORS[label], linewidth=2.5)
        if c["order_q16"] is not None and c["order_q84"] is not None:
            ax.fill_between(c["bits"], c["order_q16"], c["order_q84"], color=COLORS[label], alpha=0.15)

    for arch in ("CoordConvNet", "CoordMLP", "CoordViT"):
        c = coord_curves[arch]
        ax.plot(
            c["bits"],
            c["thresholds"],
            label=arch,
            color=COLORS[arch],
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.6,
        )

    ax.set_xlabel(r"NS depth ($-\log_2 X$)")
    ax.set_ylabel("Order Metric (Multiplicative)")
    ax.set_title("Structure Discovery: Prior Volume vs Order Achieved")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.6)
    # Keep threshold annotation away from the lower-right legend box.
    ax.text(
        2.0,
        0.12,
        "tau = 0.1 threshold",
        fontsize=9,
        color="gray",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
    )
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1_volume_comparison.png")
    plt.savefig(FIGURE_DIR / "fig1_volume_comparison.pdf")
    plt.close()
    print("  Saved: fig1_volume_comparison.png/pdf")


def fig1b_volume_comparison_zoomed():
    print("Generating Figure 1b: Volume Comparison (Zoomed 0-15 bits)...")

    prior_curves = load_prior_curves()
    coord_curves = load_res358_coord_curves()

    fig, ax = plt.subplots(figsize=(8, 5))

    if "CPPN" in prior_curves:
        c = prior_curves["CPPN"]
        mask = c["bits"] <= 15
        ax.plot(c["bits"][mask], c["order"][mask], label="CPPN", color=COLORS["CPPN"], linewidth=2.5)

    for arch in ("CoordConvNet", "CoordMLP", "CoordViT"):
        c = coord_curves[arch]
        ax.plot(
            c["bits"],
            c["thresholds"],
            label=arch,
            color=COLORS[arch],
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1.6,
        )

    tau = 0.1
    conv_bits = find_bits_at_tau(coord_curves["CoordConvNet"], tau)
    vit_bits = find_bits_at_tau(coord_curves["CoordViT"], tau)
    gap_bits = vit_bits - conv_bits

    ax.annotate("", xy=(conv_bits, tau), xytext=(vit_bits, tau), arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(
        (conv_bits + vit_bits) / 2.0,
        tau + 0.14,
        f"{gap_bits:.2f}-bit NS gap",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.set_xlabel(r"NS depth ($-\log_2 X$)")
    ax.set_ylabel("Order Metric (Multiplicative)")
    ax.set_title("Early Structure Emergence (0-15 bits)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=tau, color="gray", linestyle="--", alpha=0.6)
    # Keep threshold annotation away from the lower-right legend box.
    ax.text(
        10.7,
        0.115,
        "tau = 0.1 threshold",
        fontsize=9,
        color="gray",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1b_volume_zoomed.png")
    plt.savefig(FIGURE_DIR / "fig1b_volume_zoomed.pdf")
    plt.close()
    print("  Saved: fig1b_volume_zoomed.png/pdf")


def fig2_bits_bar_chart():
    print("Generating Figure 2: Bits Required Bar Chart...")

    prior_curves = load_prior_curves()
    coord_curves = load_res358_coord_curves()

    cppn_bits, cppn_reached = bits_to_threshold(prior_curves["CPPN"], 0.1)
    uniform_bits, uniform_reached = bits_to_threshold(prior_curves["Uniform"], 0.1)

    entries = [
        ("CPPN", cppn_bits, cppn_reached, COLORS["CPPN"]),
        ("CoordConvNet", coord_curves["CoordConvNet"]["bits_at_01"], True, COLORS["CoordConvNet"]),
        ("CoordMLP", coord_curves["CoordMLP"]["bits_at_01"], True, COLORS["CoordMLP"]),
        ("CoordViT", coord_curves["CoordViT"]["bits_at_01"], True, COLORS["CoordViT"]),
        ("Uniform", uniform_bits, uniform_reached, COLORS["Uniform"]),
    ]

    labels = [e[0] for e in entries]
    bits = [e[1] for e in entries]
    reached = [e[2] for e in entries]
    colors = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, bits, color=colors, edgecolor="black", linewidth=0.6)

    max_bits = max(bits)
    for bar, b, r in zip(bars, bits, reached):
        label = f"{b:.2f}" if r else f">={b:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max_bits * 0.02, label, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Bits to Reach tau = 0.1")
    ax.set_title("Bits to Structure at tau = 0.1")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max_bits * 1.2)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig2_bits_bar_chart.png")
    plt.savefig(FIGURE_DIR / "fig2_bits_bar_chart.pdf")
    plt.close()
    print("  Saved: fig2_bits_bar_chart.png/pdf")


def fig_threshold_curve():
    print("Generating Threshold Curve: B(tau) for Coord Architectures...")

    coord_curves = load_res358_coord_curves()

    fig, ax = plt.subplots(figsize=(8, 5))

    for arch in ("CoordConvNet", "CoordMLP", "CoordViT"):
        c = coord_curves[arch]
        ax.plot(c["thresholds"], c["bits"], marker="o", linewidth=2, color=COLORS[arch], label=arch)

    tau = 0.1
    conv_bits = find_bits_at_tau(coord_curves["CoordConvNet"], tau)
    vit_bits = find_bits_at_tau(coord_curves["CoordViT"], tau)
    ax.axvline(x=tau, color="gray", linestyle="--", alpha=0.6)
    ax.text(
        tau + 0.015,
        8.6,
        r"$\tau = 0.1$",
        color="gray",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.text(
        0.28,
        1.05,
        f"Delta at $\\tau=0.1$: {vit_bits - conv_bits:.2f} bits",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.set_xlabel(r"Order Threshold $\tau$")
    ax.set_ylabel(r"NS crossing bits $B_{\mathrm{NS}}^{\mathrm{cross}}(\tau)$")
    ax.set_title("Threshold Robustness for Coordinate-Conditioned Architectures")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, max([coord_curves[a]["bits"].max() for a in coord_curves]) * 1.1)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bits_threshold_curve.png")
    plt.savefig(FIGURE_DIR / "bits_threshold_curve.pdf")
    plt.close()
    print("  Saved: bits_threshold_curve.png/pdf")


def fig3_reconstruction_correlation():
    print("Generating Figure 3: Reconstruction Correlation...")

    if not RES336_PATH.exists():
        print(f"  Warning: {RES336_PATH} not found. Skipping.")
        return

    with open(RES336_PATH) as f:
        data = json.load(f)

    points = data.get("results", [])
    if not points:
        print("  Warning: no reconstruction points found. Skipping.")
        return

    bits = np.array([p["bits"] for p in points], dtype=float)
    mse = np.array([p["mse"] for p in points], dtype=float)
    names = [p["name"] for p in points]

    rho, p_value = stats.spearmanr(bits, mse)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bits, mse, s=55, color="#1f77b4", edgecolors="black", linewidth=0.5, alpha=0.85)

    if len(bits) >= 2:
        fit = np.polyfit(bits, mse, deg=1)
        trend = np.poly1d(fit)
        x_line = np.linspace(bits.min(), bits.max(), 200)
        ax.plot(x_line, trend(x_line), "--", color="gray", linewidth=1.5)

    # Annotate only representative extremes to reduce clutter.
    low_idx = int(np.argmin(bits))
    high_idx = int(np.argmax(bits))
    for idx in (low_idx, high_idx):
        ax.annotate(names[idx], (bits[idx], mse[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel(r"NS crossing bits $B_{\mathrm{NS}}^{\mathrm{cross}}(\tau=0.1)$ (lower = more structured)")
    ax.set_ylabel("Reconstruction MSE (lower = better)")
    ax.set_title(f"Reconstruction Quality vs NS Crossing Bits\nSpearman $\\rho$ = {rho:.3f}, p = {p_value:.4f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig3_reconstruction_correlation.png")
    plt.savefig(FIGURE_DIR / "fig3_reconstruction_correlation.pdf")
    plt.close()
    print("  Saved: fig3_reconstruction_correlation.png/pdf")


def fig4_classification_null():
    print("Generating Figure 4: Classification Null Result...")

    if not SAMPLE_EFFICIENCY_JSON.exists():
        print(f"  Warning: {SAMPLE_EFFICIENCY_JSON} not found. Skipping.")
        return

    with open(SAMPLE_EFFICIENCY_JSON) as f:
        data = json.load(f)

    mnist_data = data.get("mnist", {})
    architectures = mnist_data.get("results", [])
    if not architectures:
        print("  Warning: no sample-efficiency points found. Skipping.")
        return

    family_styles = {
        "CPPN": {"color": "#2ecc71", "marker": "o", "label": "CPPN"},
        "Conv": {"color": "#3498db", "marker": "s", "label": "Conv"},
        "ResNet": {"color": "#9b59b6", "marker": "^", "label": "ResNet"},
        "MLP": {"color": "#e74c3c", "marker": "D", "label": "MLP"},
        "Fourier": {"color": "#f39c12", "marker": "p", "label": "Fourier"},
    }

    def family_of(name):
        for family in ("CPPN", "ResNet", "Conv", "MLP", "Fourier"):
            if name.startswith(family):
                return family
        return None

    grouped = {k: {"bits": [], "acc": []} for k in family_styles}
    all_bits = []
    all_acc = []

    for arch in architectures:
        if "bits" not in arch or "mean_final_acc" not in arch:
            continue
        fam = family_of(arch.get("name", ""))
        if fam is None:
            continue
        b = float(arch["bits"])
        a = float(arch["mean_final_acc"])
        grouped[fam]["bits"].append(b)
        grouped[fam]["acc"].append(a)
        all_bits.append(b)
        all_acc.append(a)

    if not all_bits:
        print("  Warning: no valid classification points. Skipping.")
        return

    rho, p_value = stats.spearmanr(all_bits, all_acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, style in family_styles.items():
        if grouped[family]["bits"]:
            ax.scatter(
                grouped[family]["bits"],
                grouped[family]["acc"],
                s=60,
                c=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidth=0.5,
                alpha=0.8,
                label=style["label"],
            )

    ax.set_xlabel(r"NS crossing bits $B_{\mathrm{NS}}^{\mathrm{cross}}(\tau=0.1)$ (lower = more structured)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title(f"Classification vs NS Crossing Bits\nSpearman $\\rho$ = {rho:.3f}, p = {p_value:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax.text(
        0.5,
        0.97,
        "No significant correlation: bits predicts reconstruction, not classification",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig4_classification_null.png")
    plt.savefig(FIGURE_DIR / "fig4_classification_null.pdf")
    plt.close()
    print("  Saved: fig4_classification_null.png/pdf")


def fig5_sample_images_grid():
    print("Generating Figure 5: Sample Images Grid...")

    priors = ["cppn", "uniform"]
    images = {}

    for prior in priors:
        prior_dir = RESULTS_DIR / "prior_comparison_multiplicative" / prior
        if not prior_dir.exists():
            continue
        for run_dir in sorted(prior_dir.glob("run_*")):
            best_files = list(run_dir.glob("best*.pbm"))
            if not best_files:
                continue
            try:
                images[prior] = load_pbm(best_files[0])
                break
            except Exception as exc:
                print(f"  Warning: could not load {best_files[0]}: {exc}")

    if not images:
        print("  Warning: no sample PBM files found. Skipping.")
        return

    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    if len(images) == 1:
        axes = [axes]

    for ax, (prior, img) in zip(axes, images.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(prior.upper())
        ax.axis("off")

    plt.suptitle("Best Samples from Each Prior", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig5_sample_images.png")
    plt.savefig(FIGURE_DIR / "fig5_sample_images.pdf")
    plt.close()
    print("  Saved: fig5_sample_images.png/pdf")


def fig6_demo_results():
    print("Generating Figure 6: Demo Results...")

    demo_dir = PROJECT_ROOT / "demo_output"
    if not demo_dir.exists():
        print(f"  Warning: {demo_dir} not found. Skipping.")
        return

    image_files = [
        ("1_original.png", "Original"),
        ("2_masked.png", "Masked (30%)"),
        ("3_cppn_reconstruction.png", "CPPN Reconstruction"),
        ("4_mlp_reconstruction.png", "MLP Reconstruction"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, (filename, title) in zip(axes, image_files):
        img_path = demo_dir / filename
        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Sparse Reconstruction Demo: CPPN vs MLP", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig6_demo_results.png")
    plt.savefig(FIGURE_DIR / "fig6_demo_results.pdf")
    plt.close()
    print("  Saved: fig6_demo_results.png/pdf")


def load_pbm(filepath):
    with open(filepath, "rb") as f:
        magic = f.readline().decode().strip()
        if magic not in {"P1", "P4"}:
            raise ValueError(f"Not a valid PBM file: {magic}")

        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        width, height = map(int, line.decode().split())

        if magic == "P1":
            data = []
            for row in f:
                data.extend(int(x) for x in row.decode().split())
            img = np.array(data).reshape(height, width)
        else:
            row_bytes = (width + 7) // 8
            data = f.read(row_bytes * height)
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            img = bits[: height * width].reshape(height, width)

        return 1 - img


def main():
    parser = argparse.ArgumentParser(description="Generate submission figures")
    parser.add_argument("--figure", "-f", type=int, choices=[1, 2, 3, 4, 5, 6], help="Generate one figure only")
    args = parser.parse_args()

    setup_figure_dir()

    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    figure_funcs = [
        (1, fig1_volume_comparison),
        (2, fig2_bits_bar_chart),
        (3, fig3_reconstruction_correlation),
        (4, fig4_classification_null),
        (5, fig5_sample_images_grid),
        (6, fig6_demo_results),
    ]

    for fig_num, func in figure_funcs:
        if args.figure is None or args.figure == fig_num:
            try:
                func()
            except Exception as exc:
                print(f"  Error generating figure {fig_num}: {exc}")

    if args.figure is None or args.figure == 1:
        try:
            fig1b_volume_comparison_zoomed()
            fig_threshold_curve()
        except Exception as exc:
            print(f"  Error generating zoomed/threshold curves: {exc}")

    print()
    print("=" * 60)
    print(f"Figures saved to: {FIGURE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
