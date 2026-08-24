# SPDX-License-Identifier: Apache-2.0
"""Plot reconstruction quality vs compression rate from metrics.json (no GPU needed).

Usage:
    python eval_quality/plot_results.py [--metrics eval_quality/results/metrics.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Fixed color per family (validated 4-slot categorical palette, light mode).
FAMILIES = {
    "CI": ("#2a78d6", "Continuous Image (CI)"),
    "DI": ("#eb6834", "Discrete Image (DI)"),
    "CV": ("#1baf7a", "Continuous Video (CV)"),
    "DV": ("#eda100", "Discrete Video (DV)"),
}
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def plot_panel(ax, results, metric, ylabel):
    for family, (color, label) in FAMILIES.items():
        points = sorted((r for r in results if r["family"] == family), key=lambda r: r["compression"])
        if not points:
            continue
        ax.plot(
            [p["compression"] for p in points],
            [p[metric] for p in points],
            color=color, linewidth=2, marker="o", markersize=7, label=label, zorder=3,
        )
        for p in points:
            legacy = p["series"] == "legacy"
            if legacy:  # hollow marker distinguishes the 0.1 fill-in checkpoints
                ax.plot(p["compression"], p[metric], marker="o", markersize=7, color=color,
                        markerfacecolor=SURFACE, zorder=4)
            dy = 9 if family in ("CI", "CV") else -15  # stagger labels to avoid collisions
            ax.annotate(
                p["name"], (p["compression"], p[metric]),
                textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7, color=MUTED,
            )

    ax.set_xscale("log", base=2)
    ticks = sorted({r["compression"] for r in results})
    ax.set_xticks(ticks, [f"{t}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("compression rate (spatio-temporal downsampling factor)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_facecolor(SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="eval_quality/results/metrics.json")
    parser.add_argument("--output", default=None, help="default: quality_vs_compression.png next to metrics.json")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    data = json.loads(metrics_path.read_text())
    results = [r for r in data["results"] if "error" not in r]
    skipped = [r["name"] for r in data["results"] if "error" in r]
    if skipped:
        print(f"Skipping failed tokenizers: {skipped}")

    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    plot_panel(ax_psnr, results, "psnr", "PSNR (dB)")
    plot_panel(ax_ssim, results, "ssim", "SSIM")
    ax_psnr.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

    n_img = sum(1 for r in results if r["kind"] == "image")
    n_vid = len(results) - n_img
    fig.suptitle("Cosmos tokenizer reconstruction quality vs compression rate", color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.925, f"{n_img} image + {n_vid} video tokenizers · hollow markers = legacy Cosmos-0.1 checkpoints",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output = Path(args.output) if args.output else metrics_path.parent / "quality_vs_compression.png"
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
