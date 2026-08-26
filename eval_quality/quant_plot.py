# SPDX-License-Identifier: Apache-2.0
"""Plot INT8 PTQ quality vs the native-bf16 baseline (no GPU needed).

Reads eval_quality/results_quant/<config>/metrics.json produced by
quantize_ptq.py and renders: PSNR/SSIM per config, and per-clip quantization
damage vs clip motion for each quantized config.

Usage:
    python eval_quality/quant_plot.py [--results_dir eval_quality/results_quant]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ORDER = ["none", "w8_dec", "w8a8_dec", "w8a8_all", "w8a8_dec_sq"]
LABELS = {"none": "bf16\n(baseline)", "w8_dec": "W8\ndecoder", "w8a8_dec": "W8A8\ndecoder",
          "w8a8_all": "W8A8\nfull VAE", "w8a8_dec_sq": "W8A8 dec\nSmoothQuant"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical palette
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax, ylabel):
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
    parser.add_argument("--results_dir", default="eval_quality/results_quant")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = {}
    for path in results_dir.glob("*/metrics.json"):
        r = json.loads(path.read_text())["results"][0]
        runs[r.get("quant", path.parent.name)] = r
    configs = [q for q in ORDER if q in runs] + sorted(set(runs) - set(ORDER))
    if "none" not in runs:
        raise SystemExit("missing baseline run (results_quant/none) — run quantize_ptq.py --quant none first")
    base = runs["none"]
    tok_name = base["name"]

    fig, (ax_p, ax_s, ax_m) = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    xs = range(len(configs))
    for ax, metric, ylabel, fmt in ((ax_p, "psnr", "PSNR (dB)", "{:.2f}"), (ax_s, "ssim", "SSIM", "{:.3f}")):
        vals = [runs[q][metric] for q in configs]
        ax.bar(xs, vals, width=0.62, color=SERIES[0], zorder=3)
        ax.axhline(base[metric], color=MUTED, linewidth=1, linestyle="--", zorder=2)
        for x, v in zip(xs, vals):
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(list(xs), [LABELS.get(q, q) for q in configs], fontsize=8.5)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.25, 0.02)
        ax.set_ylim(lo - pad, hi + pad * 2)
        style(ax, ylabel)

    quant_configs = [q for q in configs if q != "none"]
    for i, q in enumerate(quant_configs):
        pts = [(base["samples"][s]["motion"], base["samples"][s]["psnr"] - runs[q]["samples"][s]["psnr"])
               for s in base["samples"] if s in runs[q]["samples"]]
        ax_m.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, color=SERIES[1 + i % 3],
                     label=LABELS.get(q, q).replace("\n", " "), zorder=3)
    ax_m.axhline(0, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax_m.set_xlabel("clip motion score (mean |frame diff|)", color=MUTED)
    if quant_configs:
        ax_m.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    style(ax_m, "PSNR loss vs baseline (dB)")

    fig.suptitle(f"INT8 PTQ quality — {tok_name} on DAVIS val", color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.918, "fake-quant (modelopt) · quality only, not deployed-int8 speed · dashed = native bf16 baseline",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output = Path(args.output) if args.output else results_dir / "ptq_quality.png"
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
