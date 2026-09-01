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

ORDER = ["none", "w8_dec", "w8a8_dec", "w8a8_dec_mixed", "w8a8_all"]
HIDE = {"w8a8_dec_sq"}  # identical to plain W8A8 (no-op on conv nets) — text-only result
LABELS = {"none": "bf16", "w8_dec": "W8 dec", "w8a8_dec": "W8A8 dec",
          "w8a8_all": "full VAE", "w8a8_dec_sq": "+SQ", "w8a8_dec_mixed": "mixed"}
SCATTER_SKIP = {"w8a8_all", "w8a8_dec_sq"}  # off-scale / identical-to-plain — keep the scatter readable
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
    configs = [q for q in ORDER if q in runs] + sorted(set(runs) - set(ORDER) - HIDE)
    if "none" not in runs:
        raise SystemExit("missing baseline run (results_quant/none) — run quantize_ptq.py --quant none first")
    base = runs["none"]
    tok_name = base["name"]

    fig, (ax_p, ax_s, ax_l) = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE)

    # Dot plot, not bars: differences are small vs the absolute values, so a
    # non-zero axis is needed — honest with point markers, misleading with bars.
    xs = range(len(configs))
    for ax, metric, ylabel, fmt in ((ax_p, "psnr", "PSNR (dB)", "{:.2f}"), (ax_s, "ssim", "SSIM", "{:.3f}")):
        vals = [runs[q][metric] for q in configs]
        ax.axhline(base[metric], color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.scatter(list(xs), vals, s=64, color=SERIES[0], zorder=3)
        for x, (q, v) in zip(xs, ((q, runs[q][metric]) for q in configs)):
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, color=INK)
            if q != "none":
                delta = v - base[metric]
                ax.annotate(f"{delta:+.2f}" if metric == "psnr" else f"{delta:+.3f}",
                            (x, v), textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(list(xs), [LABELS.get(q, q) for q in configs], fontsize=9)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.18, 0.02)
        ax.set_ylim(lo - pad * 1.6, hi + pad * 1.6)
        style(ax, ylabel + "  (axis not from zero)")

    # Latency panel: measured decoder TRT engines (results_e2e/bench.json; encoder is
    # unquantized in these configs except full VAE — see footnote).
    bench = {(r["tag"], r["onnx"].split("/")[-1]): r["median_ms"]
             for r in json.loads(Path("eval_quality/results_e2e/bench.json").read_text())}
    LAT = {"none": bench.get(("fp16", "decoder_h16.onnx")),
           "w8a8_dec": bench.get(("int8", "decoder_full.onnx")),
           "w8a8_dec_mixed": bench.get(("int8", "decoder_mixed.onnx")),
           "w8a8_all": bench.get(("int8", "decoder_full.onnx"))}
    lat_cfgs = [q for q in configs if LAT.get(q)]
    lx = [configs.index(q) for q in lat_cfgs]
    lv = [LAT[q] for q in lat_cfgs]
    ax_l.axhline(LAT["none"], color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax_l.scatter(lx, lv, s=64, color=SERIES[0], zorder=3)
    for x, q, v in zip(lx, lat_cfgs, lv):
        ax_l.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
        if q != "none":
            ax_l.annotate(f"{100*(v-LAT['none'])/LAT['none']:+.0f}%", (x, v), textcoords="offset points",
                          xytext=(0, -14), ha="center", fontsize=7.5, color=MUTED)
    if "w8_dec" in configs:
        ax_l.annotate("no engine\n(weights-only)", (configs.index("w8_dec"), LAT["none"]),
                      textcoords="offset points", xytext=(0, -26), ha="center", fontsize=7.5, color=MUTED)
    ax_l.set_xticks(list(range(len(configs))), [LABELS.get(q, q) for q in configs], fontsize=9)
    ax_l.set_xlim(-0.6, len(configs) - 0.4)
    lo, hi = min(lv), max(lv)
    pad = max((hi - lo) * 0.25, 0.5)
    ax_l.set_ylim(lo - pad * 1.6, hi + pad * 1.6)
    style(ax_l, "decoder engine latency, ms/clip  (axis not from zero)")

    fig.suptitle(f"INT8 PTQ quality — {tok_name} on DAVIS val", color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.918, "quality: fake-quant (modelopt) · latency: measured decoder TRT engines, 17-frame clip @480p, RTX 4090, CUDA graph · "
             "encoder unquantized in these configs (full VAE also quantizes it: 21.6→18.5 ms) · dashed = bf16 baseline",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output = Path(args.output) if args.output else results_dir / "ptq_quality.png"
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
