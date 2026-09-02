# SPDX-License-Identifier: Apache-2.0
"""Plot INT8 PTQ quality + latency, all measured on deployed TensorRT engines.

Every config is a real engine pair (encoder + decoder, fp16 IO, opset 18, no
graph rewrite, batch 1, CUDA graph): PSNR/SSIM come from the end-to-end run in
results_trtfig/fig_e2e.json, latency is the same run's end-to-end engine time —
one artifact per point, no fake-quant/engine mixing (decoder-engine-only ms
lives in fig_bench.json for the record). A PyTorch JIT bf16 reference on identical inputs is quoted in the
subtitle.

Usage:
    python eval_quality/quant_plot.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# (e2e tag, label). "dec mixed" = decoder keeps conv_out/norm_out/up.0 in fp16;
# "enc mix+dec full" = the deployed-style config (encoder keeps patcher/conv_in/down.0,
# decoder fully int8) — a DIFFERENT thing than "dec mixed" despite the shared word.
ORDER = [
    ("fig_f16", "fp16"),
    ("fig_w8dec", "W8 dec"),
    ("fig_w8a8dec", "W8A8 dec"),
    ("fig_mixed", "dec mixed"),
    ("fig_encmix_decfull", "enc mix\n+dec full"),
    ("fig_fullvae", "full VAE"),
]
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
    parser.add_argument("--trt_dir", default="eval_quality/results_trtfig")
    parser.add_argument("--output", default="eval_quality/results_quant/ptq_quality.png")
    args = parser.parse_args()

    trt_dir = Path(args.trt_dir)
    e2e = {r["tag"]: r for r in json.loads((trt_dir / "fig_e2e.json").read_text())}
    ref = e2e["pytorch_jit_bf16"]
    base = e2e[ORDER[0][0]]

    fig, (ax_p, ax_s, ax_l) = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=SURFACE)
    xs = list(range(len(ORDER)))
    labels = [label for _, label in ORDER]

    # Dot plot, not bars: differences are small vs the absolute values, so a
    # non-zero axis is needed — honest with point markers, misleading with bars.
    for ax, metric, ylabel, fmt, dfmt in ((ax_p, "psnr", "PSNR (dB)", "{:.2f}", "{:+.2f}"),
                                          (ax_s, "ssim", "SSIM", "{:.3f}", "{:+.3f}")):
        vals = [e2e[tag][metric] for tag, _ in ORDER]
        ax.axhline(base[metric], color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.scatter(xs, vals, s=64, color=SERIES[0], zorder=3)
        for x, ((tag, _), v) in enumerate(zip(ORDER, vals)):
            ax.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, color=INK)
            if tag != ORDER[0][0]:
                ax.annotate(dfmt.format(v - base[metric]), (x, v), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(xs, labels, fontsize=9)
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.18, 0.02)
        ax.set_ylim(lo - pad * 1.6, hi + pad * 1.6)
        style(ax, ylabel + "  (axis not from zero)")

    # End-to-end latency (encoder engine + decoder engine, one chained run): encoders
    # differ across configs, so decoder-only ms would show three identical points.
    lv = [e2e[tag]["median_ms"] for tag, _ in ORDER]
    ax_l.axhline(lv[0], color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax_l.scatter(xs, lv, s=64, color=SERIES[0], zorder=3)
    for x, v in zip(xs, lv):
        ax_l.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=8.5, color=INK)
        if x:
            ax_l.annotate(f"{100 * (v - lv[0]) / lv[0]:+.0f}%", (x, v), textcoords="offset points",
                          xytext=(0, -14), ha="center", fontsize=7.5, color=MUTED)
    ax_l.set_xticks(xs, labels, fontsize=9)
    ax_l.set_xlim(-0.6, len(ORDER) - 0.4)
    lo, hi = min(lv), max(lv)
    pad = max((hi - lo) * 0.25, 0.5)
    ax_l.set_ylim(lo - pad * 1.9, hi + pad * 1.6)
    style(ax_l, "e2e latency, ms/clip  (axis not from zero)")

    fig.suptitle("INT8 PTQ — 0.1-CV8x8x8 on DAVIS val, measured on TensorRT engines",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.925, "quality AND latency from the same deployed engines (fp16 IO, opset 18, no graph rewrite, "
             "17-frame clip @480p, batch 1, CUDA graph, RTX 4090, TRT 10.16 cu12) · dashed = fp16 engine baseline · "
             f"PyTorch JIT bf16 reference on identical inputs: {ref['psnr']:.2f} dB / {ref['ssim']:.3f} / {ref['median_ms']:.0f} ms",
             color=MUTED, fontsize=8)
    fig.text(0.02, 0.895, "dec mixed: decoder int8 except conv_out/norm_out/up.0 · enc mix+dec full: deployed-style — encoder int8 except "
             "patcher/conv_in/down.0, decoder fully int8 (the earlier −0.62 dB config, here without its graph rewrite/opt5)",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    output = Path(args.output)
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
