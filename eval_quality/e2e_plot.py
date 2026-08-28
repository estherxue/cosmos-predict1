# SPDX-License-Identifier: Apache-2.0
"""Throughput vs quality for the end-to-end TensorRT configs (results_e2e/e2e.json)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

LABELS = {"pytorch_jit_bf16": "PyTorch bf16 (JIT)", "trt_fp16": "TRT fp16", "trt_fp16_opt": "TRT fp16 +graph/GN18",
          "trt_int8_full": "INT8 full VAE", "trt_int8_mixed": "INT8 mixed", "trt_mixed_h16": "INT8 mixed (fp16 export)",
          "trt_mixed2_h16": "INT8 mixed, enc down.0 int8", "trt_encmix_decfull_o5": "INT8 dec-full + enc-mixed (opt5)",
          "trt_mixed_o5": "INT8 mixed (opt5)", "trt_fp16_convresize": "TRT fp16 + graph rewrite",
          "trt_cr_encmix_decfull": "INT8 dec-full + enc-mixed + graph rewrite",
          "trt_cr_encmix_decfull_o5": "INT8 dec-full + enc-mixed + graph rewrite (opt5)",
          "trt_fused_encmix_decfull": "norm rewrite (broken fp16)", "trt_fp16_fused": "fp16 norm rewrite (broken)",
          "trt_ncr_encmix_decfull": "norm rewrite v2 (broken)"}
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
OK, BAD, REF = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", default="eval_quality/results_e2e/e2e.json")
    args = parser.parse_args()
    rows = json.loads(Path(args.e2e).read_text())
    base = next(r for r in rows if r["tag"] == "trt_fp16")
    floor = base["psnr"] - 1.0
    target = base["clips_per_s"] * 1.3

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    ax.axhline(floor, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.axvline(target, color=MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.annotate("−1 dB floor", (base["clips_per_s"] * 0.42, floor), textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
    ax.annotate("+30% vs TRT fp16", (target, floor + 0.1), textcoords="offset points", xytext=(4, 0), fontsize=8, color=MUTED, rotation=90, va="bottom")
    SKIP = {"trt_mixed_h16", "trt_mixed_o5", "trt_cr_encmix_decfull", "trt_fp16_fused", "trt_ncr_encmix_decfull",
            "trt_fp16_convresize"}  # near-duplicates / broken variants — see e2e.json for all rows
    OFFSETS = {"trt_fp16": (-8, -14), "trt_fp16_opt": (6, 6), "trt_int8_mixed": (-8, 8),
               "trt_encmix_decfull_o5": (-8, -14), "trt_cr_encmix_decfull_o5": (8, 6),
               "trt_int8_full": (8, -12), "trt_mixed2_h16": (-8, -14), "trt_fused_encmix_decfull": (8, 4)}
    HA = {"trt_fp16": "right", "trt_mixed2_h16": "right", "trt_int8_mixed": "right", "trt_encmix_decfull_o5": "right"}
    for r in rows:
        if r["tag"] in SKIP:
            continue
        ref = r["tag"] in ("pytorch_jit_bf16", "trt_fp16", "trt_fp16_opt", "trt_fp16_convresize")
        color = REF if ref else (OK if r["psnr"] >= floor else BAD)
        ax.scatter(r["clips_per_s"], r["psnr"], s=64, color=color, zorder=3)
        label = LABELS.get(r["tag"], r["tag"]) + (" (3 variants)" if r["tag"] == "trt_int8_mixed" else "")
        ax.annotate(label, (r["clips_per_s"], r["psnr"]), textcoords="offset points",
                    xytext=OFFSETS.get(r["tag"], (6, 4)), ha=HA.get(r["tag"], "left"), fontsize=7.5, color=INK)
    ax.set_xlabel("throughput (clips/s, 480x854x17, batch 1, RTX 4090)", color=MUTED)
    ax.set_ylabel("PSNR (dB) on DAVIS val", color=MUTED)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_facecolor(SURFACE)
    fig.suptitle("End-to-end tokenizer (encoder→decoder): throughput vs quality", color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(0.02, 0.915, "green = references · blue = within 1 dB · orange = quality fail · same engines measured for both axes",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.e2e).parent / "e2e_qps_vs_psnr.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
