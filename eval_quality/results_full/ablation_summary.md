# Cosmos-0.1 video tokenizer sweep — ablation summary

Setup: DAVIS 2017 val (30 clips × 17 frames @480p), RTX 3090, bf16, one causal window.
Full per-variant table in `summary.csv`; per-clip and per-frame data in `metrics.json`.

## Temporal vs spatial compression: a clean dissociation

PSNR drop when doubling one compression axis, split by clip motion (mean |frame diff|,
15 low / 15 high clips); rho = Spearman(motion, per-clip drop), n=30.

| Ablation (axis doubled)     | low-motion | high-motion | rho    |
|-----------------------------|-----------|-------------|--------|
| CV 4x8x8 → 8x8x8 (temporal) | +1.00 dB  | +1.76 dB    | +0.734 |
| DV 4x8x8 → 8x8x8 (temporal) | +0.39 dB  | +0.88 dB    | +0.362 |
| CV 8x8x8 → 8x16x16 (spatial)| +2.14 dB  | +2.17 dB    | −0.049 |
| DV 8x8x8 → 8x16x16 (spatial)| +1.85 dB  | +1.49 dB    | −0.397 |

**Temporal compression cost scales with motion** (high-motion clips pay ~1.8× more;
per-clip drops span 0.27–3.43 dB, a >12× spread ordered by motion score).
**Spatial compression cost is motion-independent** (identical group means, rho ≈ 0) —
the control condition that makes the temporal result meaningful. The negative DV-spatial
rho is a floor effect: discrete high-motion clips are already degraded at 8x8x8.

## Per-frame PSNR shows the causal block structure

Frame 0 is intra-coded and reconstructs 2–4 dB above the rest; within the window,
quality oscillates with period = temporal factor (peaks at frames 3/7/11/15 for 4x,
7/15 for 8x). Per-frame curves are stored in metrics.json (`psnr_frames`).

## Continuous vs discrete (grouped, not a bug)

DV sits 2.1–3.3 dB below CV at every compression level — the FSQ-quantization cost,
reported as a separate series.

## Precision note

The released encoder/decoder JIT graphs are bf16-locked: casting to fp32 or fp16
fails inside serialized TorchScript casts. A precision/PTQ ablation requires the
PyTorch-native model path (`--mode torch` checkpoints), left as future work.
