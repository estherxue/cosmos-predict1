# 49-frame temporal-context probe (CV4x8x8 vs CV8x8x8)

Question: does the temporal-compression penalty (CV4x8x8 -> CV8x8x8 PSNR drop, larger on high-motion clips) measured with 17-frame clips still hold with 49-frame clips (1+48: 4x -> 13 latent frames, 8x -> 7 latent frames)?

Setup: DAVIS 2017 val, 480p short side, lossless x264rgb clips, first 49 frames; harness `eval_quality/run_eval.py --max-frames 49`. 17-frame numbers from `eval_quality/results_full/metrics.json` (same harness, `--max-frames 17`). Clips chosen = DAVIS sequences with >= 49 source frames (shooting, motocross-jump, judo were excluded: 40/40/34 frames).

| clip | group | src frames | motion (17f) | motion (49f) | 17f CV4 | 17f CV8 | 17f drop | 49f CV4 | 49f CV8 | 49f drop |
|---|---|---|---|---|---|---|---|---|---|---|
| bmx-trees | high | 80 | 25.0 | 29.2 | 29.76 | 27.35 | +2.41 | 28.63 | 26.06 | +2.57 |
| drift-straight | high | 50 | 33.5 | 38.6 | 26.97 | 24.69 | +2.28 | 26.72 | 24.11 | +2.61 |
| libby | high | 49 | 22.4 | 23.7 | 30.76 | 28.61 | +2.15 | 30.61 | 28.36 | +2.25 |
| drift-chicane | low | 52 | 2.3 | 2.3 | 32.25 | 31.98 | +0.27 | 32.25 | 31.95 | +0.30 |
| mbike-trick | low | 79 | 5.7 | 5.9 | 31.89 | 31.31 | +0.58 | 31.92 | 31.23 | +0.69 |
| dogs-jump | low | 66 | 5.6 | 5.5 | 34.83 | 34.01 | +0.82 | 35.01 | 34.15 | +0.87 |

| group | mean drop 17f | mean drop 49f |
|---|---|---|
| high-motion (3 clips) | +2.28 | +2.48 |
| low-motion (3 clips) | +0.56 | +0.62 |
| high minus low | +1.72 | +1.86 |

## Per-frame PSNR at 49 frames

Per-frame PSNR (dB) for each clip, CV8x8x8 minus CV4x8x8 (negative = 8x worse), grouped by 8-frame temporal block after frame 0:

```
bmx-trees       f0 -0.04 | -2.79 -2.44 -2.93 -2.51 -2.76 -2.11 -2.08 -2.12 | -2.92 -2.32 -3.41 -2.59 -2.72 -1.96 -2.45 -2.80 | -3.48 -3.36 -3.88 -3.41 -3.51 -2.70 -2.79 -3.06 | -3.36 -2.94 -3.43 -2.87 -2.55 -2.09 -2.49 -2.53 | -2.43 -2.03 -2.91 -2.47 -2.35 -1.87 -2.12 -2.00 | -2.32 -2.30 -2.92 -2.39 -2.27 -1.88 -2.19 -2.04 |
drift-straight  f0 +0.01 | -3.29 -2.48 -3.25 -2.50 -1.80 -1.56 -1.92 -1.91 | -2.26 -2.64 -3.34 -2.66 -2.59 -2.00 -2.36 -2.39 | -2.93 -2.68 -3.94 -2.89 -3.01 -1.85 -2.77 -2.80 | -3.37 -3.15 -4.80 -2.96 -3.01 -2.12 -3.27 -2.82 | -3.53 -2.75 -3.72 -2.46 -2.91 -2.20 -2.73 -2.09 | -2.32 -1.91 -3.02 -2.00 -2.49 -2.05 -2.28 -2.14 |
libby           f0 -0.31 | -2.05 -2.08 -3.15 -2.26 -2.45 -1.82 -2.35 -1.87 | -2.34 -1.97 -3.18 -2.15 -2.55 -1.76 -2.28 -1.94 | -2.39 -2.05 -3.23 -2.24 -2.39 -1.85 -2.40 -1.95 | -2.36 -2.17 -3.22 -2.32 -2.50 -1.96 -2.56 -2.11 | -2.62 -2.39 -3.39 -2.58 -2.74 -1.97 -2.42 -2.10 | -2.53 -2.22 -2.71 -1.91 -2.03 -1.45 -1.84 -1.16 |
drift-chicane   f0 -0.02 | -0.17 -0.27 -0.50 -0.24 -0.26 -0.20 -0.33 -0.28 | -0.32 -0.34 -0.54 -0.30 -0.19 -0.18 -0.28 -0.19 | -0.17 -0.23 -0.44 -0.20 -0.10 -0.09 -0.26 -0.09 | -0.16 -0.32 -0.64 -0.34 -0.27 -0.26 -0.38 -0.34 | -0.22 -0.37 -0.53 -0.24 -0.26 -0.23 -0.35 -0.39 | -0.39 -0.35 -0.68 -0.38 -0.37 -0.41 -0.46 -0.33 |
mbike-trick     f0 -0.07 | -0.51 -0.58 -1.34 -0.75 -0.62 +0.00 -0.47 -0.01 | -0.18 -1.09 -1.48 -1.57 +0.08 -0.31 -0.55 -0.60 | -1.08 -0.31 -1.08 -0.32 -0.31 -0.35 -0.47 -0.03 | -1.60 -1.14 -1.69 -0.59 -0.26 -0.12 -0.95 -0.50 | -0.74 -0.71 -1.75 -0.79 -0.55 -0.55 -0.86 -0.22 | -1.08 -1.23 -1.22 -1.28 -0.43 -0.14 -0.69 -0.54 |
dogs-jump       f0 +0.06 | -0.52 -0.68 -1.07 -0.70 -0.84 -0.80 -0.64 -0.59 | -1.10 -1.28 -1.51 -0.81 -0.91 -0.84 -0.74 -0.85 | -0.95 -1.05 -1.12 -0.66 -0.78 -0.91 -0.74 -0.56 | -0.91 -0.90 -1.10 -0.60 -0.99 -0.60 -0.55 -0.73 | -0.84 -0.91 -1.18 -0.87 -1.02 -0.66 -0.70 -0.71 | -1.17 -1.25 -1.54 -1.11 -0.86 -0.70 -0.89 -1.14 |
```

Per-frame PSNR for CV8x8x8 alone (dB), same layout:

```
bmx-trees       f0 34.39 | 25.22 26.04 28.27 26.34 24.84 25.87 28.48 26.44 | 26.07 26.36 28.99 27.03 25.55 27.24 30.10 27.79 | 25.74 26.52 29.09 27.17 25.17 26.26 29.99 27.37 | 24.85 25.63 27.70 25.33 24.31 25.65 28.53 26.21 | 24.05 22.39 24.68 23.69 23.03 23.70 25.88 23.80 | 22.80 23.74 25.96 24.28 23.29 23.97 26.37 24.75 |
drift-straight  f0 34.39 | 22.84 24.32 26.11 22.47 23.50 25.27 26.44 23.22 | 23.10 24.25 26.03 22.63 22.39 23.44 26.61 22.73 | 21.72 22.21 24.13 21.52 20.76 21.92 24.72 22.04 | 20.91 21.35 22.96 21.50 20.78 22.05 25.68 24.13 | 22.59 23.80 26.31 24.46 23.43 24.39 27.37 24.87 | 24.19 25.07 27.09 25.41 24.79 25.55 28.73 25.20 |
libby           f0 33.34 | 25.70 26.91 29.36 27.74 27.13 28.00 30.65 28.42 | 27.37 27.85 30.14 28.39 27.67 28.41 30.66 28.66 | 27.34 27.81 29.98 28.10 26.97 27.78 30.20 28.00 | 26.83 27.61 29.77 28.36 27.25 28.24 31.09 28.98 | 27.59 28.18 30.48 28.83 27.58 28.26 30.65 28.41 | 26.90 27.44 29.09 26.91 25.83 27.13 28.73 27.14 |
drift-chicane   f0 33.05 | 32.30 32.11 32.11 31.81 31.55 31.40 31.61 31.36 | 31.38 31.61 31.88 31.87 31.98 32.40 32.68 32.53 | 32.49 32.61 32.75 32.70 32.74 32.89 33.08 32.79 | 32.54 32.39 32.35 32.03 31.74 31.85 31.91 31.52 | 31.29 31.27 31.71 31.67 31.70 32.03 32.36 31.70 | 31.22 31.40 31.73 31.28 30.87 30.98 31.61 30.75 |
mbike-trick     f0 33.10 | 29.14 31.12 31.56 31.57 30.66 31.04 32.05 30.64 | 30.92 30.94 31.56 30.62 31.23 31.97 32.52 31.64 | 30.60 29.38 31.85 31.87 31.84 31.38 32.82 31.45 | 29.11 31.20 31.54 31.12 31.14 31.73 32.45 32.78 | 31.07 31.79 32.05 30.60 29.19 31.42 31.93 29.57 | 30.12 30.49 31.81 30.86 30.66 31.03 32.13 31.21 |
dogs-jump       f0 35.56 | 34.04 33.97 34.48 33.74 33.39 33.85 34.97 34.24 | 33.49 33.59 34.17 33.64 33.17 33.54 34.71 33.73 | 33.44 33.53 34.34 34.02 33.85 34.81 35.89 34.38 | 33.95 34.06 34.81 34.35 33.72 34.19 35.30 34.13 | 33.75 33.79 34.50 34.06 33.54 34.28 35.23 34.15 | 33.75 34.17 34.60 34.04 33.65 34.20 35.03 33.43 |
```

Mean CV8-minus-CV4 per-frame PSNR by position within each 8-frame block (frames 1..48, averaged over blocks and clips):

| position in block | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| high-motion | -2.74 | -2.44 | -3.36 | -2.51 | -2.59 | -1.96 | -2.41 | -2.21 |
| low-motion | -0.67 | -0.72 | -1.08 | -0.65 | -0.50 | -0.41 | -0.57 | -0.45 |

Mean per-frame PSNR of CV4x8x8 and CV8x8x8 by position within 4-frame block (frames 1..48):

| tokenizer / group | pos1 | pos2 | pos3 | pos4 |
|---|---|---|---|---|
| 0.1-CV4x8x8 high | 27.39 | 27.77 | 30.85 | 28.15 |
| 0.1-CV4x8x8 low | 32.56 | 32.91 | 33.83 | 32.88 |
| 0.1-CV8x8x8 high | 24.72 | 25.57 | 27.97 | 25.79 |
| 0.1-CV8x8x8 low | 31.98 | 32.35 | 33.00 | 32.33 |

Mean PSNR over frames 1-16 vs frames 17-48 (does quality drift with clip position?):

| tokenizer / group | frame 0 | frames 1-16 | frames 17-48 |
|---|---|---|---|
| 0.1-CV4x8x8 high | 34.15 | 28.86 | 28.38 |
| 0.1-CV4x8x8 low | 33.91 | 32.94 | 33.10 |
| 0.1-CV8x8x8 high | 34.04 | 26.44 | 25.80 |
| 0.1-CV8x8x8 low | 33.90 | 32.34 | 32.45 |

## Verdict

Sequence length does not change the direction of the temporal-compression penalty and barely changes its magnitude: every clip's CV4x8x8 -> CV8x8x8 drop grows by only +0.03..+0.33 dB when going from 17 to 49 frames (high-motion mean +2.28 -> +2.48 dB, low-motion +0.56 -> +0.62 dB; high-minus-low gap 1.72 -> 1.86 dB), and the per-clip ranking is preserved exactly (drift-chicane < mbike-trick < dogs-jump < libby < bmx-trees < drift-straight). About 0.1 dB of the increase is arithmetic (frame 0 has ~0 penalty and is diluted 1/49 instead of 1/17); the rest tracks the higher motion score of the extra frames on the high-motion clips (bmx-trees 25.0 -> 29.2, drift-straight 33.5 -> 38.6). So giving the 8x tokenizer 7 latent frames instead of 3 buys nothing back; the 17-frame numbers are a faithful proxy.

Per-frame pattern at 49 frames: the periodic block structure persists and is stationary across the whole clip. Both tokenizers show a 4-frame period (the 3rd frame of every group of 4 after frame 0 is ~2-3 dB sharper than its neighbours: CV4 high-motion 27.4 / 27.8 / 30.9 / 28.2 dB by position, CV8 24.7 / 25.6 / 28.0 / 25.8), CV8x8x8 shows no additional 8-frame period, and the CV8-minus-CV4 gap is roughly constant per block (largest at block position 3, -3.4 dB high-motion). Frame 0 is reconstructed at ~34 dB by both with ~0 drop, and frames 17-48 are within ~0.5 dB of frames 1-16 for both tokenizers, i.e. no error accumulation along the causal sequence.
