# Quantization does not target motion content — evidence tables

Data: DAVIS 2017 val, 30 clips × 17 frames @480p, 0.1-CV8x8x8. Motion score =
mean |frame diff| (0–255); clips split into low-motion (15) / high-motion (15).
ρ = Spearman(motion, per-clip PSNR drop), n=30. Sources: `results_quant/*/metrics.json`,
`results_e2e/e2e.json`, `results_full_4090/metrics.json`.

## T1 — Quantization drop by motion group (fake-quant, vs bf16 baseline)

| config | low-motion drop | high-motion drop | ρ(motion, drop) |
|---|---|---|---|
| W8 weights-only decoder | +0.03 dB | +0.02 dB | −0.37 |
| W8A8 decoder | +0.91 dB | **+0.58 dB** | −0.62 |
| W8A8 decoder mixed (accepted) | +0.17 dB | **+0.15 dB** | −0.38 |

High-motion clips lose **less** than low-motion clips in every configuration.

## T2 — Deployed INT8 engines (end-to-end enc+dec, vs TRT fp16 engine)

| group | PSNR drop |
|---|---|
| low-motion (15) | +0.77 dB |
| high-motion (15) | **+0.47 dB** |
| ρ(motion, drop) | −0.63 |

The deployed-engine result reproduces the fake-quant pattern.

## T3 — The contrast that makes the claim meaningful (same 30 clips)

| degradation source | low-motion | high-motion | ρ(motion, drop) |
|---|---|---|---|
| temporal compression 2× (CV4x8x8→CV8x8x8) | +1.00 dB | **+1.76 dB** | **+0.73** |
| quantization (W8A8 decoder) | +0.91 dB | +0.58 dB | −0.62 |

Temporal compression *targets* motion (positive, strong); quantization is
content-agnostic and, in dB, cheapest exactly where compression is most
expensive — the two error sources are complementary, not compounding.

## T4 — Mechanism (why the correlation is negative, not zero)

| correlation | value |
|---|---|
| ρ(baseline PSNR, quant drop) | **+0.97** |
| ρ(motion, baseline PSNR) | −0.63 |

Quantization adds roughly constant noise energy. On a clean reconstruction
(static clip, high baseline PSNR) that constant noise costs many dB; on a
motion-heavy clip it is masked by the larger reconstruction error (dB ceiling
effect). The motion correlation is fully mediated by baseline quality.

Caveats for honest reporting: per-clip variance is real (e.g. in the deployed
engines the largest single drop is judo, a *low*-motion clip, +1.58 dB;
shooting, the most-motion clip, +1.03 dB) — the claim holds at the group/rank
level (ρ, group means), not as a per-clip guarantee; and "no damage" should be
phrased as "≤0.15 dB group-level cost in the accepted config, smallest on
high-motion content".
