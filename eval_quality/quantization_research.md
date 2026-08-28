# Quantizing the Cosmos tokenizer (VAE) — research notes

## RESULTS (2026-08-27, 0.1-CV8x8x8, DAVIS val 30 clips, RTX 3090, modelopt 0.46 fake-quant)

| config | PSNR | SSIM | ΔPSNR | verdict |
|---|---|---|---|---|
| bf16 native (baseline) | 28.13 | 0.787 | — | parity with JIT confirmed (bit-identical metrics) |
| W8 decoder (weights-only) | 28.11 | 0.786 | 0.02 | ✅ free |
| W8A8 decoder | 27.38 | 0.738 | 0.75 | borderline |
| W8A8 dec + SmoothQuant | 27.38 | 0.738 | 0.75 | no-op on conv nets (Linear-oriented) |
| **W8A8 decoder mixed** (conv_out, norm_out, up.0 kept bf16) | **27.97** | **0.782** | **0.16** | ✅ **accepted** |
| W8A8 full VAE | 19.48 | 0.570 | 8.65 | ❌ encoder activations unquantizable |

Sensitivity scan (10-clip subset): damage is *distributed* across the decoder —
single-component recovery: head +0.16 dB, up.0 +0.21, attn/mid ≈ 0; the
head+up.0 combo recovers 0.75 → 0.16 dB on full val.

Correction (earlier note claimed motion-independence — wrong): per-clip W8A8
damage correlates *negatively* with motion (ρ=−0.62), but this is almost fully
mediated by baseline quality — ρ(baseline PSNR, drop) = **0.97**. Quantization
adds roughly constant noise energy; on a nearly-perfect reconstruction (static
clip, high baseline PSNR) that constant noise costs many dB, on an already-noisy
one it is masked (dB ceiling effect). So the mechanism contrast with temporal
compression stands, but the correct statement is: temporal compression targets
motion content; quantization noise is content-agnostic and its *dB* cost tracks
how clean the reconstruction was.

## Phase 2 RESULTS — deployed int8 speed (TensorRT 10.16, RTX 4090)

Decoder CV8x8x8, (1,16,3,60,108) latent → (1,3,17,480,864) video, median of 30 runs
(`results_quant/trt_bench.json`):

| engine | median ms | ms/frame | speedup |
|---|---|---|---|
| TRT fp16 | 39.26 | 2.31 | 1× (reference) |
| TRT int8 (all-conv Q/DQ) | 34.64 | 2.04 | **1.13×** |

For scale: PyTorch (JIT, bf16) decode on the same 4090 is 7.6 ms/frame — the TRT
runtime alone is worth 3.3×; int8 adds a further 13%. Pairing honestly: the
benchmarked int8 engine quantizes *all* decoder convs, which corresponds to the
plain W8A8 quality row (−0.75 dB); the accepted mixed config (−0.16 dB) would land
between 2.04 and 2.31 ms/frame.

### Why only 13%: per-layer profile verdict (profile_{fp16,int8}.json)

EngineInspector + IProfiler per-layer breakdown (equivalent of
`trtexec --profilingVerbosity=detailed --dumpProfile`):

| | fp16 engine (38.2 ms) | int8 engine (33.2 ms) |
|---|---|---|
| layers running int8 | — | **6.9 ms (21%)** |
| fp16 layers | 33.6 ms (88%) | 18.1 ms (55%) |
| fp32 layers | 4.4 ms (12%) | **8.0 ms (24%)** — Q/DQ broke fusions, segments fell back |
| reformat/cast | 6.4 ms | **8.3 ms** — Q/DQ conversion tax |

So the limited gain is (in order): **(1) coverage** — only ~21% of int8-engine
runtime actually executes in int8; norm/SiLU/attention/reformats (~2/3 of time,
memory-bound elementwise) are untouched by design; **(2) Q/DQ overhead** — casts
grew ~1.9 ms and some inter-Q/DQ segments dropped to fp32, eating part of the
conv win. Amdahl check: convs are 49% of the fp16 engine; perfect 2× int8 convs
would cap the speedup at ~1.33×, and we realized 1.15× of that. Next levers, in
order of expected value: quantize the wavelet ConvTranspose ops, reduce Q/DQ
boundary count (quantize longer chains), and fuse the GroupNorm+SiLU pointwise
chains — not more conv quantization.

4090 re-timing of the 6-variant sweep is in `results_full_4090/` (quality metrics
reproduce the 3090 run to ~1e-3 dB; timings ~1.7× faster across the board).

### Phase-2 pitfalls (all hit, all worked around)
- torch 2.8.0+cu128: large-tensor conv3d → CUDA illegal access (small convs fine); fixed by torch 2.4.1+cu124.
- Native eager *encoder* faults on 4090 at 480p even on 2.4.1 (fine on 3090) → calibrate via the JIT encoder.
- UnPatcher3D builds wavelet conv kernels from `x.shape` in forward → ONNX "kernel of unknown shape"; fixed by pre-caching kernels (eager warmup) so tracing sees constants.
- pip `tensorrt` now installs TRT 11 (weak-typing flags removed) → pin `tensorrt==10.*`.
- TRT 10 Pad+Conv3d tactic search needs a large workspace: 8 GB fails on the first conv, 18 GB builds.

Goal: an "int8 ms/frame gain vs PSNR loss" result for one variant (target: CV8x8x8),
PTQ first, QAT only if PTQ quality is unacceptable. Timing hardware is locked to
RTX 3090 (Ampere, sm_86): **INT8 yes, FP8 no** (FP8 needs Ada/Hopper).

## Why the bf16-locked JIT is not a blocker

`load_encoder_model/load_decoder_model` in `cosmos_predict1/tokenizer/inference/utils.py`
have a native path: passing a `tokenizer_config` (from `TokenizerConfigs`) builds the
PyTorch-native module and loads weights via `torch.jit.load(...).state_dict()` with
`strict=True` — **the JIT files double as the weight store**. So a native fp32/bf16
`nn.Module` with pretrained weights is fully recoverable from the checkpoints we
already have. The earlier fp32/fp16 failures were an artifact of the serialized
TorchScript graph only.

Op inventory (from `modules/layers3d.py`): CausalConv3d (wrapping `nn.Conv3d`),
GroupNorm (`CausalNormalize`), conv-projection attention, no `nn.Linear`.
A conv-heavy network → LLM-oriented tools (bitsandbytes, GPTQ/AWQ, torchao's
linear-focused paths) are out.

## Toolchain choice: NVIDIA TensorRT Model Optimizer (`nvidia-modelopt`)

- Supports the module set we need: QuantConv3d / QuantConvTranspose3d exist in
  NVIDIA's quantization stack; modelopt's PTQ covers INT8 (default + SmoothQuant
  configs) and QAT under one API.
- Workflow: `mtq.quantize(model, INT8_DEFAULT_CFG, calibration_forward_loop)` →
  fake-quant model in PyTorch (quality measurable immediately with our existing
  run_eval harness) → optional export to TensorRT for real speed.
- Diffusers has an official ModelOpt integration; NVIDIA used the same stack for
  SDXL INT8.

## Key precedent — this experiment has a real question in it

In SD/SDXL INT8 pipelines the U-Net/DiT is quantized but **the VAE is deliberately
kept at fp16/fp32** — the decoder is regarded as quantization-sensitive (numerical
instability, detail loss, color artifacts). Whether a *video* VAE decoder survives
INT8, and whether sensitivity concentrates in specific blocks (output convs,
GroupNorm-adjacent layers), is exactly what our per-clip/per-frame harness can
answer quantitatively.

## Plan (phased, each phase is a complete result)

**Phase 0 — native-path parity (gate, ~30 min GPU)**
1. Load CV8x8x8 natively (config + JIT state_dict), run bf16 on DAVIS val via
   run_eval; PSNR must match the JIT baseline (28.13 dB) to ~0.01 dB.
2. Risk: our sweep used 0.1-series weights but `TokenizerConfigs` lists Tokenize1
   names. `strict=True` load answers this instantly; fallbacks: (a) quantize
   Tokenize1-CV8x8x8-720p instead (its config officially exists; new bf16 baseline
   row needed), or (b) the 0.1 HF repos ship `model_config.yaml` — use that as the
   native config.
3. Bonus once native works: the fp32-vs-bf16 precision row we couldn't get from JIT.

**Phase 1 — INT8 PTQ, fake-quant quality (the deliverable row's quality half)**
- Calibrate on DAVIS *train* sequences (never val — that's our eval set), ~16-32 clips.
- Three configs, weakest first: W8-only decoder; W8A8 decoder (encoder bf16);
  W8A8 full VAE. Report ΔPSNR/ΔSSIM on DAVIS val with the existing harness,
  reusing the motion split (is quantization damage also motion-correlated?).
- If W8A8 craters: SmoothQuant config, then per-layer sensitivity scan
  (disable quantizers block by block) — "which blocks kill the VAE" is a
  publishable-quality mini-result on its own.

**Phase 2 — real speed via TensorRT (the row's ms/frame half)**
- Fake-quant in PyTorch gives no speedup; real INT8 needs a TRT engine.
- Fair comparison: TRT-fp16 engine vs TRT-int8 engine (same runtime), not
  PyTorch-bf16 vs TRT-int8.
- Risk: ONNX export of CausalConv3d (custom padding — should export as Pad+Conv)
  and the attention block; TRT supports 3D conv INT8 on Ampere with possible
  per-layer fallbacks.

**Decision rule for "loss too big" (pre-committed before seeing results)**

Reference scale: doubling temporal compression costs 1.38 dB on this data — a
quantization tax should be well below one compression step, or you'd rather just
compress harder instead.

| ΔPSNR vs native-bf16 | verdict | action |
|---|---|---|
| ≤ 0.3 dB (and ΔSSIM ≤ 0.01) | acceptable | ship the config |
| 0.3 – 1.0 dB | borderline | SmoothQuant (`w8a8_dec_sq`); still >0.3 → per-block sensitivity scan, keep sensitive blocks bf16 |
| > 1.0 dB | PTQ insufficient | QAT (Phase 3) |

**Phase 3 — QAT (only if PTQ quality is unacceptable)**
- modelopt QAT = fine-tune the fake-quant model; decoder-only, frozen encoder,
  reconstruction loss on DAVIS train, few thousand steps. The repo's
  `tokenizer/training/` has the loss/training scaffolding to borrow from.

## Sources

- [TensorRT Model Optimizer announcement](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/) / [v0.15 release notes](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)
- [pytorch-quantization toolkit docs (QuantConv3d/QuantConvTranspose3d)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/pytorch-quantization-toolkit/docs/index.html)
- [ModelOpt PyTorch quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html) / [config.py](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/config.py)
- [Diffusers × ModelOpt integration](https://huggingface.co/docs/diffusers/quantization/modelopt)
- [NeMo SDXL INT8 quantization guide](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/multimodal/text2img/sdxl_quantization.html) (U-Net quantized, VAE kept at native precision)
- [Quanto + Diffusers blog](https://huggingface.co/blog/quanto-diffusers) (VAE excluded from quantization for stability)

## Phase 3 — encoder+decoder INT8, end-to-end TRT (goal: ≥+30% QPS, ≤1 dB) — in progress

Measured with `trt_eval.py` (same engines give PSNR and clips/s; DAVIS val, 480x854x17,
RTX 4090, TRT 10.16, CUDA-graph replay):

| config | PSNR | ms/clip | clips/s | vs TRT-fp16 |
|---|---|---|---|---|
| PyTorch JIT bf16 | 28.10 | 185.3 | 5.40 | 0.33× |
| TRT fp16 (baseline) | 28.11 | 61.4 | 16.29 | 1.00× |
| TRT fp16 + fp16 IO + opset18 GN + CUDA graph | 28.11 | 59.7 | 16.74 | 1.03× |
| INT8 full VAE (torch-side Q/DQ) | 19.24 ❌ | 52.3 | 19.12 | 1.17× |
| INT8 mixed (enc: patcher/conv_in/down.0 fp16; dec: conv_out/norm_out/up.0 fp16) | 27.99 ✅ (−0.12) | 56.8 | 17.61 | 1.08× |

Findings so far: (1) pure-fp16 graph/fusion levers are worth only ~3% — TRT's fp16 graph
is already near its floor; (2) even *full* INT8 is only 1.17×, so the ceiling is set by
non-conv ops + Q/DQ boundary costs, not conv coverage; (3) batching is not a lever:
fp16 decoder at batch 4 is 16% *slower per frame* than batch 1 (2.48 vs 2.14 ms/frame),
INT8 batch 4/8 exports OOM in the fp32 trace — the GPU is saturated at batch 1
(consistent with a bandwidth-bound decoder; partial evidence, per the 30-min cap).
Engineering pitfalls this round: modelopt.onnx's preprocessing produced a cyclic graph
for the encoder (raw ONNX is acyclic) → switched to torch-side Q/DQ export; that export
must use the TorchScript exporter (`dynamo=False`) since modelopt's Q/DQ symbolics are
not registered for the dynamo exporter (modelopt's install silently upgraded torch);
and a `pgrep -f` self-match idled the GPU queue for 30 min.

Sequence-length probe (49 frames, agent-run, `results_probe49/`): temporal-compression
penalty is unchanged in direction and magnitude (+0.2 dB larger at 49f, ranking
preserved), so the 17-frame sweep is a faithful proxy.

### Phase 3 — final table (encoder+decoder INT8, end-to-end TRT, RTX 4090, batch 1, CUDA graph)

| config | PSNR (Δ) | ms/clip | clips/s | vs PyTorch bf16 | vs TRT fp16 |
|---|---|---|---|---|---|
| PyTorch bf16 JIT (shipped runtime) | 28.10 | 185.3 | 5.40 | 1.00× | — |
| TRT fp16 | 28.11 | 61.4 | 16.29 | 3.02× | 1.00× |
| TRT fp16 + fp16 IO/opset18 GN/CUDA graph | 28.11 | 59.7 | 16.74 | 3.10× | 1.03× |
| INT8 mixed (enc front + dec head/up.0 fp16), 3 variants | 27.99 (−0.12) | 56.3–56.8 | 17.6–17.8 | 3.29× | 1.09× |
| **INT8 dec-full + enc-mixed, opt level 5** | **27.49 (−0.62)** | **53.6** | **18.67** | **3.46×** | **1.15×** |
| INT8 full VAE (reference, quality fail) | 19.24 | 52.3 | 19.12 | 3.54× | 1.17× |
| INT8 mixed + enc down.0 int8 (quality fail) | 19.23 | 55.3 | 18.09 | — | — |

Dead levers, each measured: strongly-typed build (fp16 GroupNorm) — int8 engine fails to
build, fp16 decoder unchanged (36.35 vs 36.44 ms); builder opt level 5 — +0.5%; fp16
Q/DQ export — no change (the fp32 kernels are GroupNorm reductions inside myelin
fusions, not Q/DQ fallbacks); batch 4 — 16% slower per frame; encoder down.0 in int8
— catastrophic (the encoder's sensitivity lives in its highest-resolution level, not
just the wavelet front).

**Verdict.** Best quality-compliant configuration: encoder mixed + decoder fully INT8,
−0.62 dB, **18.67 clips/s = 3.46× the shipped PyTorch runtime** (+246%) and 1.15× an
already-TensorRT-fp16 deployment. Against the shipped runtime the ≥30% target is met
many times over; against TRT-fp16 it is not reachable with INT8 on this model —
full INT8 caps at 1.17× because ~2/3 of engine time is memory-bound GroupNorm/SiLU/
attention/reformat work that INT8 does not touch, and every fusion-level lever was
tested and found flat. Going further would need a custom fused GroupNorm+SiLU kernel/
plugin or model-level changes (2:4 sparsity with retraining, lower-res processing),
which are outside a quantization study.

### Phase 3 — FINAL (after export-time graph rewrites; pod stopped after this run)

Export-time rewrites (`export_patches.py`, upstream untouched, verified numerically
equivalent in eager: max|diff| = 0): CausalConv3d's repeat+cat+pad → replicate Pad with
spatial zero-padding folded into the conv attribute; repeat_interleave upsampling →
Resize. The GroupNorm rewrite (transpose-free per-frame stats) is exact in fp32 but
breaks under fp16 in the traced graph (PSNR 12) — dropped.

| config | PSNR (Δ) | ms/clip | clips/s | vs PyTorch bf16 | vs TRT fp16 |
|---|---|---|---|---|---|
| PyTorch bf16 JIT (shipped runtime) | 28.10 | 185.3 | 5.40 | 1.00× | 0.33× |
| TRT fp16 | 28.11 | 61.4 | 16.29 | 3.02× | 1.00× |
| TRT fp16 + graph rewrite | 28.11 | 60.7 | 16.47 | 3.05× | 1.01× |
| INT8 dec-full + enc-mixed | 27.49 (−0.62) | 53.6 | 18.67 | 3.46× | 1.15× |
| **INT8 dec-full + enc-mixed + graph rewrite, opt level 5** | **27.49 (−0.62)** | **50.5** | **19.80** | **3.67×** | **1.22×** |

The graph rewrite is worth ~1% on the fp16 engine but ~6% on the INT8 engine: the
removed concat/expand/reformat nodes sat exactly at Q/DQ boundaries, where each one
cost an extra int8↔fp16 relayout.

**Final statement (baseline = shipped PyTorch runtime, per the agreed framing):**
encoder+decoder INT8 PTQ (mixed precision on the encoder's wavelet front + first
level, full INT8 decoder) + TensorRT fp16/CUDA-graph + export-time graph rewrites give
**3.67× throughput (19.8 vs 5.4 clips/s at 480p×17f, RTX 4090) at −0.62 dB PSNR**.
Relative to an already-optimized TensorRT-fp16 deployment the same engine is 1.22×;
the INT8-only ceiling on this memory-bound 3D-conv VAE is ~1.17× and the graph
rewrite adds the rest. All numbers come from one artifact measured for both axes
(`results_e2e/e2e.json`, `e2e_qps_vs_psnr.png`).
