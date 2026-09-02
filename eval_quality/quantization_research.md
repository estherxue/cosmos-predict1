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

## Our DAVIS protocol vs the official one (not re-run; differences only)

| dimension | ours (`run_eval.py`, results_full*/results/) | official (paper Table 5 / `video_cli.py` defaults) |
|---|---|---|
| data | DAVIS 2017 **val, 30 seqs**, 480p JPEGs → lossless mp4 | DAVIS **1080p** (Full-Resolution); split not stated |
| resolution | short side 480 (854×480, padded to 864) | native 1080p (`short_size=None`, padded to 1088) |
| frames | **first 17 frames** only (one causal window) | whole sequence (34–104 f), `temporal_window=17` sliding, tail window temporally padded |
| precision / path | JIT, bf16, `CausalVideoTokenizer.forward` | same (CLI default bf16) — identical |
| PSNR | uint8 RGB, data_range 255, **per frame → mean** | TokenBench `metrics_cli.py`: **one MSE over the whole T×H×W×3 float32 array per video**, `20·log10(255/√mse)`, mean over videos (per-frame averaging is ≥ this by Jensen, so ours is if anything inflated) |
| SSIM | skimage, channel_axis=-1, win 7, per frame → mean | identical (same skimage call) |
| rFVD | not computed | StyleGAN-V I3D torchscript, ≤300 frames, short side 224 + center crop, FID over 50 video features |
| split / window | DAVIS **2017 val (30)**, window 17 | DAVIS **2016 (50 seqs)** inferred (Perazzi 2016 citation; AToken re-eval on "1080p, 50 videos" gets 32.25 vs paper 32.80); window **49** (0.1 & 360p) / **121** (720p) per Table 5's Frames column |
| official numbers | — | HF cards / README chart (35.28, DV 32.98) are **pre-fix** (uint8-overflow PSNR bug, TokenBench PR #3, 2024-12-30); paper Table 5 & project page are post-fix: 0.1-CV4x8x8 32.80, 0.1-CV8x8x8 30.61, T1-CV4x8x8-360p 35.85, T1-CV8x8x8-720p 31.28 |
| checkpoints | curves: 0.1 series; DAVIS main table: Tokenize1 | Table 5 reports both 0.1-CV (32.80/30.61) and Tokenize1 (35.85/31.28); the "disagreeing" 35.28 is the pre-fix bug value |
| quantized rows | fake-quant + TRT engines at 480p/17f | none |

Corrected gaps vs the post-fix official numbers: 0.1-CV4x8x8 29.51 vs 32.80 (−3.3),
0.1-CV8x8x8 28.13 vs 30.61 (−2.5), T1-CV4x8x8-360p 31.27 vs 35.85 (−4.6),
T1-CV8x8x8-720p 28.78 vs 31.28 (−2.5). Frame count (49-frame probe: −0.2 dB) and the
video-MSE PSNR convention both push *down*, so the remaining gap is essentially
**resolution (1080p vs 480p)**, plus one unspecified item: whether official metrics were
computed on H.264-recompressed outputs (`write_video`, qp 28). Relative
conclusions (compression-rate ordering, continuous > discrete, temporal-vs-spatial
ablation, quantization deltas) do not depend on these. A ready-to-run official-protocol
script exists (`eval_davis_official.py` + `davis_official_pipeline.sh`, ~1.5–2 h on a
4090) if absolute alignment is ever needed.


## Official-protocol anchor (run 2026-08-29, RTX 4090): our harness vs NVIDIA's DAVIS numbers

DAVIS 2016, all 50 sequences, native 1080p (1920×1080, padded to 1088), whole sequences,
official inference path (`CausalVideoTokenizer.forward`, bf16 JIT; code diff vs the archived
NVIDIA/Cosmos-Tokenizer repo = import paths only), TokenBench `metrics_cli.py` metrics
(video-level-MSE PSNR, per-frame SSIM). Temporal window 17 (= official CLI default) instead
of the paper's 49: 49 needs ~26 GB at 1080p (OOM on 24 GB); the 49-frame probe bounds the
window effect at ≤0.3 dB. The 4x8x8 models also needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(allocator fragmentation, 10 GB reserved-but-unallocated).

| model | official PSNR / SSIM | ours PSNR / SSIM | Δ | window |
|---|---|---|---|---|
| 0.1-CV4x8x8 | 32.80 / 0.900 | **32.42** / 0.893 (frame-avg PSNR 32.97) | -0.38 dB | 17 |
| 0.1-CV8x8x8 | 30.61 | **29.95** / 0.838 (frame-avg PSNR 30.57) | -0.66 dB | 17 |
| 0.1-DV4x8x8 | 28.81 / 0.818 | **27.14** / 0.768 (frame-avg PSNR 28.32) | -1.67 dB | 17 |

Verdict: the harness reproduces the post-bug-fix official numbers to within 0.4–0.7 dB
(video-MSE convention; the per-frame-average convention lands within 0.05–0.2 dB).
Attribution of the residual, with directions made explicit: our window 17 (vs official 49)
*favours us* (one intra-coded frame per 17 instead of per 49), so it cannot explain us being
lower. The coherent explanation is **input-side H.264 smoothing**: TokenBench's own
preprocessing writes sources to lossy mp4 (default qp≈28) and evaluates against that same
smoothed reference — reconstructing smoothed content is strictly easier than our
pristine-JPEG protocol. (Output-only recompression would push the official numbers *down*
and cannot explain the gap.) This also explains DV's larger residual (−1.67 dB): the FSQ
discrete model struggles most with high-frequency content and thus benefits most from a
smoothed input.
Our 480p/17-frame sweep numbers are therefore ~3 dB below official purely due to the 480p
evaluation resolution, not a pipeline defect. Files: `results_davis_official/*.json`.

### Recompression hypothesis — VERIFIED (qp28 experiment, 2026-08-29)

Re-running 0.1-CV4x8x8 with TokenBench-style preprocessing (H.264 round-trip of the
input, qp 28; the smoothed frames serve as both model input and PSNR reference,
everything else identical to our anchor):

| input handling | PSNR (video-MSE) | SSIM |
|---|---|---|
| pristine JPEG frames (our anchor) | 32.42 | 0.893 |
| **official PSNR (paper/page)** | **32.80** | **0.900** |
| qp28-smoothed input+reference | 33.21 | 0.917 |

Input smoothing alone is worth **+0.79 dB** — and the official number is now
*bracketed* by our two protocol variants. Conclusion: the residual anchor gap was
input preprocessing, not the harness; the harness is validated on both sides.
(Pod pitfalls this round, for the record: lean torch-2.4 image ships without
`unzip` **and** `ffmpeg`, and `apt-get install` silently no-ops before `apt-get
update`; my retry loop deleted a fully-downloaded 2.8 GB zip because the *unzip*
step failed — separate download and extraction verification.)

#### Evidence trail for the H.264 claim (documented vs inferred)

Documented in public code:
1. TokenBench GT is written as lossy H.264: `token_bench/video/preprocessing_script.py:74`
   (`media.write_video(...)` with no quality args) — github.com/NVlabs/TokenBench.
2. mediapy's default for >640×480 is `qp = 28`, codec h264, **yuv420p** (chroma subsampling):
   `mediapy/__init__.py`, `qp = 20 if math.prod(self.shape) <= 640*480 else 28`.
3. The metric CLI reads video FILES for both GT and reconstruction: `metrics_cli.py`
   (`--ext=mp4`; `read_video(input0_file)` / `read_video(input1_file)`, lines 167–168).
4. The inference CLI writes reconstructions via `write_video` (same lossy default):
   `cosmos_tokenizer/video_cli.py:204`.

NOT documented anywhere: how DAVIS (shipped as JPEG folders) was fed to this
file-based pipeline. Our claim for DAVIS is therefore an inference — supported by the
pipeline's video-file-only interface and by the bracketing experiment
(pristine 32.42 < official 32.80 < qp28 33.21) — and is labeled as such.


### Output-side recompression — VERIFIED; the official number is reproduced (2026-08-29)

Same anchor protocol (0.1-CV4x8x8, DAVIS 2016/50, 1080p, window 17), varying only
where the H.264 (mediapy default: qp 28, yuv420p) round-trip is applied:

| input/GT | reconstruction | PSNR (video-MSE) | SSIM |
|---|---|---|---|
| pristine | pristine | 32.42 | 0.893 |
| pristine | qp28 | 31.90 | 0.875 |
| qp28 | pristine | 33.21 | 0.917 |
| **qp28** | **qp28** (= the public CLI pipeline: preprocessing + `video_cli` + `metrics_cli`) | **32.77** | **0.904** |
| **official (paper / project page)** | | **32.80** | **0.900** |

Both sides compressed reproduces the official DAVIS number to **0.03 dB / 0.004 SSIM**.
Mechanism confirmed and decomposed: input smoothing raises PSNR (+0.78 dB), output
recompression lowers it (-0.52 dB), and the official pipeline does both.
So: (1) our harness is validated end-to-end against NVIDIA's own tooling; (2) published
Cosmos DAVIS/TokenBench numbers are measured on H.264-round-tripped content on both sides,
which any external comparison must replicate or declare.


### TRT 全栈重测:质量与 latency 同源 + calib 假设验证 (2026-09-02)

此前 ptq_quality 图混用两套口径(质量 = fake-quant,latency = TRT 引擎)。本轮把
五个配置全部落成真实引擎(fp16 IO、opset 18、无图重写、batch 1、CUDA graph),
PSNR/SSIM/latency 均出自同一引擎(数据: `results_trtfig/fig_e2e.json`、
`fig_bench.json`;新 pod,driver 570,TRT 10.16.1.11 **cu12**,torch 2.7.1+cu126)。

| 配置 | PSNR | Δ | SSIM | dec 引擎 ms | e2e clips/s |
|---|---|---|---|---|---|
| PyTorch JIT bf16(参考) | 28.10 | — | 0.786 | —(191 ms/clip e2e) | 5.2 |
| fp16 引擎 | 28.11 | 基线 | 0.787 | 40.4 | 15.5 |
| W8 dec(仅权重) | 28.11 | +0.00 | 0.787 | 41.1 (+2%) | 15.3 |
| W8A8 dec(全量) | 27.53 | −0.58 | 0.747 | 32.8 (−19%) | 17.5 |
| mixed(保 conv_out/norm_out/up.0) | 28.02 | −0.09 | 0.784 | 35.7 (−12%) | 16.7 |
| full VAE(enc+dec 全量) | 19.25 | −8.86 | 0.581 | 32.8(enc 23.7→20.3) | 18.5 |

要点:
1. **仅权重 W8 这次有了真实引擎**:质量与 fp16 完全相同,但 TRT 上零加速(41.1 vs
   40.4 ms)——权重 DQ 被常量折叠回 fp16 conv,坐实"仅权重量化在 TRT 无收益"。
2. **跨栈 0.2 dB 之谜解决 = 校准集,不是执行栈**。fake-quant `w8a8_dec` 用 16 段校准
   得 27.38;换成导出路径同款 8 段校准(`w8a8_dec_c8`)得 **27.54**;TRT dec-full 引擎
   27.53 —— 校准集对齐后 fake-quant 与部署引擎一致到 **0.01 dB**。此前 27.38 vs 27.49
   的差全部来自校准集(段数为主),两栈本身数值等价。
3. 各优化贡献隔离(同宿主机,`isol_bench.json`):CUDA graph fp16 −0.37 ms / int8
   −0.35 ms(约 −1%);fp16 IO −0.50 ms(−1.2%);opset18 GroupNorm −0.09 ms(≈0);
   图重写(conv pad 折叠 + resize)int8 32.76→29.98 ms(**−8.5%**,最大单项)。
4. **图重写不减 Q/DQ 节点**(118/118 不变),减的是 glue:总节点 4277→3098(−28%),
   Tile 60→0、Expand 57→0、Slice 137→43、Concat 180→88、Pad 57→25、Transpose
   120→88、Reshape 230→163(`results_trtfig/nodecount.txt`)。
5. 宿主机注意:本轮绝对 latency 与 results_e2e 时代不可直接比(fp16 decoder 40.4 vs
   当年 36.4 ms;同一 TRT 版本、不同宿主/driver)。只在同轮内比较。

环境坑(新增):modelopt 0.46 要求 torch≥2.6(`CPUOffloadPolicy` import,2.4.1 直接
挂);pip 装 `tensorrt` 元包在新环境解析到 **cu13** 轮子,driver 570(CUDA 12.8)上
CUDA error 35 —— 显式装 `tensorrt-cu12==<同版本>` 与原结果保持版本一致;后台队列脚本
必须逐步门控(`&&` + 显式 FAIL 标记),否则失败链会"空跑"并打出全部完成标记。


### 六配置 TRT 图第二轮 + 最终配置逐层 profile + 命名注明 (2026-09-02 下午)

**命名注明(两种 "mixed" 不是一回事):**
- **dec mixed**:decoder 量化、但保 `conv_out/norm_out/up.0` 三处 fp16 —— 质量优先;
- **enc mix+dec full**:部署式配置 —— encoder 量化保 `patcher/conv_in/down.0`,decoder
  全量 int8 —— 速度优先,把 1 dB 预算用满。图和下表均已按此标注。

第二轮整图重测(用户 pod,driver 580,同 TRT 10.16.1.11 cu12,torch 2.7.1+cu126;
数据 `results_trtfig/fig_e2e.json` / `fig_bench.json`,已覆盖第一轮 5 配置版本):

| 配置 | PSNR | Δ | SSIM | e2e ms | dec 引擎 ms | enc 引擎 ms |
|---|---|---|---|---|---|---|
| PyTorch bf16 | 28.10 | — | 0.786 | 181.1 | — | — |
| fp16 | 28.11 | 基线 | 0.787 | 63.9 | 40.0 | 23.0 |
| W8 dec | 28.11 | +0.00 | 0.787 | 63.7 | 39.7 | 23.0 |
| W8A8 dec | 27.53 | −0.58 | 0.747 | 56.6 | 32.0 | 23.0 |
| dec mixed | 28.02 | −0.09 | 0.784 | 59.3 | 34.8 | 23.0 |
| **enc mix+dec full** | **27.49** | **−0.62** | 0.745 | **54.1 (−15%)** | 32.0 | 20.3 |
| full VAE | 19.25 | −8.86 | 0.581 | 53.4 | 32.0 | 19.5 |

关键验证:**enc mix+dec full 在全新宿主机、全新校准下精确复现了部署配置的 −0.62 dB**
(它无图重写/opt5,加上后即为当年部署引擎)。质量列与第一轮 host 逐位一致(质量与
宿主机无关);两轮 host 的 W8 dec 分别 +2%/−1% —— "仅权重无加速"结论稳(±1% 噪声)。

**最终配置逐层 profile(这次是对的引擎:cr 重写 + enc-mixed / dec-full;
`results_trtfig/profile_final_{enc,dec}.json`)** —— 修正记录:results_e2e 里的
`profile_mixed_h16_*` 实为 enc-mixed + dec-**mixed** 无重写引擎,此前误标为部署配置:

| 算子组 | dec cr-full (30.4 ms, 397层) | enc cr-mixed (20.4 ms, 324层) |
|---|---|---|
| Conv/Deconv(int8 链,Q 融进 conv)| 17.8 ms / 58.6% | 12.6 ms / 61.7% |
| 归一化应用/激活/逐元素 | 4.7 ms / 15.6% | 2.7 ms / 13.4% |
| GroupNorm 统计(强制 fp32)| 4.2 ms / 13.7% | 2.2 ms / 10.9% |
| Reformat(拷贝)| 2.1 ms / 6.8% | 1.3 ms / 6.5% |
| Transpose/Reshape/glue | 1.5 ms / 4.8% | 1.3 ms / 6.3% |

对比无重写 dec-full(501 层):重写后 397 层,glue 组从 ~5.7% 保持相当但绝对值随
总时长下降;conv 占比 58.6%(计算主导),非 conv 最大项仍是 GroupNorm fp32 统计
(13.7%,已知死胡同)。profile 总时长(30.4+20.4)与部署 e2e ~50 ms 吻合(逐层
profiling 无法用 CUDA graph,略高)。

**优化贡献总表**(汇总,详见前文各节;同轮内自洽):有效 = TRT 化 3.0–3.4×、INT8
−19%、图重写(int8)−8.5%、encoder 量化 enc引擎 −12~15%、CUDA graph −1%、fp16 IO
−1.2%;无效 = 仅权重 W8(±1%)、SmoothQuant(no-op)、fp16 GN 重写(+34%)、opt5
(−0.2%)、strongly-typed(0)、batch4(+16%/clip)、opset18 GN(−0.2%)、fp16 下的
图重写(0)。

Pod 运维坑(新增):新版 runpod 镜像 PEP 668 拒绝 pip → `PIP_BREAK_SYSTEM_PACKAGES=1`;
`pkill -f <脚本名>` 会匹配同一条远程命令里的 `nohup bash <脚本>` 路径而自杀 —— kill 与
启动不要放同一条 ssh 命令;ssh.runpod.io 代理会无输出挂死(ConnectTimeout 不覆盖代理
握手),pod 有公网 TCP 22 映射时一律走直连(还能用 scp)。
