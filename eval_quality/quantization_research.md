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
head+up.0 combo recovers 0.75 → 0.16 dB on full val. Also: per-clip quantization
damage is **motion-independent** (flat scatter vs motion score) — the opposite
signature of temporal compression, confirming they degrade through different
mechanisms.

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
between 2.04 and 2.31 ms/frame. The modest int8 gain suggests the 3D-conv decoder
is bandwidth-/tactic-bound rather than math-bound at this size.

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
