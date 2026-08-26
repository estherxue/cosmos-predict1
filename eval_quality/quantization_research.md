# Quantizing the Cosmos tokenizer (VAE) — research notes

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
