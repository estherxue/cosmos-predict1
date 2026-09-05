# Cosmos Tokenizer: quality vs compression, official-protocol reproduction, INT8/TensorRT deployment

Study of the [Cosmos-Tokenize1](https://docs.nvidia.com/cosmos/latest/predict1/index.html)
tokenizers shipped with cosmos-predict1, in four phases (all code in this directory,
findings and decision log in `quantization_research.md`):

1. **Quality vs compression rate** — encode/decode each input, PSNR/SSIM vs the nominal
   spatio-temporal downsampling factor, four families (CI/DI/CV/DV).
2. **Motion analysis** — temporal compression selectively damages high-motion clips
   (ρ(motion, damage) = +0.73; MSE ratio ×1.06→×2.20 from static to fastest clip), spatial
   compression is motion-agnostic (ρ = −0.05). `motion_split.py`, per-clip data in `results_full_4090/`.
3. **Official DAVIS numbers reproduced** — `eval_davis_official.py` (DAVIS 2016, 50 seqs, 1080p,
   TokenBench video-MSE PSNR). Published numbers are measured on H.264-recompressed content on
   **both** sides (mediapy default qp28): pristine 32.42 < official 32.80 ≈ ours-both-qp28 32.77.
4. **INT8 PTQ + TensorRT** — deployed config (encoder-mixed + decoder-full INT8, graph rewrite,
   CUDA graph) reaches **3.67× vs PyTorch bf16 at −0.62 dB** on an RTX 4090.
   Figure: `results_quant/ptq_quality.png` (every point a real TRT engine).

"Compression rate" here is the nominal spatio-temporal downsampling factor
(e.g. 8x8 → 64×, 8x16x16 → 2048×), not a bit-rate compression ratio.

The Tokenize1 series lacks CV8x16x16 and DV8x8x8; `--include-legacy` adds those two compression
points from the older Cosmos-0.1 series (shown as hollow markers in the plot).

## Run (GPU box, e.g. RunPod 4090)

```bash
apt-get install -y ffmpeg                      # needed by mediapy video IO
git clone -b tokenizer-quality-eval https://github.com/estherxue/cosmos-predict1.git
cd cosmos-predict1
pip install -r eval_quality/requirements.txt   # torch/torchvision assumed preinstalled

python eval_quality/download_checkpoints.py --include-legacy   # encoder/decoder JIT only, no HF token needed
python eval_quality/run_eval.py --include-legacy               # writes eval_quality/results/metrics.json + comparisons/
```

Besides PSNR/SSIM, `run_eval.py` reports encode/decode ms per frame (median across samples,
CUDA-synced, after a warmup pass) and peak VRAM, and writes a one-line-per-tokenizer
`results/summary.csv`. For videos it also stores a per-clip motion score (mean |frame diff|)
in metrics.json and saves a second comparison image cropped to the motion-hot region
(`*__motioncrop.png`) — original | reconstruction of the same crop, where temporal
compression damage is most visible.

Smoke test a single tokenizer first: `python eval_quality/run_eval.py --tokenizers CI8x8-360p`

Defaults are 24GB-safe: videos are cut to `--max-frames 17` and inputs resized to `--short-side 512`.

## Plot (no GPU needed)

```bash
python eval_quality/plot_results.py            # writes eval_quality/results/quality_vs_compression.png
```

## Data

Defaults to the repo's `cosmos_predict1/tokenizer/test_data/` (1 image + 1 video, minimal smoke set).
Point `--data_dir` at any folder of `.png/.jpg` images and `.mp4/.mov/.webm` videos to evaluate on more data.

The committed `results/` were produced on **DAVIS 2017 val** (30 sequences, 480p; first 17 frames as a
lossless mp4 per sequence for the video tokenizers, first frame as png for the image tokenizers):

```bash
bash eval_quality/prepare_davis.sh                 # downloads DAVIS, builds /workspace/davis_eval
python eval_quality/run_eval.py --data_dir /workspace/davis_eval --short-side 480 --include-legacy
```

## Official-protocol evaluation

```bash
python eval_quality/eval_davis_official.py --tokenizer 0.1-CV4x8x8            # pristine inputs
python eval_quality/eval_davis_official.py --recompress_qp 28 --recompress_output_qp 28   # = public CLI pipeline
```

DAVIS 2016 50 sequences at 1080p, TokenBench video-level-MSE PSNR. The two H.264 flags
reproduce the published pipeline (input preprocessing and output writing both default to
H.264 qp28 in the official tooling); results in `results_davis_official/`.

## INT8 quantization + TensorRT (RunPod 4090)

Quality (fake-quant, PyTorch): `quantize_ptq.py` — configs none / w8_dec / w8a8_dec /
w8a8_dec_mixed / w8a8_all, `--keep_bf16` for mixed precision, results in `results_quant/`.
Visual comparisons: `make_quant_visuals.py` (`results_quant/visuals/`).

Deployment: `export_onnx.py` (fp16) and `export_qdq.py` (INT8 Q/DQ, torch-side export)
produce fixed-shape ONNX; `export_patches.py` holds the numerically-exact graph rewrites
(conv-pad fold, repeat_interleave→Resize). `trt_bench.py` builds/benches single engines,
`trt_eval.py` measures PSNR/SSIM **and** latency end-to-end on the same engine pair,
`trt_profile.py` dumps per-layer times/precision. Pin `tensorrt-cu12==10.16.*` to the
host driver and `torch==2.7.1+cu126` (modelopt 0.46 needs torch ≥2.6).

Key results (details, isolation matrices and pitfalls in `quantization_research.md`):
- Deployed: enc-mixed + dec-full INT8 + graph rewrite + CUDA graph = **19.8 clips/s,
  3.67× vs PyTorch bf16, −0.62 dB** (figure: `results_quant/ptq_quality.png`).
- What works: TRT itself (3.0×), INT8 (−19% decoder), graph rewrite (−8.5%, int8 only).
- What doesn't: weights-only W8 (±1%), SmoothQuant (no-op on conv), fp16 GroupNorm
  rewrite (+34%, precision-broken), batch>1 (+16%/clip), opt_level 5 (<0.5%).
