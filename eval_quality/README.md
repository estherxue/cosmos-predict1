# Cosmos Tokenizer: reconstruction quality vs compression rate

Evaluates the [Cosmos-Tokenize1](https://docs.nvidia.com/cosmos/latest/predict1/index.html)
tokenizers shipped with cosmos-predict1: each image/video is encoded and decoded, and the
reconstruction is compared to the input with PSNR/SSIM. Results are plotted as quality-vs-compression
curves for the four tokenizer families (CI/DI = continuous/discrete image, CV/DV = continuous/discrete video).

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
