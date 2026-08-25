# SPDX-License-Identifier: Apache-2.0
"""Reconstruction-quality + speed eval for Cosmos tokenizers (runs on GPU).

For every tokenizer in the registry: encode + decode each image/video in
--data_dir, then report PSNR/SSIM, encode/decode ms per frame (median across
samples, CUDA-synced, after a warmup pass), and peak VRAM. Results go to
--output_dir/metrics.json + summary.csv, plus side-by-side comparison images
(full mid-frame, and a motion-hot-region crop for videos).

Usage (from the repo root):
    python eval_quality/run_eval.py [--tokenizers CI8x8-360p ...] [--include-legacy]
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizers_registry import select

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}
CSV_FIELDS = ["name", "series", "family", "kind", "compression", "psnr", "ssim",
              "enc_ms_per_frame", "dec_ms_per_frame", "peak_vram_gb", "n_samples"]


def frame_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Mean PSNR/SSIM over aligned uint8 frame stacks, layout TxHxWxC."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnrs, ssims = [], []
    for orig, recon in zip(original, reconstructed):
        psnrs.append(peak_signal_noise_ratio(orig, recon, data_range=255))
        ssims.append(structural_similarity(orig, recon, data_range=255, channel_axis=-1))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def motion_heatmap(video: np.ndarray) -> tuple:
    """Per-pixel accumulated |frame diff| (HxW) and the mean-abs-diff motion score."""
    diff = np.abs(video[1:].astype(np.int16) - video[:-1].astype(np.int16))
    return diff.sum(axis=(0, 3)).astype(np.float64), float(diff.mean())


def motion_crop_box(heat: np.ndarray, size: int = 256) -> tuple:
    """Top-left corner (y, x) and side s of the s×s window maximizing summed heat."""
    height, width = heat.shape
    s = min(size, height, width)
    ii = np.pad(heat.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    win = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
    y, x = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(y), int(x), s


def autoencode_timed(tokenizer, batch: np.ndarray, kind: str, device: str) -> tuple:
    """Encode+decode a uint8 batch via upstream pad/normalize utils.

    Returns (reconstructed uint8 batch, encode seconds, decode seconds).
    """
    import torch
    from cosmos_predict1.tokenizer.inference.utils import (
        numpy2tensor, pad_image_batch, pad_video_batch, tensor2numpy,
        unpad_image_batch, unpad_video_batch,
    )

    pad, unpad = (pad_image_batch, unpad_image_batch) if kind == "image" else (pad_video_batch, unpad_video_batch)
    padded, crop_region = pad(batch)
    input_tensor = numpy2tensor(padded, dtype=tokenizer._dtype, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        latent = tokenizer.encode(input_tensor)[0]
        sync()
        t1 = time.perf_counter()
        output_tensor = tokenizer.decode(latent)
        sync()
        t2 = time.perf_counter()
    recon = unpad(tensor2numpy(output_tensor), crop_region)
    return recon, t1 - t0, t2 - t1


def eval_tokenizer(tok: dict, args, images: list, videos: list) -> dict:
    import torch
    from cosmos_predict1.tokenizer.inference.image_lib import ImageTokenizer
    from cosmos_predict1.tokenizer.inference.utils import read_image, read_video, resize_image, resize_video, write_image
    from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

    ckpt_dir = Path(args.checkpoint_dir) / tok["hf_repo"].split("/")[-1]
    enc, dec = ckpt_dir / "encoder.jit", ckpt_dir / "decoder.jit"
    if not (enc.exists() and dec.exists()):
        raise FileNotFoundError(f"{ckpt_dir} missing encoder.jit/decoder.jit; run download_checkpoints.py first")

    comparison_dir = Path(args.output_dir) / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cls = ImageTokenizer if tok["kind"] == "image" else CausalVideoTokenizer
    tokenizer = cls(checkpoint_enc=str(enc), checkpoint_dec=str(dec), device=args.device, dtype=args.dtype)

    def load(filepath):
        if tok["kind"] == "image":
            return resize_image(read_image(str(filepath)), short_size=args.short_side)[None]
        return resize_video(read_video(str(filepath))[: args.max_frames], short_size=args.short_side)[None]

    files = images if tok["kind"] == "image" else videos
    autoencode_timed(tokenizer, load(files[0]), tok["kind"], args.device)  # warmup (JIT/cuDNN init)

    samples, enc_ms, dec_ms = {}, [], []
    for filepath in files:
        batch = load(filepath)
        recon_batch, t_enc, t_dec = autoencode_timed(tokenizer, batch, tok["kind"], args.device)
        original, recon = batch[0], recon_batch[0]
        n_frames = 1 if tok["kind"] == "image" else original.shape[0]
        enc_ms.append(t_enc * 1000 / n_frames)
        dec_ms.append(t_dec * 1000 / n_frames)

        if tok["kind"] == "image":
            samples[filepath.name] = frame_metrics(original[None], recon[None])
            write_image(str(comparison_dir / f"{tok['name']}__{filepath.stem}.png"), np.concatenate([original, recon], axis=1))
        else:
            recon = recon[: original.shape[0]]
            heat, motion = motion_heatmap(original)
            samples[filepath.name] = {**frame_metrics(original, recon), "motion": motion}
            mid = original.shape[0] // 2
            write_image(str(comparison_dir / f"{tok['name']}__{filepath.stem}.png"),
                        np.concatenate([original[mid], recon[mid]], axis=1))
            y, x, s = motion_crop_box(heat, args.crop_size)
            write_image(str(comparison_dir / f"{tok['name']}__{filepath.stem}__motioncrop.png"),
                        np.concatenate([original[mid, y:y + s, x:x + s], recon[mid, y:y + s, x:x + s]], axis=1))

    result = {
        **tok,
        "samples": samples,
        "psnr": float(np.mean([s["psnr"] for s in samples.values()])),
        "ssim": float(np.mean([s["ssim"] for s in samples.values()])),
        "enc_ms_per_frame": float(np.median(enc_ms)),
        "dec_ms_per_frame": float(np.median(dec_ms)),
        "n_samples": len(samples),
    }
    if args.device == "cuda":
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="cosmos_predict1/tokenizer/test_data")
    parser.add_argument("--output_dir", default="eval_quality/results")
    parser.add_argument("--tokenizers", nargs="*", default=None, help="subset of tokenizer names; default: all")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--max-frames", type=int, default=17, help="frames per video (17 = 1+4k and 1+8k, one causal window)")
    parser.add_argument("--short-side", type=int, default=512, help="resize inputs to this short side (24GB-safe)")
    parser.add_argument("--crop-size", type=int, default=256, help="side of the motion-hot-region comparison crop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    print(f"Data: {len(images)} image(s), {len(videos)} video(s) from {data_dir}")

    results = []
    for tok in select(args.tokenizers, args.include_legacy):
        print(f"=== {tok['name']} ({tok['kind']}, {tok['compression']}x) ===")
        try:
            result = eval_tokenizer(tok, args, images, videos)
            print(f"    PSNR {result['psnr']:.2f} dB, SSIM {result['ssim']:.4f}, "
                  f"enc {result['enc_ms_per_frame']:.1f} / dec {result['dec_ms_per_frame']:.1f} ms/frame, "
                  f"peak {result.get('peak_vram_gb', 0):.2f} GB")
        except Exception as e:
            traceback.print_exc()
            result = {**tok, "error": f"{type(e).__name__}: {e}"}
        results.append(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if "error" not in r:
                writer.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in CSV_FIELDS})
    failed = [r["name"] for r in results if "error" in r]
    print(f"Wrote {output_dir}/metrics.json + summary.csv ({len(results) - len(failed)}/{len(results)} succeeded"
          + (f"; failed: {failed})" if failed else ")"))


if __name__ == "__main__":
    main()
