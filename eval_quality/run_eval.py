# SPDX-License-Identifier: Apache-2.0
"""Reconstruction-quality eval for Cosmos tokenizers (runs on GPU).

For every tokenizer in the registry: encode + decode each image/video in
--data_dir, compute PSNR/SSIM against the input, and write results to
--output_dir/metrics.json plus side-by-side comparison images.

Usage (from the repo root):
    python eval_quality/run_eval.py [--tokenizers CI8x8-360p ...] [--include-legacy]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizers_registry import select

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}


def frame_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Mean PSNR/SSIM over aligned uint8 frame stacks, layout TxHxWxC."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnrs, ssims = [], []
    for orig, recon in zip(original, reconstructed):
        psnrs.append(peak_signal_noise_ratio(orig, recon, data_range=255))
        ssims.append(structural_similarity(orig, recon, data_range=255, channel_axis=-1))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}


def eval_tokenizer(tok: dict, args, images: list, videos: list) -> dict:
    from cosmos_predict1.tokenizer.inference.image_lib import ImageTokenizer
    from cosmos_predict1.tokenizer.inference.utils import read_image, read_video, resize_image, resize_video, write_image
    from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer

    ckpt_dir = Path(args.checkpoint_dir) / tok["hf_repo"].split("/")[-1]
    enc, dec = ckpt_dir / "encoder.jit", ckpt_dir / "decoder.jit"
    if not (enc.exists() and dec.exists()):
        raise FileNotFoundError(f"{ckpt_dir} missing encoder.jit/decoder.jit; run download_checkpoints.py first")

    comparison_dir = Path(args.output_dir) / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    samples = {}
    if tok["kind"] == "image":
        tokenizer = ImageTokenizer(checkpoint_enc=str(enc), checkpoint_dec=str(dec), device=args.device, dtype=args.dtype)
        for filepath in images:
            image = resize_image(read_image(str(filepath)), short_size=args.short_side)
            recon = tokenizer(image[None])[0]
            samples[filepath.name] = frame_metrics(image[None], recon[None])
            write_image(str(comparison_dir / f"{tok['name']}__{filepath.stem}.png"), np.concatenate([image, recon], axis=1))
    else:
        tokenizer = CausalVideoTokenizer(checkpoint_enc=str(enc), checkpoint_dec=str(dec), device=args.device, dtype=args.dtype)
        for filepath in videos:
            video = resize_video(read_video(str(filepath))[: args.max_frames], short_size=args.short_side)
            recon = tokenizer(video[None], temporal_window=args.max_frames)[0]
            recon = recon[: video.shape[0]]
            samples[filepath.name] = frame_metrics(video, recon)
            mid = video.shape[0] // 2
            write_image(str(comparison_dir / f"{tok['name']}__{filepath.stem}.png"), np.concatenate([video[mid], recon[mid]], axis=1))

    return {
        **tok,
        "samples": samples,
        "psnr": float(np.mean([s["psnr"] for s in samples.values()])),
        "ssim": float(np.mean([s["ssim"] for s in samples.values()])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="cosmos_predict1/tokenizer/test_data")
    parser.add_argument("--output_dir", default="eval_quality/results")
    parser.add_argument("--tokenizers", nargs="*", default=None, help="subset of tokenizer names; default: all")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--max-frames", type=int, default=17, help="frames per video (also the causal temporal window)")
    parser.add_argument("--short-side", type=int, default=512, help="resize inputs to this short side (24GB-safe)")
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
            print(f"    PSNR {result['psnr']:.2f} dB, SSIM {result['ssim']:.4f}")
        except Exception as e:
            traceback.print_exc()
            result = {**tok, "error": f"{type(e).__name__}: {e}"}
        results.append(result)
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    output_path = Path(args.output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    failed = [r["name"] for r in results if "error" in r]
    print(f"Wrote {output_path} ({len(results) - len(failed)}/{len(results)} succeeded"
          + (f"; failed: {failed})" if failed else ")"))


if __name__ == "__main__":
    main()
