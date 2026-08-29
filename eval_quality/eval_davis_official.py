# SPDX-License-Identifier: Apache-2.0
"""DAVIS reconstruction eval following the official inference path:
native resolution (DAVIS 2017 Full-Resolution, 1080p), the whole sequence,
`CausalVideoTokenizer.forward(video, temporal_window=17)` (sliding causal
windows, bf16), per-frame PSNR/SSIM on uint8 RGB, averaged per video then
across videos (all-frame mean also reported).

Modes:
  jit        official JIT checkpoints (baseline reproduction)
  fakequant  native model + modelopt INT8 fake-quant (same scales as the
             deployed engines: calibrated on /workspace/davis_calib), optional
             --keep_bf16 exclusions — the quantized models under the same protocol

Usage (from the repo root):
    python eval_quality/eval_davis_official.py --tokenizers CV4x8x8-360p CV8x8x8-720p --tag jit
    python eval_quality/eval_davis_official.py --tokenizers 0.1-CV8x8x8 --mode fakequant \
        --keep_bf16 conv_in patcher down.0 --tag int8_encmix_decfull
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import VIDEO_EXTS, build_tokenizer, frame_metrics
from tokenizers_registry import select


def read_sequence(seq_dir: Path, max_frames=None) -> np.ndarray:
    import mediapy as media

    frames = sorted(seq_dir.glob("*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    video = np.stack([media.read_image(str(f))[..., :3] for f in frames], axis=0)
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davis_root", default="/workspace/davis_fr/DAVIS")
    parser.add_argument("--resolution", default="Full-Resolution", help="JPEGImages subfolder: Full-Resolution or 480p")
    parser.add_argument("--split", default="val", help="ImageSets/2017/<split>.txt")
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--mode", default="jit", choices=["jit", "fakequant"])
    parser.add_argument("--keep_bf16", nargs="*", default=[])
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=8)
    parser.add_argument("--temporal_window", type=int, default=17)
    parser.add_argument("--max_frames", type=int, default=None, help="debug: truncate sequences")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N sequences")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out_dir", default="eval_quality/results_davis_official")
    args = parser.parse_args()

    root = Path(args.davis_root)
    seqs = [s.strip() for s in (root / "ImageSets" / "2017" / f"{args.split}.txt").read_text().split() if s.strip()]
    if args.limit:
        seqs = seqs[: args.limit]
    img_root = root / "JPEGImages" / args.resolution
    print(f"{len(seqs)} sequences from {img_root} ({args.split}), window {args.temporal_window}, mode {args.mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name in args.tokenizers:
        tok = select([name])[0]
        if args.mode == "jit":
            tokenizer = build_tokenizer(tok, args, native=False)
        else:
            import modelopt.torch.quantization as mtq
            from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, read_video, resize_video
            from export_qdq import make_config

            torch.backends.cudnn.enabled = False  # native eager conv3d faults on 4090 with cuDNN
            tokenizer = build_tokenizer(tok, args, native=True)
            files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
            batches = [pad_video_batch(resize_video(read_video(str(p))[:17], short_size=480)[None])[0] for p in files]

            def forward_loop(_m):
                with torch.no_grad():
                    for b in batches:
                        x = numpy2tensor(b, tokenizer._dtype, args.device)
                        tokenizer.decode(tokenizer.encode(x)[0])

            mtq.quantize(tokenizer, make_config(mtq, args.keep_bf16), forward_loop=forward_loop)
            print(f"fake-quant ready ({name}, keep_bf16={args.keep_bf16})")

        per_seq, t0 = {}, time.time()
        for i, seq in enumerate(seqs):
            video = read_sequence(img_root / seq, args.max_frames)
            with torch.no_grad():
                recon = tokenizer(video[None], temporal_window=args.temporal_window)[0]
            recon = recon[: video.shape[0]]
            m = frame_metrics(video, recon)
            per_seq[seq] = {"psnr": m["psnr"], "ssim": m["ssim"], "frames": int(video.shape[0]),
                            "psnr_frames": m["psnr_frames"], "hw": list(video.shape[1:3])}
            print(f"  [{name}] {i+1:2d}/{len(seqs)} {seq:22s} {video.shape[0]:3d}f {video.shape[1]}x{video.shape[2]}  "
                  f"PSNR {m['psnr']:.2f} SSIM {m['ssim']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            del recon
            torch.cuda.empty_cache()

        n_frames = sum(v["frames"] for v in per_seq.values())
        result = {
            "name": name, "mode": args.mode, "keep_bf16": args.keep_bf16, "tag": args.tag,
            "resolution": args.resolution, "split": args.split, "temporal_window": args.temporal_window,
            "psnr_video_mean": float(np.mean([v["psnr"] for v in per_seq.values()])),
            "ssim_video_mean": float(np.mean([v["ssim"] for v in per_seq.values()])),
            "psnr_frame_mean": float(sum(v["psnr"] * v["frames"] for v in per_seq.values()) / n_frames),
            "ssim_frame_mean": float(sum(v["ssim"] * v["frames"] for v in per_seq.values()) / n_frames),
            "n_sequences": len(per_seq), "n_frames": n_frames, "per_sequence": per_seq,
        }
        print(f"[{args.tag}/{name}] PSNR {result['psnr_video_mean']:.2f} (video-mean) / {result['psnr_frame_mean']:.2f} (frame-mean)  "
              f"SSIM {result['ssim_video_mean']:.4f}  over {len(per_seq)} seqs, {n_frames} frames", flush=True)
        all_results.append(result)
        (out_dir / f"{args.tag}__{name}.json").write_text(json.dumps(result, indent=2))
        del tokenizer
        torch.cuda.empty_cache()

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
