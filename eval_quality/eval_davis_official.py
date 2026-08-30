# SPDX-License-Identifier: Apache-2.0
"""DAVIS reconstruction eval following the official protocol (TokenBench
`metrics_cli.py` + Cosmos-Tokenizer `video_cli.py`, paper Table 5):
DAVIS 2016's 50 sequences at native 1080p (Full-Resolution JPEGs), the whole
sequence, `CausalVideoTokenizer.forward(video, temporal_window=49)` (the paper's
window for 0.1 / 360p models; 720p models used 121, which does not fit 24 GB —
flagged), bf16 JIT.  Metrics exactly as TokenBench: PSNR from the *video-level*
MSE over the whole T×H×W×3 float32 RGB array (one value per video, then mean),
SSIM per frame (skimage, data_range 255) → mean per video → mean. The per-frame
PSNR mean is reported too for comparison with our 480p harness. rFVD not computed.

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
from run_eval import VIDEO_EXTS, build_tokenizer
from tokenizers_registry import select


def _frame_pair_metrics(pair):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    orig, recon = pair
    return (peak_signal_noise_ratio(orig, recon, data_range=255),
            structural_similarity(orig, recon, data_range=255, channel_axis=-1))


def frame_metrics(original, reconstructed, pool=None) -> dict:
    """TokenBench-style metrics: video-level-MSE PSNR (official), per-frame SSIM,
    plus the per-frame PSNR mean (our 480p harness convention)."""
    pairs = list(zip(original, reconstructed))
    res = list(pool.map(_frame_pair_metrics, pairs, chunksize=2)) if pool else list(map(_frame_pair_metrics, pairs))
    psnrs, ssims = zip(*res)
    mse = float(((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2).mean())
    psnr_video = 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))  # metrics_cli.py formula
    return {"psnr": float(psnr_video), "psnr_frame_avg": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "psnr_frames": [round(float(p), 3) for p in psnrs]}


def read_sequence(seq_dir: Path, max_frames=None, recompress_qp=None) -> np.ndarray:
    import mediapy as media

    frames = sorted(seq_dir.glob("*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    video = np.stack([media.read_image(str(f))[..., :3] for f in frames], axis=0)
    if recompress_qp:  # TokenBench-style preprocessing: lossy H.264 round-trip; the
        import tempfile, os  # smoothed frames become BOTH model input and PSNR reference

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        media.write_video(tmp, video, fps=24, qp=recompress_qp)
        video = media.read_video(tmp)[..., :3]
        os.unlink(tmp)
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davis_root", default="/workspace/davis_fr/DAVIS")
    parser.add_argument("--resolution", default="Full-Resolution", help="JPEGImages subfolder: Full-Resolution or 480p")
    parser.add_argument("--split", default="2016", help="'2016' = DAVIS 2016 train+val (50 seqs, official), or ImageSets/2017/<split>.txt")
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--mode", default="jit", choices=["jit", "fakequant"])
    parser.add_argument("--keep_bf16", nargs="*", default=[])
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=8)
    parser.add_argument("--temporal_window", type=int, default=49, help="paper: 49 for 0.1/360p models, 121 for 720p (OOM on 24 GB)")
    parser.add_argument("--max_frames", type=int, default=None, help="debug: truncate sequences")
    parser.add_argument("--recompress_output_qp", type=int, default=None,
                        help="also H.264 round-trip the reconstruction before metrics (official CLI writes recon via write_video)")
    parser.add_argument("--recompress_qp", type=int, default=None,
                        help="TokenBench-style H.264 round-trip of the input/reference (e.g. 28)")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N sequences")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out_dir", default="eval_quality/results_davis_official")
    args = parser.parse_args()

    root = Path(args.davis_root)
    if args.split == "2016":
        seqs = []
        for part in ("train", "val"):
            seqs += [s.strip() for s in (root / "ImageSets" / "2016" / f"{part}.txt").read_text().split() if s.strip()]
    else:
        seqs = [s.strip() for s in (root / "ImageSets" / "2017" / f"{args.split}.txt").read_text().split() if s.strip()]
    if args.limit:
        seqs = seqs[: args.limit]
    img_root = root / "JPEGImages" / args.resolution
    print(f"{len(seqs)} sequences from {img_root} ({args.split}), window {args.temporal_window}, mode {args.mode}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    from concurrent.futures import ProcessPoolExecutor
    import os

    pool = ProcessPoolExecutor(max_workers=max(1, min(16, (os.cpu_count() or 4) - 1)))
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
            video = read_sequence(img_root / seq, args.max_frames, args.recompress_qp)
            with torch.no_grad():
                recon = tokenizer(video[None], temporal_window=args.temporal_window)[0]
            recon = recon[: video.shape[0]]
            if args.recompress_output_qp:  # official video_cli.py saves recon with mediapy defaults (qp28 @1080p)
                import mediapy as media, tempfile, os
                tmp = os.path.join(tempfile.gettempdir(), f"recon_{seq}.mp4")
                media.write_video(tmp, recon, fps=24, qp=args.recompress_output_qp)
                recon = media.read_video(tmp)[..., :3][: video.shape[0]]
                os.remove(tmp)
            m = frame_metrics(video, recon, pool)
            per_seq[seq] = {"psnr": m["psnr"], "psnr_frame_avg": m["psnr_frame_avg"], "ssim": m["ssim"],
                            "frames": int(video.shape[0]), "psnr_frames": m["psnr_frames"], "hw": list(video.shape[1:3])}
            print(f"  [{name}] {i+1:2d}/{len(seqs)} {seq:22s} {video.shape[0]:3d}f {video.shape[1]}x{video.shape[2]}  "
                  f"PSNR(video-MSE) {m['psnr']:.2f}  PSNR(frame-avg) {m['psnr_frame_avg']:.2f}  SSIM {m['ssim']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            del recon
            torch.cuda.empty_cache()

        n_frames = sum(v["frames"] for v in per_seq.values())
        result = {
            "name": name, "mode": args.mode, "keep_bf16": args.keep_bf16, "tag": args.tag,
            "resolution": args.resolution, "split": args.split, "temporal_window": args.temporal_window,
            "recompress_qp": args.recompress_qp, "recompress_output_qp": args.recompress_output_qp,
            "psnr_official": float(np.mean([v["psnr"] for v in per_seq.values()])),          # TokenBench: video-MSE PSNR, mean over videos
            "psnr_frame_avg": float(np.mean([v["psnr_frame_avg"] for v in per_seq.values()])),  # our 480p-harness convention
            "ssim_official": float(np.mean([v["ssim"] for v in per_seq.values()])),
            "n_sequences": len(per_seq), "n_frames": n_frames, "per_sequence": per_seq,
        }
        print(f"[{args.tag}/{name}] PSNR {result['psnr_official']:.2f} (official, video-MSE) / {result['psnr_frame_avg']:.2f} (frame-avg)  "
              f"SSIM {result['ssim_official']:.4f}  over {len(per_seq)} seqs, {n_frames} frames", flush=True)
        all_results.append(result)
        (out_dir / f"{args.tag}__{name}.json").write_text(json.dumps(result, indent=2))
        del tokenizer
        torch.cuda.empty_cache()

    pool.shutdown()
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
