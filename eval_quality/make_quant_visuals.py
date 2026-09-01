# SPDX-License-Identifier: Apache-2.0
"""Visual comparison of quantized vs bf16 reconstructions.

Per clip: a labeled 4-panel strip (original | bf16 | INT8 mixed | INT8 full-VAE)
at the middle frame, full-frame and motion-crop versions, plus an 8x-amplified
|INT8mixed - bf16| difference heatmap. PSNR (vs original, this clip) in titles.

Usage (from the repo root, on a GPU box):
    python eval_quality/make_quant_visuals.py --clips shooting judo drift-chicane bmx-trees
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_qdq import make_config
from run_eval import VIDEO_EXTS, build_tokenizer, motion_crop_box, motion_heatmap
from tokenizers_registry import select


def recon_clip(tokenizer, video, device):
    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, tensor2numpy, unpad_video_batch

    padded, crop = pad_video_batch(video[None])
    x = numpy2tensor(padded, tokenizer._dtype, device)
    with torch.no_grad():
        y = tokenizer.decode(tokenizer.encode(x)[0])
    return unpad_video_batch(tensor2numpy(y), crop)[0][: video.shape[0]]


def psnr(a, b):
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["shooting", "judo", "drift-chicane", "bmx-trees"])
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--data_dir", default="/workspace/davis_eval")
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=16)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="eval_quality/results_quant/visuals")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    torch.backends.cudnn.enabled = False  # native eager conv3d faults on 4090 with cuDNN
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import modelopt.torch.quantization as mtq
    from cosmos_predict1.tokenizer.inference.utils import read_video, resize_video

    tok = select([args.tokenizer])[0]
    videos = {}
    for name in args.clips:
        p = Path(args.data_dir) / f"{name}.mp4"
        videos[name] = resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side)

    calib_files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
    calib = [resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side) for p in calib_files]

    configs = {"bf16": [], "int8_mixed": ["conv_in", "patcher", "down.0"], "int8_full": None}
    recons = {}
    for cfg_name, keep in configs.items():
        tokenizer = build_tokenizer(tok, args, native=True)
        if cfg_name != "bf16":
            def forward_loop(_m):
                for v in calib:
                    recon_clip(tokenizer, v, args.device)
            mtq.quantize(tokenizer, make_config(mtq, keep or []), forward_loop=forward_loop)
        recons[cfg_name] = {n: recon_clip(tokenizer, v, args.device) for n, v in videos.items()}
        print(f"{cfg_name}: " + "  ".join(f"{n} {psnr(videos[n], recons[cfg_name][n]):.2f}dB" for n in videos), flush=True)
        del tokenizer
        torch.cuda.empty_cache()

    for name, video in videos.items():
        mid = video.shape[0] // 2
        heat, motion = motion_heatmap(video)
        y, x, s = motion_crop_box(heat, 256)
        panels = [("original", video), ("bf16", recons["bf16"][name]),
                  ("INT8 mixed (accepted)", recons["int8_mixed"][name]), ("INT8 full VAE (fail)", recons["int8_full"][name])]
        for suffix, cropfn in (("panel", lambda f: f[mid]), ("motioncrop", lambda f: f[mid, y:y + s, x:x + s])):
            fig, axes = plt.subplots(1, 4, figsize=(22, 6.5), facecolor="#fcfcfb")
            for ax, (label, frames) in zip(axes, panels):
                ax.imshow(cropfn(frames))
                title = label if label == "original" else f"{label}\nPSNR {psnr(video, frames):.2f} dB"
                ax.set_title(title, fontsize=11, color="#0b0b0b")
                ax.axis("off")
            fig.suptitle(f"{name}  (motion score {motion:.1f}, mid frame)", fontsize=13, color="#0b0b0b", x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(out / f"{name}_{suffix}.png", dpi=120, facecolor="#fcfcfb")
            plt.close(fig)
        diff = np.abs(recons["int8_mixed"][name][mid].astype(np.int16) - recons["bf16"][name][mid].astype(np.int16)).mean(-1)
        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
        im = ax.imshow(np.clip(diff * 8, 0, 255), cmap="magma", vmin=0, vmax=255)
        ax.set_title(f"{name}: |INT8 mixed − bf16| × 8  (mean {diff.mean():.2f}/255)", fontsize=11, color="#0b0b0b")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / f"{name}_diff.png", dpi=120, facecolor="#fcfcfb")
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
