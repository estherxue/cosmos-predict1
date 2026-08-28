# SPDX-License-Identifier: Apache-2.0
"""INT8 quantization-aware training (modelopt fake-quant) for the tokenizer.

Steps: native load -> mtq.quantize (PTQ calibration, optional bf16 exclusions)
-> short reconstruction-loss fine-tune on random spatio-temporal crops of the
calibration clips (amax frozen, weights updated) -> full DAVIS-val eval with the
standard harness -> save fake-quant state dict for ONNX export.

Usage (from the repo root):
    python eval_quality/qat.py --steps 400 --tag qat_full
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import IMAGE_EXTS, VIDEO_EXTS, autoencode_timed, build_tokenizer, eval_tokenizer
from tokenizers_registry import select


def make_config(mtq, keep_bf16):
    import copy

    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    for pattern in keep_bf16:
        cfg["quant_cfg"].append({"quantizer_name": f"*{pattern}*", "enable": False})
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--tag", default="qat")
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--data_dir", default="/workspace/davis_eval")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--keep_bf16", nargs="*", default=[], help="module substrings left unquantized")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--frames", type=int, default=9, help="temporal crop, 1+8k")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--no_cudnn", action="store_true", default=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir or f"eval_quality/results_quant/{args.tag}"
    torch.manual_seed(0)
    np.random.seed(0)
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False  # native eager conv3d faults on 4090 with cuDNN

    import modelopt.torch.quantization as mtq
    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, read_video, resize_video

    tok = select([args.tokenizer])[0]
    tokenizer = build_tokenizer(tok, args, native=True)

    # ---- training clips (full res, uint8) ----
    files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    clips = [resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side) for p in files]
    print(f"{len(clips)} training clips")

    def sample_batch():
        v = clips[np.random.randint(len(clips))]
        t0 = np.random.randint(0, v.shape[0] - args.frames + 1)
        y0 = np.random.randint(0, v.shape[1] - args.crop + 1)
        x0 = np.random.randint(0, v.shape[2] - args.crop + 1)
        crop = v[t0 : t0 + args.frames, y0 : y0 + args.crop, x0 : x0 + args.crop][None]
        return numpy2tensor(crop, tokenizer._dtype, args.device)

    # ---- PTQ calibration (full VAE quantized unless excluded) ----
    calib = [sample_batch() for _ in range(16)]

    def forward_loop(_m):
        with torch.no_grad():
            for x in calib:
                tokenizer.decode(tokenizer.encode(x)[0])

    mtq.quantize(tokenizer, make_config(mtq, args.keep_bf16), forward_loop=forward_loop)
    mtq.print_quant_summary(tokenizer)

    # ---- QAT: reconstruction loss, amax frozen (modelopt default), weights trained ----
    params = [p for p in tokenizer.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr)
    tokenizer.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x = sample_batch()
        recon = tokenizer.decode(tokenizer.encode(x)[0])
        loss = (recon.float() - x.float()).abs().mean() + 0.5 * ((recon.float() - x.float()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.5f}  ({time.time()-t0:.0f}s)", flush=True)
    tokenizer.eval()
    for p in params:
        p.requires_grad_(False)

    # ---- full-resolution eval on DAVIS val with the standard harness ----
    data_dir = Path(args.data_dir)
    images = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if args.eval_limit:
        images, videos = images[: args.eval_limit], videos[: args.eval_limit]
    result = eval_tokenizer(tok, args, images, videos, tokenizer=tokenizer)
    result["quant"] = args.tag
    result["keep_bf16"] = args.keep_bf16
    result["qat_steps"] = args.steps
    print(f"[{args.tag}] PSNR {result['psnr']:.2f} dB, SSIM {result['ssim']:.4f}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "results": [result]}, indent=2))
    torch.save(tokenizer.state_dict(), out / "qat_state.pt")
    print(f"wrote {out}/metrics.json and qat_state.pt")


if __name__ == "__main__":
    main()
