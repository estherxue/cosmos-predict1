# SPDX-License-Identifier: Apache-2.0
"""End-to-end TensorRT evaluation: encoder engine -> decoder engine on DAVIS val.

Reports PSNR/SSIM against the original frames AND throughput (clips/s, frames/s)
for the same engines, so quality and speed are measured on one artifact.
--reference additionally runs the PyTorch JIT bf16 tokenizer on identical
(cropped, fixed-shape) inputs for an apples-to-apples baseline.

Usage:
    python eval_quality/trt_eval.py --enc_engine /workspace/trt/encoder.fp16.engine \
        --dec_engine /workspace/trt/decoder.fp16.engine --tag trt_fp16 --reference
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import VIDEO_EXTS, build_tokenizer, frame_metrics
from tokenizers_registry import select


class TRTModule:
    """Minimal single-input/single-output TensorRT engine runner on torch buffers."""

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = torch.float32 if self.engine.get_tensor_dtype(name) == trt.DataType.FLOAT else torch.float16
            self.buffers[name] = torch.empty(shape, dtype=dt, device="cuda")
            self.ctx.set_tensor_address(name, self.buffers[name].data_ptr())
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.stream = torch.cuda.Stream()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        buf = self.buffers[self.in_name]
        buf.copy_(x.to(buf.dtype))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.out_name].clone()


def load_clips(data_dir, max_frames, short_side, width, limit):
    from cosmos_predict1.tokenizer.inference.utils import read_video, resize_video

    files = sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    clips = []
    for p in files[:limit] if limit else files:
        video = resize_video(read_video(str(p))[:max_frames], short_size=short_side)
        video = video[:, :short_side, :width]
        if video.shape[1:3] != (short_side, width):
            print(f"skip {p.name}: {video.shape[1:3]} smaller than {(short_side, width)}")
            continue
        clips.append((p.name, video))
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enc_engine")
    parser.add_argument("--dec_engine")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reference", action="store_true", help="also run the PyTorch JIT bf16 tokenizer")
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="/workspace/davis_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default="/workspace/trt/e2e.json")
    args = parser.parse_args()

    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, tensor2numpy, unpad_video_batch

    clips = load_clips(args.data_dir, args.max_frames, args.short_side, args.width, args.limit)
    print(f"{len(clips)} clips @ {args.short_side}x{args.width}x{args.max_frames}")
    rows = []

    def evaluate(tag, run_fn, in_dtype):
        psnrs, ssims, times, samples = [], [], [], {}
        for name, video in clips:
            padded, crop = pad_video_batch(video[None])
            x = numpy2tensor(padded, in_dtype, "cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = run_fn(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            recon = unpad_video_batch(tensor2numpy(y), crop)[0][: video.shape[0]]
            m = frame_metrics(video, recon)
            samples[name] = {"psnr": m["psnr"], "ssim": m["ssim"]}
            psnrs.append(m["psnr"]); ssims.append(m["ssim"])
        med = float(np.median(times[1:])) if len(times) > 1 else float(times[0])
        row = {"tag": tag, "psnr": round(float(np.mean(psnrs)), 3), "ssim": round(float(np.mean(ssims)), 4),
               "median_ms": round(med, 2), "clips_per_s": round(1000 / med, 3),
               "frames_per_s": round(1000 / med * args.max_frames, 1), "n": len(clips), "samples": samples}
        print(f"[{tag}] PSNR {row['psnr']:.2f}  SSIM {row['ssim']:.4f}  {row['median_ms']:.1f} ms/clip  "
              f"{row['clips_per_s']:.2f} clips/s  {row['frames_per_s']:.0f} frames/s")
        rows.append(row)

    if args.reference:
        tok = select([args.tokenizer])[0]
        tokenizer = build_tokenizer(tok, args, native=False)
        with torch.no_grad():
            first = numpy2tensor(pad_video_batch(clips[0][1][None])[0], tokenizer._dtype, "cuda")
            tokenizer.decode(tokenizer.encode(first)[0])  # warmup
            evaluate("pytorch_jit_bf16", lambda x: tokenizer.decode(tokenizer.encode(x)[0]), tokenizer._dtype)
        del tokenizer
        torch.cuda.empty_cache()

    if args.enc_engine and args.dec_engine:
        enc, dec = TRTModule(args.enc_engine), TRTModule(args.dec_engine)
        warm = numpy2tensor(pad_video_batch(clips[0][1][None])[0], torch.float32, "cuda")
        dec.run(enc.run(warm))
        evaluate(args.tag, lambda x: dec.run(enc.run(x)), torch.float32)

    out = Path(args.out)
    prev = json.loads(out.read_text()) if out.exists() else []
    tags = {r["tag"] for r in rows}
    out.write_text(json.dumps([r for r in prev if r["tag"] not in tags] + rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
