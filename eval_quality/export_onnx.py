# SPDX-License-Identifier: Apache-2.0
"""Export the native encoder and/or decoder to ONNX (fp32, fixed shape) plus
calibration data for modelopt ONNX quantization.

Outputs in --out_dir:
  decoder.onnx, calib_latents.npy      (decoder input = encoder output of calib clips)
  encoder.onnx, calib_videos.npy       (encoder input = normalized padded clips, [-1,1])

Usage (from the repo root):
    python eval_quality/export_onnx.py --tokenizer 0.1-CV8x8x8 --out_dir /workspace/trt --part both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import VIDEO_EXTS, build_tokenizer
from tokenizers_registry import select


def install_export_friendly_idwt(decoder):
    """UnPatcher3D builds its wavelet conv kernels inside forward from x.shape,
    which the ONNX tracer can't resolve to a static kernel shape. Cache the
    kernels per (groups, dtype) during an eager warmup run, so tracing hits the
    cache and records them as constants. Upstream code is left untouched.
    """
    import types

    import torch.nn.functional as F
    from cosmos_predict1.tokenizer.modules.patching import UnPatcher3D

    def cached_idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        g = int(x.shape[1]) // 8
        key = (g, dtype)
        if key not in self._kernel_cache:
            h = self.wavelets
            hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype=dtype)
            hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
            self._kernel_cache[key] = (hl.detach().clone(), hh.detach().clone())
        hl, hh = self._kernel_cache[key]

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)
        xll = F.conv_transpose3d(xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xll += F.conv_transpose3d(xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlh = F.conv_transpose3d(xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlh += F.conv_transpose3d(xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhl = F.conv_transpose3d(xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhl += F.conv_transpose3d(xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhh = F.conv_transpose3d(xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhh += F.conv_transpose3d(xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xl = F.conv_transpose3d(xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xl += F.conv_transpose3d(xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh = F.conv_transpose3d(xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh += F.conv_transpose3d(xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        x = F.conv_transpose3d(xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        x += F.conv_transpose3d(xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        if rescale:
            x = x * (2 * torch.sqrt(torch.tensor(2.0)))
        return x

    for m in decoder.modules():
        if isinstance(m, UnPatcher3D):
            m._kernel_cache = {}
            m._idwt = types.MethodType(cached_idwt, m)


def install_export_friendly_dwt(encoder):
    """Same trick for the encoder's Patcher3D forward haar transform (kernels
    built from x.shape in forward)."""
    import types

    import torch.nn.functional as F
    from cosmos_predict1.tokenizer.modules.patching import Patcher3D

    def cached_dwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        n = self.wavelets.shape[0]
        g = int(x.shape[1])
        key = (g, dtype)
        if key not in self._kernel_cache:
            h = self.wavelets
            hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
            hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
            self._kernel_cache[key] = (hl.detach().clone(), hh.detach().clone())
        hl, hh = self._kernel_cache[key]
        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    for m in encoder.modules():
        if isinstance(m, Patcher3D):
            m._kernel_cache = {}
            m._dwt = types.MethodType(cached_dwt, m)


class EncoderWrap(torch.nn.Module):
    """Return only the latent the decoder consumes (encoders may return tuples)."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        out = self.enc(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--part", default="decoder", choices=["decoder", "encoder", "both"])
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="/workspace/trt")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--width", type=int, default=854, help="crop width so every clip has one fixed shape")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--export_device", default="cuda", help="cuda (cudnn off) or cpu for tracing")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = select([args.tokenizer])[0]

    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, read_video, resize_video

    # --- calibration clips (fixed shape) + latents via the JIT encoder (proven on 4090) ---
    tokenizer_jit = build_tokenizer(tok, args, native=False)
    calib_files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
    videos, latents = [], []
    with torch.no_grad():
        for p in calib_files:
            video = resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side)[None]
            video = video[:, :, : args.short_side, : args.width]
            if video.shape[2:4] != (args.short_side, args.width):
                print(f"skip {p.name}: {video.shape[2:4]} smaller than {(args.short_side, args.width)}")
                continue
            padded, _ = pad_video_batch(video)
            x = numpy2tensor(padded, tokenizer_jit._dtype, args.device)
            videos.append(x.float().cpu().numpy())
            latents.append(tokenizer_jit.encode(x)[0].float().cpu().numpy())
    calib_videos = np.concatenate(videos, axis=0)
    calib_latents = np.concatenate(latents, axis=0)
    np.save(out_dir / "calib_latents.npy", calib_latents)
    np.save(out_dir / "calib_videos.npy", calib_videos)
    print(f"calib videos {calib_videos.shape}, latents {calib_latents.shape}")
    del tokenizer_jit
    torch.cuda.empty_cache()

    tokenizer = build_tokenizer(tok, args, native=True)
    torch.backends.cudnn.enabled = False  # dodge 4090 cuDNN conv3d faults; one-off trace, perf irrelevant
    dev = args.export_device

    if args.part in ("decoder", "both"):
        decoder = tokenizer._dec_model.float().eval().to(dev)
        install_export_friendly_idwt(decoder)
        example = torch.from_numpy(calib_latents[:1]).to(dev)
        with torch.no_grad():
            decoder(example)  # eager warmup fills the kernel cache before tracing
        torch.onnx.export(decoder, example, str(out_dir / "decoder.onnx"),
                          input_names=["latent"], output_names=["video"], opset_version=17)
        print(f"exported {out_dir/'decoder.onnx'} (input {tuple(example.shape)})")

    if args.part in ("encoder", "both"):
        encoder = EncoderWrap(tokenizer._enc_model.float().eval()).to(dev)
        install_export_friendly_dwt(encoder)
        example = torch.from_numpy(calib_videos[:1]).to(dev)
        with torch.no_grad():
            out = encoder(example)
        print(f"encoder eager ok: {tuple(example.shape)} -> {tuple(out.shape)}")
        torch.onnx.export(encoder, example, str(out_dir / "encoder.onnx"),
                          input_names=["video"], output_names=["latent"], opset_version=17)
        print(f"exported {out_dir/'encoder.onnx'}")


if __name__ == "__main__":
    main()
