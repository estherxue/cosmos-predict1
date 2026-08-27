# SPDX-License-Identifier: Apache-2.0
"""Export the native decoder to ONNX (fp32, fixed shape) + save calibration latents.

Produces decoder.onnx and calib_latents.npy (encoder outputs of DAVIS-train clips,
for modelopt ONNX INT8 quantization).

Usage (from the repo root):
    python eval_quality/export_onnx.py --tokenizer 0.1-CV8x8x8 --out_dir /workspace/trt
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="/workspace/trt")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = select([args.tokenizer])[0]

    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, read_video, resize_video

    # Calibration latents via the JIT encoder — the native eager encoder hits
    # CUDA misaligned-address/illegal-access errors on 4090 at 480p.
    tokenizer_jit = build_tokenizer(tok, args, native=False)
    calib_files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
    latents, ref_hw = [], None
    with torch.no_grad():
        for p in calib_files:
            video = resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side)[None]
            ref_hw = ref_hw or video.shape[2:4]  # crop to a common size (DAVIS aspect ratios vary)
            video = video[:, :, : ref_hw[0], : ref_hw[1]]
            if video.shape[2:4] != ref_hw:
                print(f"skip {p.name}: smaller than reference {ref_hw}")
                continue
            padded, _ = pad_video_batch(video)
            latents.append(tokenizer_jit.encode(numpy2tensor(padded, tokenizer_jit._dtype, args.device))[0].float().cpu().numpy())
    calib = np.concatenate(latents, axis=0)
    np.save(out_dir / "calib_latents.npy", calib)
    print(f"calib latents: {calib.shape} -> {out_dir/'calib_latents.npy'}")
    del tokenizer_jit
    torch.cuda.empty_cache()

    tokenizer = build_tokenizer(tok, args, native=True)
    torch.backends.cudnn.enabled = False  # dodge 4090 cuDNN conv3d faults; one-off trace, perf irrelevant
    decoder = tokenizer._dec_model.float().eval()
    install_export_friendly_idwt(decoder)
    example = torch.from_numpy(calib[:1]).to(args.device)
    with torch.no_grad():
        decoder(example)  # eager warmup fills the kernel cache before tracing
    torch.onnx.export(
        decoder, example, str(out_dir / "decoder.onnx"),
        input_names=["latent"], output_names=["video"], opset_version=17,
    )
    print(f"exported {out_dir/'decoder.onnx'} (input {tuple(example.shape)})")


if __name__ == "__main__":
    main()
