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
    tokenizer = build_tokenizer(tok, args, native=True)

    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, read_video, resize_video

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
            latents.append(tokenizer.encode(numpy2tensor(padded, tokenizer._dtype, args.device))[0].float().cpu().numpy())
    calib = np.concatenate(latents, axis=0)
    np.save(out_dir / "calib_latents.npy", calib)
    print(f"calib latents: {calib.shape} -> {out_dir/'calib_latents.npy'}")

    decoder = tokenizer._dec_model.float().eval()
    example = torch.from_numpy(calib[:1]).to(args.device)
    torch.onnx.export(
        decoder, example, str(out_dir / "decoder.onnx"),
        input_names=["latent"], output_names=["video"], opset_version=17,
    )
    print(f"exported {out_dir/'decoder.onnx'} (input {tuple(example.shape)})")


if __name__ == "__main__":
    main()
