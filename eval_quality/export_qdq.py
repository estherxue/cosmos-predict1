# SPDX-License-Identifier: Apache-2.0
"""Export Q/DQ ONNX (encoder + decoder) from a modelopt fake-quant PyTorch model.

Quantization params come from the same PyTorch fake-quant model we evaluate
(PTQ calibration here, or a QAT state dict via --qat_state), so the deployed
TensorRT engine and the measured quality share one set of scales. This also
sidesteps modelopt.onnx's graph preprocessing (which produced a cyclic graph
for this encoder).

Usage (from the repo root):
    python eval_quality/export_qdq.py --tag full --out_dir /workspace/trt
    python eval_quality/export_qdq.py --tag mixed --keep_bf16 conv_out norm_out up.0 conv_in patcher down.0
    python eval_quality/export_qdq.py --tag qat --qat_state eval_quality/results_quant/qat/qat_state.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import EncoderWrap, install_export_friendly_dwt, install_export_friendly_idwt
from run_eval import VIDEO_EXTS, build_tokenizer
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--keep_bf16", nargs="*", default=[])
    parser.add_argument("--qat_state", default=None, help="state dict from qat.py (skips calibration)")
    parser.add_argument("--calib_dir", default="/workspace/davis_calib")
    parser.add_argument("--calib_n", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="/workspace/trt")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch", type=int, default=1, help="fixed batch size of the exported graph")
    parser.add_argument("--part", default="both", choices=["both", "encoder", "decoder"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.enabled = False  # native eager conv3d faults on 4090 with cuDNN

    import modelopt.torch.quantization as mtq
    import modelopt.torch.quantization.export_onnx  # noqa: F401  (registers Q/DQ symbolics)
    from cosmos_predict1.tokenizer.inference.utils import numpy2tensor, pad_video_batch, read_video, resize_video

    tok = select([args.tokenizer])[0]
    tokenizer = build_tokenizer(tok, args, native=True)

    files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
    batches = []
    for p in files:
        v = resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side)[None]
        v = v[:, :, : args.short_side, : args.width]
        if v.shape[2:4] == (args.short_side, args.width):
            batches.append(pad_video_batch(v)[0])
    print(f"{len(batches)} calibration clips")

    def forward_loop(_m):
        with torch.no_grad():
            for b in batches:
                x = numpy2tensor(b, tokenizer._dtype, args.device)
                tokenizer.decode(tokenizer.encode(x)[0])

    mtq.quantize(tokenizer, make_config(mtq, args.keep_bf16), forward_loop=forward_loop)
    if args.qat_state:
        sd = torch.load(args.qat_state, map_location=args.device)
        missing, unexpected = tokenizer.load_state_dict(sd, strict=False)
        print(f"loaded QAT state: {len(missing)} missing, {len(unexpected)} unexpected keys")
    mtq.print_quant_summary(tokenizer)

    tokenizer = tokenizer.float().eval()
    video_np = np.concatenate(batches[: args.batch], axis=0)
    video_np = np.resize(video_np, (args.batch, *video_np.shape[1:]))
    example_video = numpy2tensor(video_np, torch.float32, args.device)

    encoder = EncoderWrap(tokenizer._enc_model)
    install_export_friendly_dwt(encoder)
    with torch.no_grad():
        latent = encoder(example_video)
    if args.part in ("both", "encoder"):
        torch.onnx.export(encoder, example_video, str(out_dir / f"encoder_{args.tag}.onnx"),
                          input_names=["video"], output_names=["latent"], opset_version=17, dynamo=False)
        print(f"exported encoder_{args.tag}.onnx  {tuple(example_video.shape)} -> {tuple(latent.shape)}")
    del encoder
    torch.cuda.empty_cache()

    if args.part in ("both", "decoder"):
        decoder = tokenizer._dec_model
        install_export_friendly_idwt(decoder)
        with torch.no_grad():
            decoder(latent)
        torch.onnx.export(decoder, latent, str(out_dir / f"decoder_{args.tag}.onnx"),
                          input_names=["latent"], output_names=["video"], opset_version=17, dynamo=False)
        print(f"exported decoder_{args.tag}.onnx  (batch {args.batch})")


if __name__ == "__main__":
    main()
