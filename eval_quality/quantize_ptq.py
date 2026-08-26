# SPDX-License-Identifier: Apache-2.0
"""INT8 post-training quantization of a Cosmos tokenizer (nvidia-modelopt),
evaluated with the run_eval harness on the same data/metrics.

Loads the PyTorch-native model (JIT files as weight store), optionally
fake-quantizes it after calibration, then runs the standard eval. With
--quant none this is the native-path parity check (Phase 0).

Usage (from the repo root):
    python eval_quality/quantize_ptq.py --quant none                # parity check
    python eval_quality/quantize_ptq.py --quant w8a8_dec --calib_dir /workspace/davis_calib
Note: fake-quant runs measure QUALITY; their timings are not deployed-int8 speed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import IMAGE_EXTS, VIDEO_EXTS, autoencode_timed, build_tokenizer, eval_tokenizer
from tokenizers_registry import select

QUANT_CHOICES = ["none", "w8_dec", "w8a8_dec", "w8a8_all", "w8a8_dec_sq"]


def make_config(mtq, quant: str):
    import copy

    base = mtq.INT8_SMOOTHQUANT_CFG if quant.endswith("_sq") else mtq.INT8_DEFAULT_CFG
    cfg = copy.deepcopy(base)
    if quant == "w8_dec":  # weights-only: disable all activation/input quantizers
        for key, entry in cfg["quant_cfg"].items():
            if "input_quantizer" in key and isinstance(entry, dict):
                entry["enable"] = False
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="0.1-CV8x8x8")
    parser.add_argument("--quant", default="none", choices=QUANT_CHOICES)
    parser.add_argument("--calib_dir", default="/workspace/davis_calib", help="folder of calibration videos (DAVIS train)")
    parser.add_argument("--calib_n", type=int, default=16)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="/workspace/davis_eval")
    parser.add_argument("--output_dir", default=None, help="default: eval_quality/results_quant/<quant>")
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--short-side", type=int, default=480)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    args.output_dir = args.output_dir or f"eval_quality/results_quant/{args.quant}"

    tok = select([args.tokenizer])[0]
    print(f"Loading {tok['name']} natively (config-mapped, JIT weights, {args.dtype})")
    tokenizer = build_tokenizer(tok, args, native=True)

    if args.quant != "none":
        import modelopt.torch.quantization as mtq
        from cosmos_predict1.tokenizer.inference.utils import read_video, resize_video

        calib_files = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)[: args.calib_n]
        if not calib_files:
            raise FileNotFoundError(f"no calibration videos in {args.calib_dir}")
        calib_batches = [
            resize_video(read_video(str(p))[: args.max_frames], short_size=args.short_side)[None] for p in calib_files
        ]

        def forward_loop(_model):
            for batch in calib_batches:
                autoencode_timed(tokenizer, batch, tok["kind"], args.device)

        target = tokenizer if args.quant == "w8a8_all" else tokenizer._dec_model
        print(f"PTQ {args.quant}: calibrating on {len(calib_batches)} clips from {args.calib_dir}")
        mtq.quantize(target, make_config(mtq, args.quant), forward_loop=forward_loop)
        mtq.print_quant_summary(target)

    data_dir = Path(args.data_dir)
    images = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    videos = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    result = eval_tokenizer(tok, args, images, videos, tokenizer=tokenizer)
    result["quant"] = args.quant
    print(f"[{args.quant}] PSNR {result['psnr']:.2f} dB, SSIM {result['ssim']:.4f}")

    output_path = Path(args.output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"args": vars(args), "results": [result]}, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
