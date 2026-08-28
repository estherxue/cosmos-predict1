# SPDX-License-Identifier: Apache-2.0
"""Per-layer TensorRT profile: which layers actually run INT8, and where time goes.

Rebuilds the engine with ProfilingVerbosity.DETAILED, reads per-layer precision
from the EngineInspector, and collects per-layer times with IProfiler (mean over
N runs). Equivalent of `trtexec --profilingVerbosity=detailed --dumpProfile`,
done via the python API (the pip tensorrt wheel ships no trtexec binary).

Usage:
    python eval_quality/trt_profile.py --onnx /workspace/trt/decoder_int8.onnx --tag int8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import tensorrt as trt
import torch


def build_detailed(onnx_path: str, engine_path: Path, workspace_gb: int = 18):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves external weight files (decoder.onnx.data) next to the model
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("engine build failed")
    engine_path.write_bytes(engine_bytes)


class LayerTimer(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms = defaultdict(float)
        self.runs = 0

    def report_layer_time(self, layer_name, ms):
        self.ms[layer_name] += ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out_dir", default="/workspace/trt")
    args = parser.parse_args()

    engine_path = Path(args.onnx).with_suffix(f".{args.tag}.prof.engine")
    if not engine_path.exists():
        print(f"building {engine_path} (detailed verbosity)...")
        build_detailed(args.onnx, engine_path)

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())

    # Per-layer metadata (incl. precision) from the inspector.
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    meta = {}
    for layer in info.get("Layers", []):
        if isinstance(layer, dict):
            name = layer.get("Name", "?")
            outs = layer.get("Outputs", [])
            dtype = outs[0].get("Format/Datatype", "?") if outs else "?"
            meta[name] = {"type": layer.get("LayerType", "?"), "out_dtype": dtype,
                          "precision": layer.get("Precision", "?")}

    ctx = engine.create_execution_context()
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dt = torch.float32 if engine.get_tensor_dtype(name) == trt.DataType.FLOAT else torch.float16
        buffers[name] = torch.empty(shape, dtype=dt, device="cuda")
        ctx.set_tensor_address(name, buffers[name].data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(5):  # warmup without profiler
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    timer = LayerTimer()
    ctx.profiler = timer
    for _ in range(args.iters):
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    rows = []
    for name, total in timer.ms.items():
        m = meta.get(name, {})
        rows.append({"name": name, "mean_ms": round(total / args.iters, 4),
                     "type": m.get("type", "?"), "precision": m.get("precision", "?"),
                     "out_dtype": m.get("out_dtype", "?")})
    rows.sort(key=lambda r: -r["mean_ms"])
    total_ms = sum(r["mean_ms"] for r in rows)

    by_prec = defaultdict(float)
    for r in rows:
        key = r["precision"] if r["precision"] != "?" else r["out_dtype"]
        by_prec[key] += r["mean_ms"]
    print(f"[{args.tag}] total {total_ms:.2f} ms over {len(rows)} layers")
    for k, v in sorted(by_prec.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v:7.2f} ms  ({100*v/total_ms:4.1f}%)")
    print("top 12 layers:")
    for r in rows[:12]:
        print(f"  {r['mean_ms']:7.3f} ms  [{r['precision']:>6s}/{r['out_dtype']:<6s}] {r['type']:<14s} {r['name'][:80]}")

    out = Path(args.out_dir) / f"profile_{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "total_ms": round(total_ms, 3),
                               "by_precision": {k: round(v, 3) for k, v in by_prec.items()},
                               "layers": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
