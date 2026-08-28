# SPDX-License-Identifier: Apache-2.0
"""Build TensorRT engines (fp16, and int8 from a Q/DQ ONNX) and benchmark ms/frame.

TRT >=10 removed implicit int8 calibration, so the int8 engine must come from an
explicitly quantized (Q/DQ) ONNX — produce it with modelopt.onnx before calling:
    python -m modelopt.onnx.quantization --onnx_path decoder.onnx \
        --quantize_mode int8 --calibration_data calib_latents.npy \
        --calibration_shapes latent:8x16x3x60x108 --output_path decoder_int8.onnx

Usage:
    python eval_quality/trt_bench.py --onnx /workspace/trt/decoder.onnx --tag fp16
    python eval_quality/trt_bench.py --onnx /workspace/trt/decoder_int8.onnx --tag int8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


def build_engine(onnx_path: str, engine_path: Path, workspace_gb: int = 18, fp16: bool = True, opt_level: int = 3):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: explicit batch is the default, the old flag is gone
    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves external weight files (decoder.onnx.data) next to the model
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)  # int8 Q/DQ ONNX: non-quantized layers fall back to fp16
    config.builder_optimization_level = opt_level
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("engine build failed")
    engine_path.write_bytes(engine_bytes)
    return engine_path


def bench(engine_path: Path, n_warmup: int = 5, n_iter: int = 30, cuda_graph: bool = False):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()

    buffers, shapes = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = torch.float32 if engine.get_tensor_dtype(name) == trt.DataType.FLOAT else torch.float16
        buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
        shapes[name] = shape
        ctx.set_tensor_address(name, buffers[name].data_ptr())
    print("io:", shapes)

    stream = torch.cuda.Stream()
    graph = None
    if cuda_graph:  # capture the whole engine execution once; replay removes per-layer launch overhead
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ctx.execute_async_v3(stream.cuda_stream)
    times = []
    for i in range(n_warmup + n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if graph is not None:
            graph.replay()
        else:
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        torch.cuda.synchronize()
        if i >= n_warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--frames", type=int, default=17, help="video frames one latent decodes to (for ms/frame)")
    parser.add_argument("--out", default="/workspace/trt/bench.json")
    parser.add_argument("--no-fp16", action="store_true", help="build without the FP16 flag (pure fp32 fallback)")
    parser.add_argument("--workspace-gb", type=int, default=18)
    parser.add_argument("--opt-level", type=int, default=3, help="TRT builder optimization level (5 = max tactic search)")
    parser.add_argument("--cuda-graph", action="store_true", help="benchmark with CUDA graph replay")
    args = parser.parse_args()

    engine_path = Path(args.onnx).with_suffix(f".{args.tag}.engine")
    if not engine_path.exists():
        print(f"building {engine_path} ...")
        build_engine(args.onnx, engine_path, workspace_gb=args.workspace_gb, fp16=not args.no_fp16, opt_level=args.opt_level)
    ms, shapes = bench(engine_path, cuda_graph=args.cuda_graph)
    row = {"tag": args.tag, "onnx": args.onnx, "median_ms": round(ms, 2), "cuda_graph": args.cuda_graph, "opt_level": args.opt_level,
           "ms_per_frame": round(ms / args.frames, 3), "io": {k: list(v) for k, v in shapes.items()},
           "trt": trt.__version__, "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(row))
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows = [r for r in rows if r["tag"] != args.tag] + [row]
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
