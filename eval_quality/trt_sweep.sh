#!/usr/bin/env bash
# Sweep modelopt ONNX INT8 configs for encoder+decoder, build TRT engines, and
# run the end-to-end eval (quality + QPS) for each. Run from the repo root on the pod.
# Usage: bash eval_quality/trt_sweep.sh <config_name> [<config_name> ...]
set -x
TRT=/workspace/trt
DEC_EXCL='.*conv_out.*|.*norm_out.*|.*up[./]0[./].*'
ENC_EXCL='.*conv_in.*|.*patcher.*|.*down[./]0[./].*'

quant() {  # quant <part> <suffix> <extra flags...>
  local part=$1 suffix=$2; shift 2
  local data=$TRT/calib_latents.npy; [ "$part" = encoder ] && data=$TRT/calib_videos.npy
  python3 -m modelopt.onnx.quantization --onnx_path $TRT/$part.onnx --quantize_mode int8 \
    --calibration_data $data --output_path $TRT/${part}_$suffix.onnx "$@"
}

run_cfg() {  # run_cfg <name> <flags...>  (flags shared by encoder and decoder)
  local name=$1; shift
  quant encoder $name "$@" || return 1
  quant decoder $name "$@" || return 1
  python3 eval_quality/trt_bench.py --onnx $TRT/encoder_$name.onnx --tag int8 || return 1
  python3 eval_quality/trt_bench.py --onnx $TRT/decoder_$name.onnx --tag int8 || return 1
  python3 eval_quality/trt_eval.py --enc_engine $TRT/encoder_$name.int8.engine \
    --dec_engine $TRT/decoder_$name.int8.engine --tag trt_$name
}

for cfg in "$@"; do
  case $cfg in
    hp16)       run_cfg hp16 --high_precision_dtype fp16 ;;
    hp16_max)   run_cfg hp16_max --high_precision_dtype fp16 --calibration_method max ;;
    hp16_ct)    run_cfg hp16_ct --high_precision_dtype fp16 --op_types_to_quantize Conv ConvTranspose MatMul ;;
    hp16_mixed) # protect quality-sensitive blocks (decoder head/up.0, encoder front)
                quant encoder hp16_mixed --high_precision_dtype fp16 --nodes_to_exclude "$ENC_EXCL" || continue
                quant decoder hp16_mixed --high_precision_dtype fp16 --nodes_to_exclude "$DEC_EXCL" || continue
                python3 eval_quality/trt_bench.py --onnx $TRT/encoder_hp16_mixed.onnx --tag int8
                python3 eval_quality/trt_bench.py --onnx $TRT/decoder_hp16_mixed.onnx --tag int8
                python3 eval_quality/trt_eval.py --enc_engine $TRT/encoder_hp16_mixed.int8.engine \
                  --dec_engine $TRT/decoder_hp16_mixed.int8.engine --tag trt_hp16_mixed ;;
    hp16_decmixed) # encoder fully int8, decoder protected
                quant encoder hp16_decmixed --high_precision_dtype fp16 || continue
                quant decoder hp16_decmixed --high_precision_dtype fp16 --nodes_to_exclude "$DEC_EXCL" || continue
                python3 eval_quality/trt_bench.py --onnx $TRT/encoder_hp16_decmixed.onnx --tag int8
                python3 eval_quality/trt_bench.py --onnx $TRT/decoder_hp16_decmixed.onnx --tag int8
                python3 eval_quality/trt_eval.py --enc_engine $TRT/encoder_hp16_decmixed.int8.engine \
                  --dec_engine $TRT/decoder_hp16_decmixed.int8.engine --tag trt_hp16_decmixed ;;
    *) echo "unknown config $cfg" ;;
  esac
  echo "SWEEP_CFG_DONE_$cfg"
done
echo "SWEEP_ALL_DONE"
