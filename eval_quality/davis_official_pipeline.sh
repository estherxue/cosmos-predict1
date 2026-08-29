#!/usr/bin/env bash
# Fresh-pod pipeline: official-protocol DAVIS (1080p, full sequences) reproduction
# for the JIT checkpoints + the quantized configs under the same protocol.
# Run on the pod: nohup bash /workspace/cosmos-predict1/eval_quality/davis_official_pipeline.sh > /workspace/davis_official.log 2>&1 &
set -x
python3 -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
cd /workspace && rm -rf cosmos-predict1 && git clone -q -b tokenizer-quality-eval https://github.com/estherxue/cosmos-predict1.git
cd /workspace/cosmos-predict1
python3 -m pip install -q -r eval_quality/requirements.txt "nvidia-modelopt[torch]"
python3 eval_quality/download_checkpoints.py --tokenizers CV4x8x8-360p CV8x8x8-720p 0.1-CV4x8x8 0.1-CV8x8x8 DV4x8x8-360p DV8x16x16-720p
bash eval_quality/prepare_davis.sh /workspace/davis /workspace/davis_calib train 16     # 480p calibration clips (as deployed)
bash eval_quality/prepare_davis_fullres.sh /workspace/davis_fr
echo STEP_SETUP_DONE
E="python3 eval_quality/eval_davis_official.py"
# anchor: paper Table 5 rows (targets: T1-CV4x8x8-360p 35.85, T1-CV8x8x8-720p 31.28 [paper window 121, we use 49], 0.1-CV4x8x8 32.80, 0.1-CV8x8x8 30.61)
$E --tokenizers CV4x8x8-360p CV8x8x8-720p --tag jit_tokenize1 && echo STEP_JIT_T1_DONE
$E --tokenizers 0.1-CV4x8x8 0.1-CV8x8x8 --tag jit_legacy && echo STEP_JIT_LEGACY_DONE
# optional: quantized config under the same protocol
$E --tokenizers 0.1-CV8x8x8 --mode fakequant --keep_bf16 conv_in patcher down.0 --tag int8_encmix_decfull && echo STEP_FQ_LEGACY_DONE
echo DAVIS_OFFICIAL_ALL_DONE
