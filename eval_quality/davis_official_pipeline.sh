#!/usr/bin/env bash
# 40-minute anchor: reproduce the official Cosmos-Tokenizer DAVIS numbers for the 0.1 checkpoints
# (project page: CV4x8x8 32.80 / DV4x8x8 28.81; paper Table 5: CV8x8x8 30.61) with the official
# inference path (window 49, bf16 JIT, native 1080p, whole sequences) and TokenBench metrics.
# Run on the pod: nohup bash /workspace/cosmos-predict1/eval_quality/davis_official_pipeline.sh > /workspace/davis_official.log 2>&1 &
set -x
python3 -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
cd /workspace && rm -rf cosmos-predict1 && git clone -q -b tokenizer-quality-eval https://github.com/estherxue/cosmos-predict1.git
cd /workspace/cosmos-predict1
python3 -m pip install -q -r eval_quality/requirements.txt
# evidence: the archived official repo's inference code == the copy we call
git clone -q --depth 1 https://github.com/NVIDIA/Cosmos-Tokenizer.git /workspace/Cosmos-Tokenizer
for f in video_lib.py utils.py; do
  diff -q /workspace/Cosmos-Tokenizer/cosmos_tokenizer/$f cosmos_predict1/tokenizer/inference/$f && echo "IDENTICAL $f" || echo "DIFFERS $f"
done
diff /workspace/Cosmos-Tokenizer/cosmos_tokenizer/video_lib.py cosmos_predict1/tokenizer/inference/video_lib.py | head -20
python3 eval_quality/download_checkpoints.py --tokenizers 0.1-CV4x8x8 0.1-CV8x8x8 0.1-DV4x8x8
bash eval_quality/prepare_davis_fullres.sh /workspace/davis_fr
echo STEP_SETUP_DONE
E="python3 eval_quality/eval_davis_official.py --temporal_window 49"
$E --tokenizers 0.1-CV4x8x8 --tag official_0.1 && echo STEP_CV4_DONE
$E --tokenizers 0.1-CV8x8x8 --tag official_0.1 && echo STEP_CV8_DONE
$E --tokenizers 0.1-DV4x8x8 --tag official_0.1 && echo STEP_DV4_DONE
echo DAVIS_OFFICIAL_ALL_DONE
