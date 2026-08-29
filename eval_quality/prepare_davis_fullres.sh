#!/usr/bin/env bash
# Download DAVIS 2017 trainval at Full-Resolution (1080p) for the official-protocol eval.
# Usage: bash eval_quality/prepare_davis_fullres.sh [download_dir]
set -ex
DAVIS_DIR=${1:-/workspace/davis_fr}
mkdir -p "$DAVIS_DIR" && cd "$DAVIS_DIR"
if [ ! -d DAVIS/JPEGImages/Full-Resolution ]; then
  wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
  unzip -q DAVIS-2017-trainval-Full-Resolution.zip
  rm -f DAVIS-2017-trainval-Full-Resolution.zip
fi
echo "FULLRES_READY: $(ls DAVIS/JPEGImages/Full-Resolution | wc -l) sequences"
