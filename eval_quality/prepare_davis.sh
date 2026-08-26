#!/usr/bin/env bash
# Build a DAVIS 2017 (480p) eval/calibration set for run_eval.py:
# per sequence, a lossless 17-frame mp4 plus the first frame as png.
# Usage: bash eval_quality/prepare_davis.sh [download_dir] [output_dir] [split] [limit]
set -ex
DAVIS_DIR=${1:-/workspace/davis}
OUT_DIR=${2:-/workspace/davis_eval}
SPLIT=${3:-val}
LIMIT=${4:-999}

mkdir -p "$DAVIS_DIR"
cd "$DAVIS_DIR"
if [ ! -d DAVIS ]; then
  wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
  unzip -q DAVIS-2017-trainval-480p.zip
fi

mkdir -p "$OUT_DIR"
while read -r seq; do
  ffmpeg -nostdin -y -loglevel error -start_number 0 -i "DAVIS/JPEGImages/480p/$seq/%05d.jpg" \
    -frames:v 17 -c:v libx264rgb -qp 0 "$OUT_DIR/$seq.mp4"
  ffmpeg -nostdin -y -loglevel error -i "DAVIS/JPEGImages/480p/$seq/00000.jpg" "$OUT_DIR/$seq.png"
done < <(head -n "$LIMIT" "DAVIS/ImageSets/2017/$SPLIT.txt")

echo "PREP_COMPLETE: $(ls "$OUT_DIR" | wc -l) files in $OUT_DIR"
