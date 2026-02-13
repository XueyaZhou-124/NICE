#!/usr/bin/env bash
# Batch run Stage1 predict_and_filter for multiple samples.
# Usage: ./run_stage1_predict_batch.sh <sample_list> [BAM_ROOT] [MODEL_PATH] [RES_DIR] [THRESHOLD]
# sample_list: one sample ID per line (directory name under BAM_ROOT).
# Default THRESHOLD=0.2; set to 1 to skip BAM filtering.

set -e
SAMPLE_LIST="${1:?Usage: $0 sample_list.txt [BAM_ROOT] [MODEL_PATH] [RES_DIR] [THRESHOLD]}"
BAM_ROOT="${2:-.}"
MODEL_PATH="${3:?Set MODEL_PATH or pass as 3rd arg}"
RES_DIR="${4:-./scores}"
THRESHOLD="${5:-0.2}"
READS_DIR="./tmp_reads"
mkdir -p "$RES_DIR" "$READS_DIR"

while IFS= read -r sample || [[ -n "$sample" ]]; do
  sample=$(echo "$sample" | tr -d '\r')
  [[ -z "$sample" ]] && continue
  bam_path="${BAM_ROOT}/${sample}/${sample}_id.bam"
  if [[ ! -f "$bam_path" ]]; then
    echo "Skip (no BAM): $bam_path"
    continue
  fi
  echo "Running: $sample"
  python -m stage1_decent.predict_and_filter \
    --bam_path "$bam_path" \
    --model_path "$MODEL_PATH" \
    --reads_dir "$READS_DIR" \
    --res_dir "$RES_DIR" \
    --threshold "$THRESHOLD"
done < "$SAMPLE_LIST"
echo "Done. Results in $RES_DIR"
