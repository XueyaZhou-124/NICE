#!/usr/bin/env bash
# Compute target-region methylation from single5mC and feature BEDs.
# Usage: ./run_target_methylation.sh <SAMPLE> <SINGLEC_FILE> <FEATURES_DIR> <OUT_DIR> [R_SCRIPT]
# SINGLEC_FILE: e.g. sample_id_02.single5mC
# FEATURES_DIR: directory of BED/feature files
# OUT_DIR: writes <feature_name>.site_methylation.rate.xls per feature
# R_SCRIPT: optional R script for methylation rate (default: calculate_methylation_rate.R in same dir or PATH)

set -e
SAMPLE="${1:?}"
SINGLEC_FILE="${2:?}"
FEATURES_DIR="${3:?}"
OUT_DIR="${4:?}"
R_SCRIPT="${5:-}"

mkdir -p "$OUT_DIR"
# CpG with depth >= 3
grep -w 'CpG' "$SINGLEC_FILE" | awk 'BEGIN{OFS="\t"}{if($5>=3){print $1,$2-1,$2,$5,$6}}' | bedtools sort -i - > "${SAMPLE}.CpG_sorted.bed"

for feature_path in "$FEATURES_DIR"/*; do
  [[ -e "$feature_path" ]] || continue
  feature_name=$(basename "$feature_path" .bed)
  feature_name=$(basename "$feature_name" .txt)
  bedtools intersect -a "$feature_path" -b "${SAMPLE}.CpG_sorted.bed" -wb -wa | awk '{OFS="\t"; print $1,$2,$3,$7,$8}' > "${SAMPLE}_${feature_name}.single5mC"
  bedtools groupby -g 1,2,3 -c 4,5 -o sum,sum -i "${SAMPLE}_${feature_name}.single5mC" | awk '{OFS="\t"; print $1,$2,$3,$4,$5,$5/$4}' > "${SAMPLE}_${feature_name}.single5mC2"
  if [[ -n "$R_SCRIPT" && -f "$R_SCRIPT" ]]; then
    Rscript "$R_SCRIPT" "${SAMPLE}_${feature_name}.single5mC2" "${OUT_DIR}/${feature_name}"
  else
    # minimal: just copy or leave table
    cp "${SAMPLE}_${feature_name}.single5mC2" "${OUT_DIR}/${feature_name}.site_methylation.rate.xls" 2>/dev/null || true
  fi
  rm -f "${SAMPLE}_${feature_name}.single5mC" "${SAMPLE}_${feature_name}.single5mC2"
done
rm -f "${SAMPLE}.CpG_sorted.bed"
echo "----- ${SAMPLE} target methylation completed -----"
