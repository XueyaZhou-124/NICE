#!/usr/bin/env bash
# Genome-wide methylation in fixed bins (e.g. 5Mb).
# Usage: ./run_gwm.sh <SAMPLE> <SINGLEC_FILE> <BIN_BED> <OUT_DIR>
# BIN_BED: e.g. hg19_lambda_5M.bed

set -e
SAMPLE="${1:?}"
SINGLEC_FILE="${2:?}"
BIN_BED="${3:?}"
OUT_DIR="${4:?}"

mkdir -p "$OUT_DIR"
grep -w 'CpG' "$SINGLEC_FILE" | awk 'BEGIN{OFS="\t"}{if($5>=3){print $1,$2-1,$2,$5,$6}}' | bedtools sort -i - > "${SAMPLE}.CpG_sorted.bed"
bedtools intersect -a "$BIN_BED" -b "${SAMPLE}.CpG_sorted.bed" -wb -wa | awk '{OFS="\t"; print $1,$2,$3,$7,$8}' > "${SAMPLE}_gwm.single5mC"
bedtools groupby -g 1,2,3 -c 4,5 -o sum,sum -i "${SAMPLE}_gwm.single5mC" | awk '{OFS="\t"; print $1,$2,$3,$4,$5,$5/$4}' > "${OUT_DIR}/${SAMPLE}_gwm.single5mC2"
rm -f "${SAMPLE}_gwm.single5mC" "${SAMPLE}.CpG_sorted.bed"
echo "----- ${SAMPLE} GWM completed -----"
