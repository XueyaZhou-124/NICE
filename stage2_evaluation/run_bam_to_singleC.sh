#!/usr/bin/env bash
# Convert filtered BAM to single-base methylation (single5mC) using mpileup + perl script.
# Usage: ./run_bam_to_singleC.sh <REF_FA> <BAM_DIR> <SAMPLE_NAME> <BAM_SUFFIX> <OUT_SINGLEC_DIR> [PILEUP_SCRIPT]
# Example: ./run_bam_to_singleC.sh /path/to/hg19_lambda.fa /path/to/bam PBAT_C10_B1 _id_02.bam /path/to/singleC_output
# Optional: PILEUP_SCRIPT = path to singleC_metLevel.hg19.pl (or similar)

set -e
REF_FA="${1:?Usage: $0 REF_FA BAM_DIR SAMPLE BAM_SUFFIX OUT_SINGLEC_DIR [PILEUP_SCRIPT]}"
BAM_DIR="${2:?}"
SAMPLE="${3:?}"
BAM_SUFFIX="${4:-_id_02.bam}"
OUT_SINGLEC_DIR="${5:?}"
PILEUP_SCRIPT="${6:-}"   # e.g. singleC_metLevel.hg19.pl

BAM="${BAM_DIR}/${SAMPLE}/${SAMPLE}${BAM_SUFFIX}"
mkdir -p "${OUT_SINGLEC_DIR}/${SAMPLE}"
OUT_FILE="${OUT_SINGLEC_DIR}/${SAMPLE}/${SAMPLE}${BAM_SUFFIX%.bam}.single5mC"
PILEUP="${BAM_DIR}/${SAMPLE}/${SAMPLE}${BAM_SUFFIX%.bam}.pileup"

if [[ ! -f "$BAM" ]]; then
  echo "BAM not found: $BAM" >&2
  exit 1
fi

echo "---- ${SAMPLE} processing ----"
samtools view -h "$BAM" | samtools view -uSb /dev/stdin | samtools mpileup -O -f "$REF_FA" /dev/stdin > "$PILEUP"
if [[ -n "$PILEUP_SCRIPT" && -x "$PILEUP_SCRIPT" || -f "$PILEUP_SCRIPT" ]]; then
  perl "$PILEUP_SCRIPT" "$PILEUP" > "${OUT_FILE}.tmp"
  grep -v "lambda" "${OUT_FILE}.tmp" | grep -v "chrM" > "$OUT_FILE"
  rm -f "${OUT_FILE}.tmp" "$PILEUP"
else
  echo "PILEUP_SCRIPT not set or not found; leaving pileup at $PILEUP" >&2
fi
echo "------ ${SAMPLE} completed ------"
