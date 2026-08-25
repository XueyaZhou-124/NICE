#!/usr/bin/env bash
# Parallel Stage1 predict_and_filter for multiple samples on one node.
# Usage:
#   ./run_stage1_predict_parallel.sh <sample_list> [BAM_ROOT] [MODEL_PATH] [RES_DIR] [THRESHOLD] [PARALLEL_JOBS]

set -u

SAMPLE_LIST="${1:?Usage: $0 sample_list.txt [BAM_ROOT] [MODEL_PATH] [RES_DIR] [THRESHOLD] [PARALLEL_JOBS]}"
BAM_ROOT="${2:-.}"
MODEL_PATH="${3:?Set MODEL_PATH or pass as 3rd arg}"
RES_DIR="${4:-./scores}"
THRESHOLD="${5:-0.2}"
PARALLEL_JOBS="${6:-6}"
TMP_ROOT="${STAGE1_TMP_ROOT:-${SLURM_TMPDIR:-./tmp_reads_parallel}}"
THRESH_TAG="$(echo "${THRESHOLD}" | tr -d '.')"
SUCCESS_FILE="${RES_DIR}/success_samples.txt"
FAIL_FILE="${RES_DIR}/failed_samples.txt"

mkdir -p "${RES_DIR}" "${TMP_ROOT}"
: > "${SUCCESS_FILE}"
: > "${FAIL_FILE}"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
fi

run_one_sample() {
  local sample="$1"
  sample="$(echo "${sample}" | tr -d '\r')"
  [[ -z "${sample}" ]] && return 0

  local bam_path="${BAM_ROOT}/${sample}/${sample}_id.bam"
  local csv_path="${RES_DIR}/${sample}_id.csv"
  local out_bam="${BAM_ROOT}/${sample}/${sample}_id_${THRESH_TAG}.bam"
  local reads_dir="${TMP_ROOT}/${sample}"

  if [[ ! -f "${bam_path}" ]]; then
    echo "Skip (no id BAM): ${bam_path}"
    return 0
  fi

  if [[ -f "${csv_path}" && -f "${out_bam}" ]]; then
    echo "Skip (already done): ${sample}"
    echo "${sample}" >> "${SUCCESS_FILE}"
    return 0
  fi

  mkdir -p "${reads_dir}"
  echo "Running: ${sample}"
  if python -m stage1_filtering.predict_and_filter \
    --bam_path "${bam_path}" \
    --model_path "${MODEL_PATH}" \
    --reads_dir "${reads_dir}" \
    --res_dir "${RES_DIR}" \
    --threshold "${THRESHOLD}"; then
    echo "${sample}" >> "${SUCCESS_FILE}"
  else
    echo "Failed: ${sample}"
    echo "${sample}" >> "${FAIL_FILE}"
  fi
  rm -rf "${reads_dir}"
}

running_jobs=0
while IFS= read -r sample || [[ -n "${sample}" ]]; do
  sample="$(echo "${sample}" | tr -d '\r')"
  [[ -z "${sample}" ]] && continue

  run_one_sample "${sample}" &
  running_jobs=$((running_jobs + 1))

  if (( running_jobs >= PARALLEL_JOBS )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done < "${SAMPLE_LIST}"

wait

fail_count=$(wc -l < "${FAIL_FILE}" | tr -d ' ')
echo "Tmp root used: ${TMP_ROOT}"
echo "Done. success=$(wc -l < "${SUCCESS_FILE}" | tr -d ' '), failed=${fail_count}"
if (( fail_count > 0 )); then
  exit 1
fi
