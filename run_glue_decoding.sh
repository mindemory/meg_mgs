#!/usr/bin/env bash
# run_glue_decoding.sh
#
# Runs glue_decoding/run_glue_cell.py across all subjects on Vader.
#
# Configuration:
#   - Stim-locked only (resp-locked excluded)
#   - Conditions: ampOnly + ampPhase  (unfiltered/broadband excluded)
#   - Bands: theta alpha beta lowgamma  (highgamma excluded)
#
# One background job per subject -- 21 jobs, comfortably under vader's
# ~48 cores.  Each job loops ROI x condition x band SEQUENTIALLY,
# single-threaded.  Skip-if-exists is handled inside run_glue_cell.py
# per output file, so re-running only computes what is missing.
#
# Usage:
#   bash run_glue_decoding.sh [voxRes] [max_parallel]
#
# Examples:
#   bash run_glue_decoding.sh               # 8mm, up to 21 parallel
#   bash run_glue_decoding.sh 8mm 10        # throttle to 10

set -euo pipefail

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-21}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
LOCK_TYPE="stim"
CONDITIONS=(ampOnly ampPhase)
BANDS=(theta alpha beta lowgamma)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force single-threaded BLAS inside every job -- the outer subject grid IS
# the parallelism; oversubscribing cores here makes each job slower.
# run_glue_cell.py also sets these itself (belt-and-suspenders).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========================================================"
echo " glue_decoding Runner"
echo " VoxRes      : ${VOX_RES}"
echo " Max Parallel: ${MAX_PARALLEL}"
echo " Subjects    : ${SUBJ_LIST[*]}"
echo " Lock type   : ${LOCK_TYPE}"
echo " Conditions  : ${CONDITIONS[*]}"
echo " Bands       : ${BANDS[*]}"
echo "========================================================"

LOG_DIR="logs_glue_decoding_${VOX_RES}"
mkdir -p "${LOG_DIR}"
echo "Logging to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    log_file="${LOG_DIR}/sub-${subjID}_${LOCK_TYPE}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ${LOCK_TYPE} ..."

    ( python3 "${SCRIPT_DIR}/glue_decoding/run_glue_cell.py" \
          "${subj_num}" "${LOCK_TYPE}" \
          --voxRes      "${VOX_RES}" \
          --conditions  "${CONDITIONS[@]}" \
          --bands       "${BANDS[@]}" \
    ) > "${log_file}" 2>&1 &

    count=$((count + 1))
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel (${MAX_PARALLEL}). Waiting..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
    fi
done

wait
echo ""
echo "========================================================"
echo " All subjects completed!  $(date)"
echo "========================================================"
