#!/usr/bin/env bash
# run_glue_decoding.sh
#
# Runs glue_decoding/run_glue_cell.py across all subjects x lock types on
# Vader. One background job per (subject, lockType) -- 21 subjects x 2 lock
# types = 42 jobs, fitting comfortably under vader's ~48 cores with headroom.
# Each job internally loops ROI x condition x band SEQUENTIALLY, single
# process, single-threaded (see below) -- glue_decoding's parallelism lives
# at exactly this one level (the outer subject x lockType grid), never
# inside a job, to avoid oversubscribing cores.
#
# Usage:
#   bash run_glue_decoding.sh [voxRes] [max_parallel]
#
# Examples:
#   bash run_glue_decoding.sh                  # all 21 subjects, both lock types, 8mm
#   bash run_glue_decoding.sh 8mm 10           # throttle to 10 concurrent jobs
#
# Skip-if-exists is handled INSIDE run_glue_cell.py, per output file (one
# file per condition x band), so re-running this script after a partial run
# only computes what's missing.

set -euo pipefail

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-42}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
LOCK_TYPES=("stim" "resp")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force single-threaded BLAS inside every job -- the outer subject x lockType
# grid IS the parallelism; oversubscribing cores here would make each job
# slower, not faster. run_glue_cell.py also sets these itself
# (belt-and-suspenders), but exporting here ensures they're set before the
# Python interpreter (and therefore numpy/sklearn) even starts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========================================================"
echo " glue_decoding Runner (Vader)"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Lock types   : ${LOCK_TYPES[*]}"
echo "========================================================"

LOG_DIR="logs_glue_decoding_${VOX_RES}"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    for lockType in "${LOCK_TYPES[@]}"; do
        log_file="${LOG_DIR}/sub-${subjID}_${lockType}.log"
        echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ${lockType} in background..."

        ( python3 "${SCRIPT_DIR}/glue_decoding/run_glue_cell.py" "${subj_num}" "${lockType}" \
              --voxRes "${VOX_RES}" ) > "$log_file" 2>&1 &

        count=$((count + 1))
        if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Reached max parallel limit ($MAX_PARALLEL). Waiting..."
            wait
            echo "[$(date '+%H:%M:%S')] Batch complete. Starting next batch..."
        fi
    done
done

wait
echo "========================================================"
echo " All subjects completed! $(date)"
echo "========================================================"
