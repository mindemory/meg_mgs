#!/usr/bin/env bash
# run_glue_capacity.sh
#
# Parallelised glue manifold-capacity runner across subjects. Launches one
# background job per SUBJECT (manifold_capacity.py loops bands x rois x
# epochs internally for that one subject), with a concurrency limit that
# defaults to the number of subjects -- i.e. by default every subject runs
# concurrently.
#
# Requires the `glue` package (github.com/cnchou/glue) importable -- run
# this from whichever conda env has it (e.g. `conda activate eegmne` on
# vader) BEFORE launching this script, since it just calls `python3`.
#
# Once all jobs complete, calls aggregate_glue_capacity.py to build the
# cross-subject bar-plot figures.
#
# Usage:
#   bash run_glue_capacity.sh [voxRes] [max_parallel] [n_hyperplanes] [seed] [force]
#
# Examples:
#   bash run_glue_capacity.sh                       # 8mm, max parallel = n_subjects, n_hyperplanes=200
#   bash run_glue_capacity.sh 8mm 21 200 42 true     # overwrite existing per-subject CSVs

set -euo pipefail

VOX_RES="${1:-8mm}"
N_HYPERPLANES="${3:-200}"
SEED="${4:-42}"
# Any of true/1/yes (case-insensitive) forces manifold_capacity.py to
# overwrite an existing per-subject results CSV instead of skipping it.
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta lowgamma highgamma)
ROIS=(visual parietal frontal)
EPOCHS=(stim delay)

# Default max parallel = number of subjects. Override with arg 2.
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/manifold_capacity.py"
AGG_SCRIPT="${SCRIPT_DIR}/glue_decoding/aggregate_glue_capacity.py"

# Single-threaded BLAS inside every job -- outer parallel grid handles CPU utilization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_glue_capacity_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Glue Manifold-Capacity Runner"
echo " VoxRes         : ${VOX_RES}"
echo " Max Parallel   : ${MAX_PARALLEL}"
echo " N Hyperplanes  : ${N_HYPERPLANES}"
echo " Seed           : ${SEED}"
echo " Force          : ${FORCE_RAW}"
echo " Subjects       : ${SUBJ_LIST[*]}"
echo " Bands          : ${BANDS[*]}"
echo " ROIs           : ${ROIS[*]}"
echo " Epochs         : ${EPOCHS[*]}"
echo " Jobs           : ${#SUBJ_LIST[@]} (one per subject)"
echo " Logging to     : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    log_file="${LOG_DIR}/sub-${subjID}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ..."

    ( python3 "${CELL_SCRIPT}" \
          --subjID "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --epochs "${EPOCHS[@]}" \
          --n_hyperplanes "${N_HYPERPLANES}" \
          --seed "${SEED}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${log_file}" 2>&1 &

    count=$((count + 1))
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL}). Waiting for batch to complete..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
    fi
done

# Wait for remaining cell jobs
wait
echo ""
echo "[$(date '+%H:%M:%S')] All per-subject glue capacity jobs finished."

# ── Aggregation / plotting ──────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Running aggregator/plotter ..."
echo "========================================================"

python3 "${AGG_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --rois "${ROIS[@]}" \
    --epochs "${EPOCHS[@]}" \
    > "${LOG_DIR}/aggregator.log" 2>&1

echo "[$(date '+%H:%M:%S')] Aggregator finished. Log: ${LOG_DIR}/aggregator.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
