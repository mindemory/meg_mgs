#!/usr/bin/env bash
# run_representational_distance_ts.sh
#
# Parallelised representational-distance-over-time runner across subjects.
# Launches one background job per SUBJECT (representational_distance_ts_cell.py
# loops bands x rois x conditions internally for that one subject), with a
# concurrency limit that defaults to the number of subjects.
#
# Pure numpy/scipy -- unlike run_glue_capacity.sh, this does NOT need the
# `glue` conda env; run it with this repo's normal Python environment.
#
# Once all jobs complete, calls plot_representational_distance_ts.py to
# aggregate results and create figures.
#
# Usage:
#   bash run_representational_distance_ts.sh [voxRes] [max_parallel] [n_perm] [seed] [force]
#
# Examples:
#   bash run_representational_distance_ts.sh                       # 8mm, max parallel = n_subjects, n_perm=1000
#   bash run_representational_distance_ts.sh 8mm 21 1000 0 true     # overwrite existing per-cell .npz

set -euo pipefail

VOX_RES="${1:-8mm}"
N_PERM="${3:-1000}"
SEED="${4:-0}"
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
CONDITIONS=(ampOnly)

# Default max parallel = number of subjects. Override with arg 2.
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/representational_distance_ts_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/plot_representational_distance_ts.py"

# Single-threaded BLAS inside every job -- outer parallel grid handles CPU utilization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_repdist_ts_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Representational-Distance-Over-Time Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " N Perm       : ${N_PERM}"
echo " Seed         : ${SEED}"
echo " Force        : ${FORCE_RAW}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Bands        : ${BANDS[*]}"
echo " ROIs         : ${ROIS[*]}"
echo " Conditions   : ${CONDITIONS[*]}"
echo " Jobs         : ${#SUBJ_LIST[@]} (one per subject)"
echo " Logging to   : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    log_file="${LOG_DIR}/sub-${subjID}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ..."

    ( python3 "${CELL_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --n_perm "${N_PERM}" \
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

wait
echo ""
echo "[$(date '+%H:%M:%S')] All per-subject representational-distance jobs finished."

# ── Plotting ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Running plotter ..."
echo "========================================================"

python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --rois "${ROIS[@]}" \
    --conditions "${CONDITIONS[@]}" \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
