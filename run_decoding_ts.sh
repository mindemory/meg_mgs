#!/usr/bin/env bash
# run_decoding_ts.sh
#
# Parallelised linear decoding-over-time runner across subjects.
#
# Launches one background job per SUBJECT (decoding_ts_cell.py loops
# bands/conditions/ROIs internally via --bands), with a concurrency limit.
# Once all jobs complete, calls plot_decoding_ts.py to aggregate results and
# create figures.
#
# Usage:
#   bash run_decoding_ts.sh [voxRes] [max_parallel] [win_ms] [n_shuffle]
#
# Examples:
#   bash run_decoding_ts.sh                # 8mm, max 10 parallel jobs
#   bash run_decoding_ts.sh 8mm 10 50 100  # explicit parameters

set -euo pipefail

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-10}"
WIN_MS="${3:-50}"
# decoding_ts_cell.py's decoder is a closed-form Ridge LOO with an O(1)-per-
# shuffle shortcut (see its module docstring), so n_shuffle=100 is cheap --
# matches decoding_ts_cell.py's own DEFAULT_N_SHUFFLE.
N_SHUFFLE="${4:-100}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta lowgamma)
CONDITIONS=(ampOnly ampPhase)
ROIS=(visual parietal frontal)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/decoding_ts_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/plot_decoding_ts.py"

# Single-threaded BLAS inside every job -- outer parallel grid handles CPU utilization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_decoding_ts_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Linear Decoding-Over-Time Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Window (ms)  : ±${WIN_MS} ms"
echo " N Shuffle    : ${N_SHUFFLE}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Bands        : ${BANDS[*]}"
echo " Conditions   : ${CONDITIONS[*]}"
echo " ROIs         : ${ROIS[*]}"
echo " Logging to   : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    log_file="${LOG_DIR}/sub-${subjID}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (bands: ${BANDS[*]}) ..."

    ( python3 "${CELL_SCRIPT}" \
          "${subj_num}" \
          --bands "${BANDS[@]}" \
          --voxRes "${VOX_RES}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --win_ms "${WIN_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
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
echo "[$(date '+%H:%M:%S')] All cell decoding jobs finished."

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
