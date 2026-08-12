#!/usr/bin/env bash
# run_decoding_ts.sh
#
# Parallelised linear decoding-over-time runner across subjects x bands.
#
# Launches one background job per (SUBJECT, BAND) pair (decoding_ts_cell.py
# still loops conditions/ROIs internally for that one band), with a
# concurrency limit that defaults to the number of subjects -- i.e. by
# default every subject gets to run one band at a time in parallel, and
# bands for a given subject queue up behind each other rather than all
# piling on at once.
# Once all jobs complete, calls plot_decoding_ts.py to aggregate results and
# create figures.
#
# Usage:
#   bash run_decoding_ts.sh [voxRes] [max_parallel] [win_ms] [n_shuffle] [force]
#
# Examples:
#   bash run_decoding_ts.sh                     # 8mm, max parallel = n_subjects
#   bash run_decoding_ts.sh 8mm 10 50 100       # explicit parameters
#   bash run_decoding_ts.sh 8mm 21 50 100 true  # overwrite existing .npz

set -euo pipefail

VOX_RES="${1:-8mm}"
WIN_MS="${3:-50}"
# decoding_ts_cell.py's decoder is a closed-form Ridge LOO with an O(1)-per-
# shuffle shortcut (see its module docstring), so n_shuffle=100 is cheap --
# matches decoding_ts_cell.py's own DEFAULT_N_SHUFFLE.
N_SHUFFLE="${4:-100}"
# Any of true/1/yes (case-insensitive) forces decoding_ts_cell.py to
# overwrite existing .npz outputs instead of skipping them.
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
# Portable case-insensitive match (avoids bash 4+ ${VAR,,} lowercasing,
# which macOS's default /bin/bash 3.2 doesn't support).
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
# highgamma is ampOnly-only -- decoding_ts_cell.py silently skips it for
# ampPhase (phase is only available for theta/alpha/beta, per
# AMP_PHASE_BANDS in constants.py), so listing it here is safe.
BANDS=(theta alpha beta lowgamma highgamma)
CONDITIONS=(ampOnly ampPhase)
ROIS=(visual parietal frontal)

# Default max parallel = number of subjects (one band in flight per subject
# at a time, all subjects going concurrently). Override with arg 2.
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/decoding_ts_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/plot_decoding_ts.py"

# Single-threaded BLAS inside every job -- outer parallel grid handles CPU utilization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_decoding_ts_${VOX_RES}"
mkdir -p "${LOG_DIR}"

N_JOBS=$(( ${#SUBJ_LIST[@]} * ${#BANDS[@]} ))

echo "========================================================"
echo " Linear Decoding-Over-Time Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Window (ms)  : ±${WIN_MS} ms"
echo " N Shuffle    : ${N_SHUFFLE}"
echo " Force        : ${FORCE_RAW}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Bands        : ${BANDS[*]}"
echo " Conditions   : ${CONDITIONS[*]}"
echo " ROIs         : ${ROIS[*]}"
echo " Jobs         : ${N_JOBS} (subjects x bands)"
echo " Logging to   : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    for band in "${BANDS[@]}"; do
        log_file="${LOG_DIR}/sub-${subjID}_${band}.log"
        echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} band=${band} ..."

        ( python3 "${CELL_SCRIPT}" \
              "${subj_num}" \
              --bands "${band}" \
              --voxRes "${VOX_RES}" \
              --rois "${ROIS[@]}" \
              --conditions "${CONDITIONS[@]}" \
              --win_ms "${WIN_MS}" \
              --n_shuffle "${N_SHUFFLE}" \
              ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
        ) > "${log_file}" 2>&1 &

        count=$((count + 1))
        if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL}). Waiting for batch to complete..."
            wait
            echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
        fi
    done
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
