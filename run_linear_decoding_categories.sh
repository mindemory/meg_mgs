#!/usr/bin/env bash
# run_linear_decoding_categories.sh
#
# Parallelised linear (closed-form ridge, one-vs-rest, LOO) decoding runner
# across subjects, at several category granularities (2/4/6/10, see
# constants.CATEGORY_SCHEMES). Launches one background job per SUBJECT
# (linear_decoding_categories_cell.py loops bands x rois x conditions x
# schemes internally), with a concurrency limit that defaults to the number
# of subjects.
#
# CLASSIFIER: closed-form ridge (NOT SVM -- an earlier version used SVM with
# exhaustive per-fold LOO refits and needed --cv/--n_splits/--alpha; those
# no longer exist -- see linear_decoding_categories_cell.py's module
# docstring for the full ridge/adaptive-alpha rewrite and the two
# correctness bugs found and fixed during validation). This is cheap: full
# auto-balanced points_per_category + n_shuffle=100 + all 4 schemes runs in
# well under a minute per subject (validated), not hours.
#
# ERP removal (grand trial-average subtracted from every trial, per band/
# roi/condition cell, before windowing) is ON by default in the cell
# script -- pass --no_erp_removal there directly if you want it off (no
# launcher flag for this, to keep the positional arg list from growing).
#
# Significance in the plotted figures is now a cluster-based permutation
# test (sign-flip, Maris & Oostenveld 2007) against chance, NOT FDR --
# see plot_linear_decoding_categories.py's module docstring.
#
# Pure numpy -- does NOT need the `glue` conda env; run with this repo's
# normal Python environment.
#
# Once all jobs complete, calls plot_linear_decoding_categories.py to
# aggregate results and create figures.
#
# Usage:
#   bash run_linear_decoding_categories.sh [voxRes] [max_parallel] [n_shuffle] [seed] [force] [points_per_category] [schemes]
#
# Examples:
#   bash run_linear_decoding_categories.sh                                # 8mm, n_shuffle=100, auto points_per_category
#   bash run_linear_decoding_categories.sh 8mm 21 100 0 true              # overwrite existing per-cell .npz
#   bash run_linear_decoding_categories.sh 8mm 21 100 0 false 10 "2 10"   # fixed 10 pts/category, schemes 2/10 only
#
# points_per_category: leave empty/unset (default) for auto -- each
# (subject, scheme) is balanced to that subject's own smallest category
# count for that scheme, NOT a fixed value shared across schemes (see
# linear_decoding_categories_cell.py's module docstring). Pass an integer
# to force a fixed cap instead.

set -euo pipefail

VOX_RES="${1:-8mm}"
N_SHUFFLE="${3:-100}"
SEED="${4:-0}"
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac
POINTS_PER_CATEGORY="${6:-}"
PPC_FLAG=()
if [ -n "${POINTS_PER_CATEGORY}" ]; then
    PPC_FLAG=(--points_per_category "${POINTS_PER_CATEGORY}")
fi
SCHEMES_RAW="${7:-2 4 6 10}"
SCHEMES=(${SCHEMES_RAW})

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta lowgamma highgamma)
ROIS=(visual parietal frontal)
CONDITIONS=(ampOnly)
WIN_MS=50
TIME_STRIDE_MS=50

# Default max parallel = number of subjects. Override with arg 2.
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/linear_decoding_categories_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/plot_linear_decoding_categories.py"

# Single-threaded BLAS inside every job -- outer parallel grid handles CPU utilization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_lindecode_cat_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Linear Decoding by Category Runner (ridge, adaptive alpha)"
echo " VoxRes         : ${VOX_RES}"
echo " Max Parallel   : ${MAX_PARALLEL}"
echo " N Shuffle      : ${N_SHUFFLE}"
echo " Win (ms)       : +-${WIN_MS}"
echo " Time stride ms : ${TIME_STRIDE_MS}"
echo " Seed           : ${SEED}"
echo " Force          : ${FORCE_RAW}"
echo " Subjects       : ${SUBJ_LIST[*]}"
echo " Bands          : ${BANDS[*]}"
echo " ROIs           : ${ROIS[*]}"
echo " Conditions     : ${CONDITIONS[*]}"
echo " Schemes        : ${SCHEMES[*]}"
echo " Pts/category   : ${POINTS_PER_CATEGORY:-auto (per subject/scheme)}"
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
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --schemes "${SCHEMES[@]}" \
          ${PPC_FLAG[@]+"${PPC_FLAG[@]}"} \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
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
echo "[$(date '+%H:%M:%S')] All per-subject linear-decoding-by-category jobs finished."

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
    --schemes "${SCHEMES[@]}" \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
