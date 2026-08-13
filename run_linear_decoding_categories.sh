#!/usr/bin/env bash
# run_linear_decoding_categories.sh
#
# Parallelised linear (SVM, one-vs-rest, LOO CV) decoding runner across
# subjects, at several category granularities (2/4/6/10, see
# constants.CATEGORY_SCHEMES). Launches one background job per SUBJECT
# (linear_decoding_categories_cell.py loops bands x rois x conditions x
# schemes internally), with a concurrency limit that defaults to the number
# of subjects.
#
# COST WARNING -- read linear_decoding_categories_cell.py's module docstring
# before raising n_shuffle above 0 with --cv loo: even n_shuffle=5 pushes a
# full subject's 15 band x roi cells to several hours (see the benchmark
# table in chat history / the script's docstring). Default here keeps
# n_shuffle=0 (theoretical chance line 1/n_categories used instead of an
# empirical null) so a full run stays ~1 hour/subject.
#
# Pure numpy/scipy/scikit-learn -- does NOT need the `glue` conda env; run
# with this repo's normal Python environment.
#
# Once all jobs complete, calls plot_linear_decoding_categories.py to
# aggregate results and create figures.
#
# Usage:
#   bash run_linear_decoding_categories.sh [voxRes] [max_parallel] [cv] [n_shuffle] [seed] [force] [points_per_category] [schemes]
#
# Examples:
#   bash run_linear_decoding_categories.sh                                  # 8mm, LOO, n_shuffle=0
#   bash run_linear_decoding_categories.sh 8mm 21 loo 0 0 true               # overwrite existing per-cell .npz
#   bash run_linear_decoding_categories.sh 8mm 21 kfold 20 0 false 10 "2 10" # much cheaper kfold + empirical null, schemes 2/10 only

set -euo pipefail

VOX_RES="${1:-8mm}"
CV="${3:-loo}"
N_SHUFFLE="${4:-0}"
SEED="${5:-0}"
FORCE_RAW="${6:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac
POINTS_PER_CATEGORY="${7:-10}"
SCHEMES_RAW="${8:-2 4 6 10}"
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
echo " Linear Decoding by Category Runner"
echo " VoxRes         : ${VOX_RES}"
echo " Max Parallel   : ${MAX_PARALLEL}"
echo " CV             : ${CV}"
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
echo " Pts/category   : ${POINTS_PER_CATEGORY}"
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
          --points_per_category "${POINTS_PER_CATEGORY}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --cv "${CV}" \
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
