#!/usr/bin/env bash
# run_glue_capacity.sh
#
# Parallelised glue manifold-capacity runner across subjects. Launches one
# background job per SUBJECT (manifold_capacity.py loops bands x rois x
# conditions x schemes x time-windows internally for that one subject), with
# a concurrency limit that defaults to the number of subjects -- i.e. by
# default every subject runs concurrently.
#
# Requires the `glue` package (github.com/cnchou/glue) importable -- run
# this from whichever conda env has it (e.g. `conda activate eegmne` on
# vader) BEFORE launching this script, since it just calls `python3`.
#
# Once all jobs complete, calls aggregate_glue_capacity.py to build the
# cross-subject TIME-COURSE figures.
#
# CURRENT DESIGN (see manifold_capacity.py's module docstring for the why):
#   * sliding 100 ms window, non-overlapping (stride 100 ms), over
#     [-0.5, 1.7] s -- NOT the old two fixed epochs (stim / delay)
#   * ERP removed (grand trial-average subtracted per cell before windowing)
#   * bands theta / alpha / beta (the bands with saved phase)
#   * conditions ampOnly (all ROIs) + ampPhase (visual only)
#   * scheme 4 = quadrant manifolds, which excludes the two axis-aligned
#     locations (0 deg and 180 deg)
#
# Usage:
#   bash run_glue_capacity.sh [voxRes] [max_parallel] [n_hyperplanes] [seed] [force] \
#                             [win_ms] [stride_ms] [schemes] [points_per_category]
#
# Examples:
#   bash run_glue_capacity.sh                                   # defaults above
#   bash run_glue_capacity.sh 8mm 21 200 42 true                # overwrite existing per-subject CSVs
#   bash run_glue_capacity.sh 8mm 21 200 42 true 100 50         # 50 ms stride (overlapping; 2x the fits)
#   bash run_glue_capacity.sh 8mm 21 200 42 true 100 100 "4 10" # also run P=10 raw-location manifolds
#
# schemes: 2=left/right hemifield, 4=quadrants (default), 6=quadrants+axis,
# 10=every raw location -- see constants.CATEGORY_SCHEMES.
# points_per_category: leave empty/unset (default) for auto -- each
# (subject, scheme) balanced to that subject's own smallest category trial
# count for that scheme (see manifold_capacity.py's module docstring).
#
# COST: manifold_capacity.py builds ONE (window-averaged) point per trial per
# window -- an intermediate version treated every timepoint as its own point
# instead, which was far more faithful to per-timepoint dynamics but made
# cvxopt's QP solves intractably slow (a real 4-scheme x 5-band x 3-roi x
# 21-subject stim-only run was still running after 5+ hours with zero cells
# finished); reverted for tractability, and doubly necessary now that each
# cell is fit in ~22 windows rather than 2 epochs. See manifold_capacity.py's
# module docstring for the dimensionality-collapse caveat this reintroduces.
# Run ONE subject first and check its wall time before launching the fleet.

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
BANDS=(theta alpha beta)
# VISUAL ONLY, per the request for this analysis. Each extra ROI adds
# n_windows x n_bands more glue QP fits per subject (~66 at the 100 ms /
# 22-window default), and glue fits are the expensive part of this pipeline --
# an earlier full sweep here ran 5+ hours before being killed. Add
# `parietal frontal` back here only if you actually want them.
ROIS=(visual)
CONDITIONS=(ampOnly ampPhase)
# ampPhase runs for these ROIs only; ampOnly runs for every ROI in ROIS.
PHASE_ROIS=(visual)

WIN_MS="${6:-100}"
STRIDE_MS="${7:-100}"
TMIN="-0.5"
TMAX="1.7"

SCHEMES_RAW="${8:-4}"
SCHEMES=(${SCHEMES_RAW})
POINTS_PER_CATEGORY="${9:-}"
PPC_FLAG=()
if [ -n "${POINTS_PER_CATEGORY}" ]; then
    PPC_FLAG=(--points_per_category "${POINTS_PER_CATEGORY}")
fi

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
echo " Glue Manifold-Capacity Runner (sliding window, ERP removed)"
echo " VoxRes         : ${VOX_RES}"
echo " Max Parallel   : ${MAX_PARALLEL}"
echo " N Hyperplanes  : ${N_HYPERPLANES}"
echo " Seed           : ${SEED}"
echo " Force          : ${FORCE_RAW}"
echo " Subjects       : ${SUBJ_LIST[*]}"
echo " Bands          : ${BANDS[*]}"
echo " ROIs           : ${ROIS[*]}"
echo " Conditions     : ${CONDITIONS[*]} (phase conditions: ${PHASE_ROIS[*]} only)"
echo " Window         : ${WIN_MS} ms, stride ${STRIDE_MS} ms, span [${TMIN}, ${TMAX}] s"
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
          --subjID "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --phase_rois "${PHASE_ROIS[@]}" \
          --schemes "${SCHEMES[@]}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${STRIDE_MS}" \
          --tmin "${TMIN}" \
          --tmax "${TMAX}" \
          ${PPC_FLAG[@]+"${PPC_FLAG[@]}"} \
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
    --conditions "${CONDITIONS[@]}" \
    --schemes "${SCHEMES[@]}" \
    > "${LOG_DIR}/aggregator.log" 2>&1

echo "[$(date '+%H:%M:%S')] Aggregator finished. Log: ${LOG_DIR}/aggregator.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
