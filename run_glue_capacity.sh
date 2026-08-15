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
#   bash run_glue_capacity.sh [voxRes] [max_parallel] [n_hyperplanes] [seed] [force] [epochs] [schemes] [points_per_category]
#
# Examples:
#   bash run_glue_capacity.sh                                  # 8mm, max parallel = n_subjects, n_hyperplanes=200, stim+delay, schemes 2 4 6 10
#   bash run_glue_capacity.sh 8mm 21 200 42 true                # overwrite existing per-subject CSVs, stim+delay, all schemes
#   bash run_glue_capacity.sh 8mm 21 200 42 true stim            # force, stim epoch only
#   bash run_glue_capacity.sh 8mm 21 200 42 true stim "2"         # force, stim only, P=2 (left/right) only
#
# schemes: 2=left/right hemifield, 4=quadrants, 6=quadrants+axis, 10=every
# raw location (the only option before) -- see constants.CATEGORY_SCHEMES.
# points_per_category: leave empty/unset (default) for auto -- each
# (subject, scheme) balanced to that subject's own smallest category trial
# count for that scheme (see manifold_capacity.py's module docstring).
#
# manifold_capacity.py builds ONE (time-averaged) point per trial -- an
# intermediate version treated every timepoint as its own point instead,
# which was far more faithful to per-timepoint dynamics but made cvxopt's
# QP solves intractably slow (a real 4-scheme x 5-band x 3-roi x 21-subject
# stim-only run was still running after 5+ hours with zero cells finished);
# reverted for tractability. See manifold_capacity.py's module docstring
# for the dimensionality-collapse caveat this reintroduces.

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
# arg 6: space-separated epoch list, e.g. "stim" to skip delay's much larger
# per-timepoint point count (see manifold_capacity.py's module docstring).
EPOCHS_RAW="${6:-stim delay}"
EPOCHS=(${EPOCHS_RAW})
SCHEMES_RAW="${7:-2 4 6 10}"
SCHEMES=(${SCHEMES_RAW})
POINTS_PER_CATEGORY="${8:-}"
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
          --epochs "${EPOCHS[@]}" \
          --schemes "${SCHEMES[@]}" \
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
    --epochs "${EPOCHS[@]}" \
    --schemes "${SCHEMES[@]}" \
    > "${LOG_DIR}/aggregator.log" 2>&1

echo "[$(date '+%H:%M:%S')] Aggregator finished. Log: ${LOG_DIR}/aggregator.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
