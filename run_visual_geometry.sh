#!/usr/bin/env bash
# run_visual_geometry.sh
#
# Pre-capacity geometry of the spatial-WM location code in the VISUAL ROI:
# an interpretable signal / noise / alignment decomposition computed BEFORE
# committing to manifold capacity. See visual_geometry_cell.py and
# aggregate_visual_geometry.py module docstrings for the full method.
#
# Grid: 3 bands (theta/alpha/beta) x 2 feature reps (ampOnly / ampPhase)
#       x 3 epochs (fixation / stim / delay) = 18 cells per subject.
# ROI  : visual only. Stim-locked only. ERP ALWAYS removed (no ERP-kept arm).
#
# Cross-subject handling (Option A): geometry is computed entirely within
# each subject's own source space; only rotation-invariant scalars and
# cross-subject-safe RDM summaries are aggregated. Raw trials are NEVER
# pooled across subjects -- the 597-dim source spaces share a dimensionality
# but not an identity, so pooling them would be meaningless.
#
# Requires the visual ROI caches to exist (built by precompute_roi_splits.py),
# same as every other roi= consumer in this repo.
#
# Pure numpy/scipy/matplotlib -- does NOT need the `glue` conda env.
#
# Usage:
#   bash run_visual_geometry.sh [voxRes] [max_parallel] [seed] [force]
#
# Examples:
#   bash run_visual_geometry.sh                    # 8mm, all subjects in parallel
#   bash run_visual_geometry.sh 8mm 21 0 true      # overwrite existing per-cell .npz

set -euo pipefail

VOX_RES="${1:-8mm}"
SEED="${3:-0}"
FORCE_RAW="${4:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta)
FEATURE_REPS=(ampOnly ampPhase)
ROI=visual

MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/visual_geometry_cell.py"
AGG_SCRIPT="${GLUE_DIR}/aggregate_visual_geometry.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"

BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/visualGeometry"
DATA_DIR="${BASE_DIR}/data"
FIG_DIR="${BASE_DIR}/figures"
CSV_DIR="${BASE_DIR}/tables"
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${CSV_DIR}"

LOG_DIR="logs_visual_geometry_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Visual Geometry (pre-capacity) Runner"
echo " VoxRes        : ${VOX_RES}"
echo " Max Parallel  : ${MAX_PARALLEL}"
echo " Seed          : ${SEED}"
echo " Force         : ${FORCE_RAW}"
echo " Subjects      : ${SUBJ_LIST[*]}"
echo " Bands         : ${BANDS[*]}"
echo " Feature reps  : ${FEATURE_REPS[*]}"
echo " ROI           : ${ROI}"
echo " Epochs        : fixation stim delay (ERP always removed)"
echo " BIDS root     : ${BIDS_ROOT}"
echo " Data          : ${DATA_DIR}/"
echo " Figures       : ${FIG_DIR}/"
echo " Tables        : ${CSV_DIR}/"
echo " Logging to    : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ..."

    ( python3 "${CELL_SCRIPT}" \
          "${subj_num}" \
          --bands "${BANDS[@]}" \
          --feature_reps "${FEATURE_REPS[@]}" \
          --voxRes "${VOX_RES}" \
          --roi "${ROI}" \
          --seed "${SEED}" \
          --outdir "${DATA_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}.log" 2>&1 &

    count=$((count + 1))
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL}). Waiting..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
    fi
done

wait
echo ""
echo "[$(date '+%H:%M:%S')] All per-subject visual-geometry jobs finished."

# ── Aggregation / figures ────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Running aggregator ..."
echo "========================================================"

python3 "${AGG_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --roi "${ROI}" \
    --bands "${BANDS[@]}" \
    --feature_reps "${FEATURE_REPS[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" \
    2>&1 | tee "${LOG_DIR}/aggregator.log"

echo ""
echo "[$(date '+%H:%M:%S')] Aggregator finished. Log: ${LOG_DIR}/aggregator.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
