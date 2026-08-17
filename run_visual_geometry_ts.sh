#!/usr/bin/env bash
# run_visual_geometry_ts.sh
#
# TIME-RESOLVED RDM -> MDS geometry (no temporal generalization: each
# timepoint is independent, like the decoding timecourses).
#
# Grid: 3 bands (theta/alpha/beta) x 2 feature reps (ampOnly / ampPhase)
#       x ROIs (visual by default) . Stim-locked, ERP removed.
# Windows: +/- 50 ms => 100 ms per RDM, evaluated every 50 ms.
#
# Tracks over time: ring-ness, lambda2/lambda1, top-2 variance fraction,
# PR of the MDS spectrum, radial CV, negative-eigenvalue fraction, and the
# inter-subject consistency gate -- each against a per-timepoint LABEL-SHUFFLE
# null, since none of these metrics has a chance level of zero.
#
# READ THIS BEFORE INTERPRETING lambda2/lambda1: the usual "a ring gives ~1"
# rule assumes uniformly spaced angles. This study's 10 locations are NOT
# uniform (gaps 25,25,80,25,25,25,25,80,25,25 deg), and a PERFECT ring at
# exactly these angles gives lambda2/lambda1 = 0.441 while random geometry
# averages 0.641 -- so "closer to 1" would rank noise ABOVE a true ring. The
# figures draw the correct reference line; see plot_visual_geometry_ts.py.
#
# Parallelism: one process per subject, single-threaded BLAS (same rationale
# as every other run_*.sh here). The label-shuffle null is cheap because the
# PCA basis is computed once per timepoint and reused across shuffles.
#
# Usage:
#   bash run_visual_geometry_ts.sh [voxRes] [max_parallel] [n_null] [force] [rois...]
#
# Examples:
#   bash run_visual_geometry_ts.sh                          # 8mm, visual, 50 shuffles
#   bash run_visual_geometry_ts.sh 8mm 21 50 true           # overwrite
#   bash run_visual_geometry_ts.sh 8mm 21 50 false visual parietal frontal

set -euo pipefail

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta)
FEATURE_REPS=(ampOnly ampPhase)
WIN_MS=50            # one-sided => 100 ms window
TIME_STRIDE_MS=50

# All fixed positionals are read BEFORE the shift: reading "${2}" afterwards
# would pick up the first trailing ROI instead of max_parallel. (This only
# stayed hidden here because the ROI list was never actually passed.)
VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"
N_NULL="${3:-50}"
FORCE_RAW="${4:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac
shift $(( $# > 4 ? 4 : $# )) || true
ROIS=("$@"); [ ${#ROIS[@]} -eq 0 ] && ROIS=(visual)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/visual_geometry_ts_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_visual_geometry_ts.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/visualGeometryTS"
DATA_DIR="${BASE_DIR}/data"; FIG_DIR="${BASE_DIR}/figures"; CSV_DIR="${BASE_DIR}/tables"
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${CSV_DIR}"

LOG_DIR="logs_visual_geometry_ts_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Time-resolved RDM -> MDS Geometry Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Window       : +-${WIN_MS} ms (=> $((2*WIN_MS)) ms), stride ${TIME_STRIDE_MS} ms"
echo " N null       : ${N_NULL} label shuffles per timepoint"
echo " Force        : ${FORCE_RAW}"
echo " Bands        : ${BANDS[*]}"
echo " Feature reps : ${FEATURE_REPS[*]}"
echo " ROIs         : ${ROIS[*]}"
echo " Data         : ${DATA_DIR}/"
echo " Figures      : ${FIG_DIR}/"
echo " Tables       : ${CSV_DIR}/"
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
          --rois "${ROIS[@]}" \
          --voxRes "${VOX_RES}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_null "${N_NULL}" \
          --outdir "${DATA_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}.log" 2>&1 &
    count=$((count + 1))
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit. Waiting..."; wait
    fi
done
wait
echo ""
echo "[$(date '+%H:%M:%S')] All per-subject jobs finished."

echo ""
echo "========================================================"
echo " Running aggregator / plotter ..."
echo "========================================================"
python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --feature_reps "${FEATURE_REPS[@]}" \
    --rois "${ROIS[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" \
    2>&1 | tee "${LOG_DIR}/plotter.log"

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
