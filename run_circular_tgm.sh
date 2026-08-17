#!/usr/bin/env bash
# run_circular_tgm.sh
#
# Circular (sin/cos) Temporal Generalization Matrix decoding via closed-form
# LOO ridge -- see circular_tgm_cell.py's module docstring for the method and
# the Sherman-Morrison cross-time LOO derivation that makes it fast.
#
# Grid: 5 bands (theta/alpha/beta/lowgamma/highgamma) x 2 conditions
#       (ampOnly/ampPhase, the latter only for theta/alpha/beta since the
#       gamma bands have no saved phase) x 3 ROIs = 24 cells per subject.
# Stim-locked only. ERP removed (no ERP-kept arm).
#
# PARALLELISM -- one process per SUBJECT (21), single-threaded BLAS.
# Measured: ~7 s per ampOnly cell and ~13 s per ampPhase cell at realistic
# sizes (N~300 trials, T=54 evaluated timepoints, F=597/1194 features), so
# ~3 min of compute per subject and ~3-5 min wall-clock for the whole grid
# with all subjects running concurrently. There is therefore NO reason to add
# a second parallel level over timepoints: joblib overhead per train-timepoint
# would be a meaningful fraction of the ~0.1 s each one costs, and 21 jobs x
# N threads would oversubscribe vader's cores and slow everything down (the
# same reasoning behind the OMP/MKL/OPENBLAS_NUM_THREADS=1 exports below and
# in every other run_*.sh here). Peak memory is ~200 MB (ampOnly) to ~390 MB
# (ampPhase) per job -- ~8 GB if all 21 hit an ampPhase cell simultaneously --
# because the native-resolution array is freed as soon as the time grid is
# subsampled.
#
# For contrast: svr_tgm.py's RBF-SVR version refits inside the LOO loop
# (n_trials x n_train_t ~ 16k SVR fit-pairs per cell), which is hours per
# subject. Ridge gives identical LOO semantics in closed form -- validated
# against brute-force explicit LOO refitting to float32 precision.
#
# Pure numpy/scipy -- does NOT need the `glue` conda env.
#
# Usage:
#   bash run_circular_tgm.sh [voxRes] [max_parallel] [time_stride_ms] [alpha] [force]
#
# Examples:
#   bash run_circular_tgm.sh                      # 8mm, 21 parallel, 50 ms stride
#   bash run_circular_tgm.sh 8mm 21 50 1.0 true   # overwrite existing .npz
#   bash run_circular_tgm.sh 8mm 21 25            # finer time grid (4x the cells!)

set -euo pipefail

VOX_RES="${1:-8mm}"
TIME_STRIDE_MS="${3:-50}"
ALPHA="${4:-1.0}"
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
# lowgamma/highgamma have no saved phase (constants.AMP_PHASE_BANDS is
# theta/alpha/beta), so circular_tgm_cell.py skips ampPhase for them
# automatically -- they contribute ampOnly cells only, and the plotter drops
# the resulting empty rows from the amp+phase figures.
BANDS=(theta alpha beta lowgamma highgamma)
CONDITIONS=(ampOnly ampPhase)
ROIS=(visual parietal frontal)
WIN_MS=50

MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/circular_tgm_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_circular_tgm.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/circularTGM"
DATA_DIR="${BASE_DIR}/data"
FIG_DIR="${BASE_DIR}/figures"
mkdir -p "${DATA_DIR}" "${FIG_DIR}"

LOG_DIR="logs_circular_tgm_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Circular TGM Runner (LOO ridge, sin/cos)"
echo " VoxRes        : ${VOX_RES}"
echo " Max Parallel  : ${MAX_PARALLEL} (one process per subject)"
echo " Time stride   : ${TIME_STRIDE_MS} ms   (TGM is QUADRATIC in timepoints)"
echo " Ridge alpha   : ${ALPHA}"
echo " Force         : ${FORCE_RAW}"
echo " Subjects      : ${SUBJ_LIST[*]}"
echo " Bands         : ${BANDS[*]}"
echo " Conditions    : ${CONDITIONS[*]}"
echo " ROIs          : ${ROIS[*]}"
echo " Cells/subject : $(( ${#BANDS[@]} * ${#CONDITIONS[@]} * ${#ROIS[@]} ))"
echo " BIDS root     : ${BIDS_ROOT}"
echo " Data          : ${DATA_DIR}/"
echo " Figures       : ${FIG_DIR}/"
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
          --conditions "${CONDITIONS[@]}" \
          --rois "${ROIS[@]}" \
          --voxRes "${VOX_RES}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --alpha "${ALPHA}" \
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
echo "[$(date '+%H:%M:%S')] All per-subject circular-TGM jobs finished."

echo ""
echo "========================================================"
echo " Running aggregator / plotter ..."
echo "========================================================"

python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --rois "${ROIS[@]}" \
    --conditions "${CONDITIONS[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    2>&1 | tee "${LOG_DIR}/plotter.log"

echo ""
echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
