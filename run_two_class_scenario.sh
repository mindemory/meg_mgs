#!/usr/bin/env bash
# run_two_class_scenario.sh
#
# Focused visual-ROI case study, theta/alpha/beta only, across THREE
# decoders and THREE feature conditions, each with/without ERP removal:
#   decoders:
#     - LOO ridge binary classification, left vs right   (scheme=2)
#     - LOO ridge binary classification, top vs bottom    (scheme=3)
#     - LOO ridge circular regression, all 10 locations   (sin/cos targets)
#   conditions: ampOnly, ampPhase, phaseOnly
#   ERP: removed (default), kept (--no_erp_removal)
#
# ROI is 'visual' only (whole ROI, NOT split by hemisphere -- the earlier
# ipsi/contra-visual amplitude comparison has been dropped from this script;
# ipsi_contra_cell.py still exists standalone if needed again).
#
# Reuses linear_decoding_categories_cell.py (classifiers) and
# decoding_ts_cell.py (circular regression) UNCHANGED -- both already
# validated -- just run twice per subject each (ERP removed / kept), each
# internally looping bands x conditions x schemes.
#
# NOTE (see plot_two_class_scenario.py's module docstring for the full
# derivation): BOTH decoders z-score every feature across trials at each
# independently-evaluated timepoint before fitting (see
# linear_decoding_categories_cell.py's ridge_ovr_timeseries and
# decoding_ts_cell.py's ridge_loocv_timeseries). That per-timepoint
# z-scoring subtracts out any trial-invariant constant -- exactly what ERP
# removal subtracts -- so for BOTH decoders here, remove_erp has ZERO effect
# on the result, provably (verified earlier this session on real saved
# accuracy arrays via np.allclose). This is expected, not a bug -- both ERP
# states are still run/plotted purely as a live sanity check on that claim.
#
# All data and figures are saved under the BIDS derivatives tree (host-aware
# via constants.get_bids_root()), not the repo directory. Only logs stay
# local (logs_two_class_scenario_<voxRes>/).
#
# Pure numpy -- does NOT need the `glue` conda env; run with this repo's
# normal Python environment.
#
# Usage:
#   bash run_two_class_scenario.sh [voxRes] [max_parallel] [n_shuffle] [seed] [force]
#
# Examples:
#   bash run_two_class_scenario.sh                       # 8mm, n_shuffle=0 (fast; see below)
#   bash run_two_class_scenario.sh 8mm 21 0 0 true        # overwrite existing per-cell .npz
#
# n_shuffle default is 0 -- this is a quick look-first-then-decide step, and
# the cluster-permutation significance test in plot_two_class_scenario.py
# uses the real per-subject values directly, not an empirical per-cell
# shuffle. Pass a larger --n_shuffle if you also want the optional
# single-subject shuffle reference band.

set -euo pipefail

VOX_RES="${1:-8mm}"
N_SHUFFLE="${3:-0}"
SEED="${4:-0}"
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss])
        FORCE_FLAG=(--force)
        ;;
esac

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta)
ROIS=(visual)
CONDITIONS=(ampOnly ampPhase phaseOnly)
CLASSIFIER_SCHEMES=(2 3)   # 2=left/right, 3=top/bottom -- see constants.CATEGORY_SCHEMES
WIN_MS=50
TIME_STRIDE_MS=50

MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CLASSIFIER_SCRIPT="${GLUE_DIR}/linear_decoding_categories_cell.py"
CIRCULAR_SCRIPT="${GLUE_DIR}/decoding_ts_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_two_class_scenario.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Host-aware BIDS root (zod/vader/other -- see constants.get_bids_root()) --
# everything below is nested under here, in the same derivatives/ tree the
# rest of the pipeline uses, per convention.
BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"

BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/twoClassScenario"
CLASSIFIER_REMOVED_DIR="${BASE_DIR}/linDecodeCat/erpRemoved"
CLASSIFIER_KEPT_DIR="${BASE_DIR}/linDecodeCat/erpKept"
CIRCULAR_REMOVED_DIR="${BASE_DIR}/decodingTS/erpRemoved"
CIRCULAR_KEPT_DIR="${BASE_DIR}/decodingTS/erpKept"
FIG_DIR="${BASE_DIR}/figures"
mkdir -p "${CLASSIFIER_REMOVED_DIR}" "${CLASSIFIER_KEPT_DIR}" \
         "${CIRCULAR_REMOVED_DIR}" "${CIRCULAR_KEPT_DIR}" "${FIG_DIR}"

LOG_DIR="logs_two_class_scenario_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Two-Class Scenario Runner (visual ROI, theta/alpha/beta)"
echo " VoxRes           : ${VOX_RES}"
echo " Max Parallel      : ${MAX_PARALLEL}"
echo " N Shuffle         : ${N_SHUFFLE}"
echo " Seed              : ${SEED}"
echo " Force             : ${FORCE_RAW}"
echo " Subjects          : ${SUBJ_LIST[*]}"
echo " Bands             : ${BANDS[*]}"
echo " Conditions        : ${CONDITIONS[*]}"
echo " Classifier schemes: ${CLASSIFIER_SCHEMES[*]} (2=left/right, 3=top/bottom)"
echo " BIDS root         : ${BIDS_ROOT}"
echo " Data (classifier, erpRemoved): ${CLASSIFIER_REMOVED_DIR}/"
echo " Data (classifier, erpKept)   : ${CLASSIFIER_KEPT_DIR}/"
echo " Data (circular, erpRemoved)  : ${CIRCULAR_REMOVED_DIR}/"
echo " Data (circular, erpKept)     : ${CIRCULAR_KEPT_DIR}/"
echo " Figures           : ${FIG_DIR}/"
echo " Jobs              : $(( ${#SUBJ_LIST[@]} * 4 )) (4 per subject: classifier x2 ERP + circular x2 ERP)"
echo " Logging to        : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))

    ( python3 "${CLASSIFIER_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --schemes "${CLASSIFIER_SCHEMES[@]}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --seed "${SEED}" \
          --outdir "${CLASSIFIER_REMOVED_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_classifier_erpRemoved.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${CLASSIFIER_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --schemes "${CLASSIFIER_SCHEMES[@]}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --seed "${SEED}" \
          --outdir "${CLASSIFIER_KEPT_DIR}" \
          --no_erp_removal \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_classifier_erpKept.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${CIRCULAR_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --win_ms "${WIN_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --outdir "${CIRCULAR_REMOVED_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_circular_erpRemoved.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${CIRCULAR_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --win_ms "${WIN_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --outdir "${CIRCULAR_KEPT_DIR}" \
          --no_erp_removal \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_circular_erpKept.log" 2>&1 &
    count=$((count + 1))

    if [ $((count % (MAX_PARALLEL * 4))) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL} subjects x4). Waiting for batch to complete..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
    fi
done

wait
echo ""
echo "[$(date '+%H:%M:%S')] All per-subject two-class-scenario jobs finished."

# ── Plotting ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Running plotter ..."
echo "========================================================"

python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --roi "${ROIS[0]}" \
    --conditions "${CONDITIONS[@]}" \
    --classifier_removed_dir "${CLASSIFIER_REMOVED_DIR}" \
    --classifier_kept_dir "${CLASSIFIER_KEPT_DIR}" \
    --circular_removed_dir "${CIRCULAR_REMOVED_DIR}" \
    --circular_kept_dir "${CIRCULAR_KEPT_DIR}" \
    --figdir "${FIG_DIR}" \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
