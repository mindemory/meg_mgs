#!/usr/bin/env bash
# run_two_class_scenario.sh
#
# Focused P=2 (left/right) case study, visual ROI only, theta/alpha/beta
# only, ampOnly -- see chat history for the motivation (collaborator's
# suggestion to focus GLUE on the epoch where the standard linear
# classifier peaks, before committing to an expensive GLUE moving-window
# timecourse). Produces:
#   (a) P=2 ridge-LOO classifier accuracy timecourse, ERP removed vs kept
#       (linear_decoding_categories_cell.py, unchanged, reused as-is)
#   (b) ipsi- vs contra-visual amplitude timecourse, ERP removed vs kept
#       (ipsi_contra_cell.py) -- REQUIRES the visual_left/visual_right ROI
#       caches to exist first:
#           python glue_decoding/precompute_roi_splits.py --rois visual_left visual_right
#       (one-time; see atlas.py's MASK_KEYS / constants.HEMI_ROI_NAMES)
# both plotted side-by-side in one figure by plot_two_class_scenario.py.
#
# NOTE (verified analytically, see plot_two_class_scenario.py's module
# docstring): for (a), remove_erp has ZERO effect on the plotted accuracy --
# the classifier's own per-timepoint z-scoring already removes exactly what
# ERP removal would have. Both variants are still run/plotted as a sanity
# check on that claim, not because a difference is expected there. (b) has
# no such re-centering step, so ERP removal DOES change those curves.
#
# All data and figures are saved under the BIDS derivatives tree (host-aware
# via constants.get_bids_root()) -- NOT under the repo directory -- same
# convention as run_linear_decoding_categories.sh / run_glue_capacity.sh.
# Only logs stay local (logs_two_class_scenario_<voxRes>/), matching every
# other run_*.sh script in this repo.
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
# n_shuffle default is 0 here (unlike run_linear_decoding_categories.sh's
# 100) -- this is a quick look-first-then-decide step, and the cluster-
# permutation significance test in plot_two_class_scenario.py doesn't need
# it (that test uses the real per-subject accuracy values directly, not an
# empirical per-cell shuffle). Pass a larger --n_shuffle if you also want
# the optional single-subject shuffle reference band.

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
CONDITIONS=(ampOnly)
SCHEMES=(2)
WIN_MS=50
TIME_STRIDE_MS=50

MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/linear_decoding_categories_cell.py"
IPSI_CONTRA_SCRIPT="${GLUE_DIR}/ipsi_contra_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_two_class_scenario.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Host-aware BIDS root (zod/vader/other -- see constants.get_bids_root()) --
# everything below is nested under here, in the same derivatives/ tree the
# rest of the pipeline uses, per convention.
BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"

BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/twoClassScenario"
ERP_REMOVED_DIR="${BASE_DIR}/linDecodeCat/erpRemoved"
ERP_KEPT_DIR="${BASE_DIR}/linDecodeCat/erpKept"
IPSI_CONTRA_REMOVED_DIR="${BASE_DIR}/ipsiContra/erpRemoved"
IPSI_CONTRA_KEPT_DIR="${BASE_DIR}/ipsiContra/erpKept"
FIG_DIR="${BASE_DIR}/figures"
mkdir -p "${ERP_REMOVED_DIR}" "${ERP_KEPT_DIR}" "${IPSI_CONTRA_REMOVED_DIR}" "${IPSI_CONTRA_KEPT_DIR}" "${FIG_DIR}"

LOG_DIR="logs_two_class_scenario_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Two-Class (P=2) Scenario Runner"
echo " VoxRes         : ${VOX_RES}"
echo " Max Parallel   : ${MAX_PARALLEL}"
echo " N Shuffle      : ${N_SHUFFLE}"
echo " Seed           : ${SEED}"
echo " Force          : ${FORCE_RAW}"
echo " Subjects       : ${SUBJ_LIST[*]}"
echo " Bands          : ${BANDS[*]}"
echo " ROIs           : ${ROIS[*]}"
echo " Scheme         : ${SCHEMES[*]} (left/right)"
echo " BIDS root      : ${BIDS_ROOT}"
echo " Data (classifier, erpRemoved): ${ERP_REMOVED_DIR}/"
echo " Data (classifier, erpKept)   : ${ERP_KEPT_DIR}/"
echo " Data (ipsi/contra, erpRemoved): ${IPSI_CONTRA_REMOVED_DIR}/"
echo " Data (ipsi/contra, erpKept)   : ${IPSI_CONTRA_KEPT_DIR}/"
echo " Figures        : ${FIG_DIR}/"
echo " Jobs           : $(( ${#SUBJ_LIST[@]} * 4 )) (4 per subject: classifier x2 ERP states + ipsi/contra x2 ERP states)"
echo " Logging to     : ${LOG_DIR}/"
echo "========================================================"
echo ""
echo "NOTE: ipsi/contra jobs need the visual_left/visual_right ROI caches --"
echo "if they haven't been built yet, run this first (one-time):"
echo "    python3 ${GLUE_DIR}/precompute_roi_splits.py --rois visual_left visual_right"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))

    ( python3 "${CELL_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --schemes "${SCHEMES[@]}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --seed "${SEED}" \
          --outdir "${ERP_REMOVED_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_classifier_erpRemoved.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${CELL_SCRIPT}" \
          "${subj_num}" \
          --voxRes "${VOX_RES}" \
          --bands "${BANDS[@]}" \
          --rois "${ROIS[@]}" \
          --conditions "${CONDITIONS[@]}" \
          --schemes "${SCHEMES[@]}" \
          --win_ms "${WIN_MS}" \
          --time_stride_ms "${TIME_STRIDE_MS}" \
          --n_shuffle "${N_SHUFFLE}" \
          --seed "${SEED}" \
          --outdir "${ERP_KEPT_DIR}" \
          --no_erp_removal \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_classifier_erpKept.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${IPSI_CONTRA_SCRIPT}" \
          "${subj_num}" \
          --bands "${BANDS[@]}" \
          --voxRes "${VOX_RES}" \
          --outdir "${IPSI_CONTRA_REMOVED_DIR}" \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_ipsiContra_erpRemoved.log" 2>&1 &
    count=$((count + 1))

    ( python3 "${IPSI_CONTRA_SCRIPT}" \
          "${subj_num}" \
          --bands "${BANDS[@]}" \
          --voxRes "${VOX_RES}" \
          --outdir "${IPSI_CONTRA_KEPT_DIR}" \
          --no_erp_removal \
          ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
    ) > "${LOG_DIR}/sub-${subjID}_ipsiContra_erpKept.log" 2>&1 &
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
    --erp_removed_dir "${ERP_REMOVED_DIR}" \
    --erp_kept_dir "${ERP_KEPT_DIR}" \
    --ipsi_contra_removed_dir "${IPSI_CONTRA_REMOVED_DIR}" \
    --ipsi_contra_kept_dir "${IPSI_CONTRA_KEPT_DIR}" \
    --figdir "${FIG_DIR}" \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
