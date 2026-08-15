#!/usr/bin/env bash
# run_two_class_scenario.sh
#
# Focused P=2 (left/right) case study, visual ROI only, theta/alpha/beta
# only, ampOnly -- see chat history for the motivation (collaborator's
# suggestion to focus GLUE on the epoch where the standard linear
# classifier peaks, before committing to an expensive GLUE moving-window
# timecourse). This is step 1 of that plan: get the P=2 ridge-LOO
# classifier timecourse, WITH and WITHOUT ERP removal, so we can see it
# before deciding how to scope the (not yet built) GLUE step.
#
# Reuses linear_decoding_categories_cell.py's run_cell UNCHANGED (already
# validated) -- just run twice per subject, once with ERP removal (default)
# and once without (--no_erp_removal), each into its own --outdir so the
# two variants don't collide (output_path doesn't encode remove_erp in the
# filename). Data lands in data_two_class_scenario_<voxRes>/{erpRemoved,
# erpKept}/, NOT under the BIDS derivatives tree -- this is a scratch/focused
# analysis, deliberately kept separate from the main linDecodeCat pipeline's
# output.
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
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/linear_decoding_categories_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/plot_two_class_scenario.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATA_DIR="${SCRIPT_DIR}/data_two_class_scenario_${VOX_RES}"
ERP_REMOVED_DIR="${DATA_DIR}/erpRemoved"
ERP_KEPT_DIR="${DATA_DIR}/erpKept"
mkdir -p "${ERP_REMOVED_DIR}" "${ERP_KEPT_DIR}"

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
echo " Data (erpRemoved): ${ERP_REMOVED_DIR}/"
echo " Data (erpKept)   : ${ERP_KEPT_DIR}/"
echo " Jobs           : $(( ${#SUBJ_LIST[@]} * 2 )) (2 per subject: erpRemoved + erpKept)"
echo " Logging to     : ${LOG_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))

    log_file_removed="${LOG_DIR}/sub-${subjID}_erpRemoved.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (ERP removed) ..."
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
    ) > "${log_file_removed}" 2>&1 &
    count=$((count + 1))

    log_file_kept="${LOG_DIR}/sub-${subjID}_erpKept.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (ERP kept) ..."
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
    ) > "${log_file_kept}" 2>&1 &
    count=$((count + 1))

    if [ $((count % (MAX_PARALLEL * 2))) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL} subjects x2). Waiting for batch to complete..."
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
    --figdir "${DATA_DIR}" \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plotter finished. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
