#!/usr/bin/env bash
# run_ccgp_epochs.sh
#
# Per-subject CCGP + shattering dimensionality over the four task epochs, then
# cross-subject aggregation (mean +/- SEM ACROSS SUBJECTS).
#
# PARALLELISM -- two levels, and they MULTIPLY, so only one is opened up at a
# time. This runner parallelises over SUBJECTS (one process each, up to
# max_parallel) and therefore passes --n_jobs 1 to the cell script, which would
# otherwise fan out again over its 18 (band, condition, roi) cells and give
# max_parallel * 18 processes. To parallelise over cells instead, run the cell
# script directly for a single subject with --n_jobs 18.
#
# Requires the `decodanda` env -- NOT eegmne (which carries `glue` and hits a
# numpy/sklearn ABI mismatch if decodanda is installed into it):
#   conda activate decodanda && bash run_ccgp_epochs.sh
#
# Usage:
#   bash run_ccgp_epochs.sh [voxRes] [max_parallel] [n_shuffles] [force] [rois...]
#
# Examples:
#   bash run_ccgp_epochs.sh                        # 8mm, all subjects, all 3 ROIs
#   bash run_ccgp_epochs.sh 8mm 8 50 true          # 8 at a time, 50 shuffles, overwrite
#   bash run_ccgp_epochs.sh 8mm 21 25 false visual # visual only

set -euo pipefail

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta)
CONDITIONS=(ampOnly ampPhase)

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"
# CCGP chance is not 0.5, so significance is read against a shuffled null; the
# permutation p floor is 1/(n_shuffles+1), i.e. 0.038 at 25.
N_SHUFFLES="${3:-25}"
FORCE_RAW="${4:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac
shift $(( $# > 4 ? 4 : $# )) || true
ROIS=("$@"); [ ${#ROIS[@]} -eq 0 ] && ROIS=(visual parietal frontal)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/ccgp_epochs_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_ccgp_epochs.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 -c "import decodanda" 2>/dev/null || {
    echo "ERROR: decodanda not importable. Activate its env first:"
    echo "  conda activate decodanda"
    exit 1
}

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/ccgpEpochs"
DATA_DIR="${BASE_DIR}/data"; FIG_DIR="${BASE_DIR}/figures"; CSV_DIR="${BASE_DIR}/tables"
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${CSV_DIR}"

LOG_DIR="logs_ccgp_epochs_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " CCGP + shattering dimensionality runner"
echo " VoxRes       : ${VOX_RES}"
echo " Subjects     : ${#SUBJ_LIST[@]}  (max parallel ${MAX_PARALLEL})"
echo " Shuffles     : ${N_SHUFFLES}  (p floor 1/(n+1))"
echo " Force        : ${FORCE_RAW}"
echo " Bands        : ${BANDS[*]}"
echo " Conditions   : ${CONDITIONS[*]}  (ampPhase needs saved phase)"
echo " ROIs         : ${ROIS[*]}"
echo " Data         : ${DATA_DIR}/"
echo " Figures      : ${FIG_DIR}/"
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
          --n_shuffles "${N_SHUFFLES}" \
          --n_jobs 1 \
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
echo " Aggregating across subjects ..."
echo "========================================================"
python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${BANDS[@]}" \
    --conditions "${CONDITIONS[@]}" \
    --rois "${ROIS[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" \
    2>&1 | tee "${LOG_DIR}/aggregate.log"

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
