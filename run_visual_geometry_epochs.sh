#!/usr/bin/env bash
# run_visual_geometry_epochs.sh
#
# Epoch-based crossnobis RDM -> MDS geometry (the four-epoch counterpart of
# run_visual_geometry_ts.sh's sliding window).
#
# Epochs (half-open, so adjacent epochs never share a sample):
#   fixation    -1.0 .. 0.0 s
#   stimulus     0.0 .. 0.2 s
#   early_delay  0.2 .. 0.8 s
#   late_delay   1.0 .. 1.6 s
# The 0.8-1.0 s gap is deliberate -- early and late delay are separated rather
# than contiguous. NOTE the epochs are unequal length (1.0/0.2/0.6/0.6 s), so
# the stimulus RDM averages ~5x fewer timepoints and is intrinsically noisier;
# the per-cell label-shuffle null is what keeps that from being read as
# geometry (see visual_geometry_epochs_cell.py's docstring).
#
# Bands x conditions: theta/alpha/beta run ampOnly AND ampPhase;
# lowgamma/highgamma have no saved phase (constants.AMP_PHASE_BANDS) and are
# skipped for ampPhase automatically, contributing amplitude-only cells.
# ROIs: visual, parietal and frontal by default. Pass a different set as
# trailing arguments to narrow it (e.g. "... false visual").
#
# Parallelism: one process per subject, single-threaded BLAS -- same as every
# other run_*.sh here. The label-shuffle null is cheap because the PCA basis is
# computed once per epoch and reused across shuffles.
#
# PERFORMANCE BINS (arg 6, default 3): trials are split into tertiles of
# initial-saccade error and the RDM/geometry computed per bin, with bin 'all'
# always kept as a same-pipeline reference. The split is done WITHIN each target
# location, because saccade error varies with location -- a global split would
# load the "worst" bin with the hard locations and any geometry difference would
# partly be a difference in which locations dominate each bin.
#
# NOTE this thins the data: 138-351 trials/subject over 10 locations and 3 bins
# leaves ~5-12 trials per location per bin, and crossnobis drops any location
# with fewer than 4. The per-cell log reports the smallest per-location count in
# each bin; if that runs low, 2 bins (arg 5 = 2) roughly doubles it.
#
# Usage:
#   bash run_visual_geometry_epochs.sh [voxRes] [max_parallel] [n_null] [force] [n_bins] [rois...]
#
# n_bins comes BEFORE the ROI list: the ROI list is variadic, so nothing fixed
# can follow it.
#
# Examples:
#   bash run_visual_geometry_epochs.sh                       # 8mm, all 3 ROIs, 3 bins
#   bash run_visual_geometry_epochs.sh 8mm 21 100 true 3     # overwrite, 3 bins
#   bash run_visual_geometry_epochs.sh 8mm 21 100 true 2     # 2 bins (more trials each)
#   bash run_visual_geometry_epochs.sh 8mm 21 100 true 3 visual   # visual only

set -euo pipefail

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta lowgamma highgamma)
CONDITIONS=(ampOnly ampPhase)

# Capture EVERY fixed positional BEFORE shifting. Reading e.g. "${2}" after the
# shift silently picks up a trailing ROI instead of max_parallel, which is what
# previously produced "Max Parallel : parietal". Note also that n_bins must come
# BEFORE the variadic ROI list, not after it -- a fixed positional cannot follow
# a variadic one.
VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-${#SUBJ_LIST[@]}}"
N_NULL="${3:-100}"
FORCE_RAW="${4:-false}"
N_BINS="${5:-3}"      # performance tertiles from initial-saccade error
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac
shift $(( $# > 5 ? 5 : $# )) || true
ROIS=("$@"); [ ${#ROIS[@]} -eq 0 ] && ROIS=(visual parietal frontal)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/visual_geometry_epochs_cell.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_visual_geometry_epochs.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/visualGeometryEpochs"
DATA_DIR="${BASE_DIR}/data"; FIG_DIR="${BASE_DIR}/figures"; CSV_DIR="${BASE_DIR}/tables"
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${CSV_DIR}"

LOG_DIR="logs_visual_geometry_epochs_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Epoch-based RDM -> MDS Geometry Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " N null       : ${N_NULL} label shuffles per epoch"
echo " N bins       : ${N_BINS} (performance tertiles of i_sacc_err)"
echo " Force        : ${FORCE_RAW}"
echo " Bands        : ${BANDS[*]}"
echo " Conditions   : ${CONDITIONS[*]}  (ampPhase skipped for the gamma bands)"
echo " ROIs         : ${ROIS[*]}"
echo " Epochs       : fixation stimulus early_delay late_delay"
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
          --conditions "${CONDITIONS[@]}" \
          --rois "${ROIS[@]}" \
          --voxRes "${VOX_RES}" \
          --n_null "${N_NULL}" \
          --n_bins "${N_BINS}" \
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
    --conditions "${CONDITIONS[@]}" \
    --rois "${ROIS[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" \
    2>&1 | tee "${LOG_DIR}/plotter.log"

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
