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
# PERFORMANCE BINS (args 5-6, default 3 bins x 0.4): trials are split by
# initial-saccade error and the RDM/geometry computed per bin, with bin 'all'
# always kept as a same-pipeline reference. The split is done WITHIN each target
# location, because saccade error varies with location -- a global split would
# load the "worst" bin with the hard locations and any geometry difference would
# partly be a difference in which locations dominate each bin.
#
# bin_frac (arg 6) sets each bin's width as a fraction of trials. The default
# 0.4 with 3 bins gives bottom 40% / middle 40% / top 40% -- OVERLAPPING windows
# [0,.4] [.3,.7] [.6,1]. That is ~20% more trials per bin than disjoint
# tertiles, which matters because thinness is the binding constraint here, but
# adjacent bins then SHARE 25% of their trials: they are not independent
# samples, and the bin-to-bin contrast is diluted. Bins 0 and n-1 remain fully
# disjoint, so first-vs-last is the clean comparison. Pass "" for tertiles.
#
# NOTE this still thins the data: 138-351 trials/subject over 10 locations
# leaves single-digit counts per location per bin, and crossnobis drops any
# location below MIN_TRIALS_PER_LOC (2). The per-cell log reports the smallest
# per-location count in each bin; if that runs low, 2 bins (arg 5 = 2) roughly
# doubles it.
#
# Usage:
#   bash run_visual_geometry_epochs.sh [voxRes] [max_parallel] [n_null] [force] [n_bins] [bin_frac] [rois...]
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
# COMPUTE every band -- the cell script's output is data, and lowgamma/highgamma
# cost little there. PLOT only theta/alpha/beta: the gamma bands carry no saved
# phase (so they exist for ampOnly only) and are not what these figures are for.
BANDS=(theta alpha beta lowgamma highgamma)
PLOT_BANDS=(theta alpha beta)
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
N_BINS="${5:-3}"      # performance bins from initial-saccade error
# Width of each bin as a fraction of trials. 0.4 with N_BINS=3 = bottom 40% /
# middle 40% / top 40%, i.e. OVERLAPPING windows [0,.4] [.3,.7] [.6,1]: ~20%
# more trials per bin than tertiles, at the cost of adjacent bins sharing 25%
# of their trials (so they are not independent; first-vs-last stays disjoint).
# Set to "" for disjoint tertiles.
BIN_FRAC="${6:-0.4}"
BIN_FRAC_FLAG=()
[ -n "${BIN_FRAC}" ] && BIN_FRAC_FLAG=(--bin_frac "${BIN_FRAC}")
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac
shift $(( $# > 6 ? 6 : $# )) || true
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
echo " Bins         : ${N_BINS} x ${BIN_FRAC:-disjoint}"
echo " N null       : ${N_NULL} label shuffles per epoch"
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
          "${BIN_FRAC_FLAG[@]}" \
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
# This runner is the BINNED pipeline: it just computed N_BINS performance bins,
# so it asks the plotter for them explicitly. The plotter's own default is
# --bins all (unsplit), which is right when it is run standalone but would
# silently drop the bins this runner exists to produce.
BINS_FLAG=(--bins all)
if [ "${N_BINS}" -gt 1 ]; then BINS_FLAG=(--bins auto); fi

python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --bands "${PLOT_BANDS[@]}" \
    --conditions "${CONDITIONS[@]}" \
    --rois "${ROIS[@]}" \
    "${BINS_FLAG[@]}" \
    --outdir "${DATA_DIR}" \
    --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" \
    2>&1 | tee "${LOG_DIR}/plotter.log"

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
