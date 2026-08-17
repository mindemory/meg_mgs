#!/usr/bin/env bash
# run_connectivity_imcoh.sh
#
# ROI-pair imaginary coherence over the four task epochs
# (visual-parietal, visual-frontal, parietal-frontal) x theta/alpha/beta.
#
# ImCoh rather than plain coherence because it ignores zero-lag coupling, which
# is what spatial leakage produces; only genuinely lagged interaction counts.
# Bands are limited to theta/alpha/beta because ImCoh needs phase and the gamma
# bands have none saved.
#
# TWO METHODOLOGICAL POINTS, both verified numerically -- see
# connectivity_imcoh_epochs.py's docstring:
#   * The measure is mean|ImCoh| across source pairs, NOT the signed mean.
#     Im(coherency) is signed by lag direction, so averaging signed values over
#     pairs with mixed directions cancels them: simulated true coupling of 0.65
#     averaged to 0.0006 signed but 0.648 in absolute value.
#   * mean|ImCoh| is noise-biased as ~1/sqrt(n_samples), so the four epochs --
#     1.0 / 0.2 / 0.6 / 0.6 s -- would have noise floors differing by 2.2x and
#     the stimulus epoch would appear elevated for no reason. Sample counts are
#     therefore EQUALISED across epochs, and a trial-shuffle surrogate floor is
#     computed per cell and reported alongside. Read the gap, not the height.
#
# Parallelism: one process per subject, single-threaded BLAS. Memory is the
# binding constraint here, not CPU: a subject's amplitude+phase for one ROI is
# ~1.2 GB transiently while the analytic signal is built, so the default
# parallelism is deliberately BELOW the subject count.
#
# Usage:
#   bash run_connectivity_imcoh.sh [voxRes] [max_parallel] [n_null] [force]

set -euo pipefail

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
BANDS=(theta alpha beta)
PAIRS=(visual-parietal visual-frontal parietal-frontal)

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-8}"
N_NULL="${3:-20}"
FORCE_RAW="${4:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/connectivity_imcoh_epochs.py"
PLOT_SCRIPT="${GLUE_DIR}/plot_connectivity_imcoh.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/imcohEpochs"
DATA_DIR="${BASE_DIR}/data"; FIG_DIR="${BASE_DIR}/figures"; CSV_DIR="${BASE_DIR}/tables"
mkdir -p "${DATA_DIR}" "${FIG_DIR}" "${CSV_DIR}"
LOG_DIR="logs_imcoh_${VOX_RES}"; mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " ROI-pair Imaginary Coherence Runner"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}  (memory-bound: ~1.2 GB/subject transient)"
echo " N null       : ${N_NULL} trial-shuffle surrogates per epoch"
echo " Force        : ${FORCE_RAW}"
echo " Bands        : ${BANDS[*]}"
echo " Pairs        : ${PAIRS[*]}"
echo " Epochs       : fixation stimulus early_delay late_delay (samples equalised)"
echo " Data         : ${DATA_DIR}/"
echo "========================================================"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ..."
    ( python3 "${CELL_SCRIPT}" "${subj_num}" \
          --bands "${BANDS[@]}" --pairs "${PAIRS[@]}" \
          --voxRes "${VOX_RES}" --n_null "${N_NULL}" \
          --outdir "${DATA_DIR}" ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
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
python3 "${PLOT_SCRIPT}" --voxRes "${VOX_RES}" --bands "${BANDS[@]}" \
    --pairs "${PAIRS[@]}" --outdir "${DATA_DIR}" --figdir "${FIG_DIR}" \
    --csvdir "${CSV_DIR}" 2>&1 | tee "${LOG_DIR}/plotter.log"

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
