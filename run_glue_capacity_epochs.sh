#!/usr/bin/env bash
# run_glue_capacity_epochs.sh
#
# glue manifold capacity on the four task epochs, with manifolds POOLED ACROSS
# SUBJECTS (all trials for a location, from every subject, in one manifold).
# See manifold_capacity_epochs.py's module docstring for the method and, in
# particular, for how cross-subject source identity is handled.
#
# Epochs match visual_geometry_epochs_cell.py exactly (fixation / stimulus /
# early_delay / late_delay), so the two analyses line up cell for cell.
# Scheme 4 = quadrant manifolds (drops 0 and 180 deg). ROIs: visual, parietal,
# frontal. Conditions: ampOnly everywhere, ampPhase for --phase_rois (default
# visual) since it doubles the feature count and the QP cost.
#
# PARALLELISM IS DIFFERENT HERE. Every other runner in this repo fans out one
# job per SUBJECT; that is impossible when the analysis pools subjects, since
# each fit needs all of them at once. Parallelism is therefore over (band, roi)
# CELLS -- 3 bands x 3 rois = 9 jobs, each handling its conditions x 4 epochs.
# Each job independently loads all subjects for its own cell and writes its own
# CSV; the CSVs are concatenated at the end.
#
# COST WARNING: pooling multiplies points-per-manifold by ~n_subjects, and
# glue's QP time scales badly with point count (an earlier sweep here ran 5+
# hours before being killed for that reason). Points per manifold are capped by
# --points_per_category (default 200). TIME ONE CELL BEFORE THE FULL GRID:
#
#   conda activate eegmne
#   python3 glue_decoding/manifold_capacity_epochs.py \
#       --bands alpha --rois visual --conditions ampOnly --force
#
# Requires the `glue` conda env -- this script only calls python3, it does NOT
# activate anything. Activate it yourself first.
#
# Usage:
#   bash run_glue_capacity_epochs.sh [voxRes] [max_parallel] [points_per_category] [n_hyperplanes] [force]

set -euo pipefail

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-9}"
PPC="${3:-200}"
N_HYP="${4:-200}"
FORCE_RAW="${5:-false}"
FORCE_FLAG=()
case "${FORCE_RAW}" in
    [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) FORCE_FLAG=(--force) ;;
esac

BANDS=(theta alpha beta)
ROIS=(visual parietal frontal)
CONDITIONS=(ampOnly ampPhase)
PHASE_ROIS=(visual)
SCHEMES=(4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLUE_DIR="${SCRIPT_DIR}/glue_decoding"
CELL_SCRIPT="${GLUE_DIR}/manifold_capacity_epochs.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if ! python3 -c "from glue.contrib import glue_analysis_dataframe" 2>/dev/null; then
    echo "ERROR: the \`glue\` package is not importable in this environment."
    echo "       Activate the env that has it first, e.g.:  conda activate eegmne"
    echo "       (check with: python3 -c 'from glue.contrib import glue_analysis_dataframe')"
    exit 1
fi

BIDS_ROOT="$(cd "${GLUE_DIR}" && python3 -c 'from constants import get_bids_root; print(get_bids_root())')"
BASE_DIR="${BIDS_ROOT}/derivatives/glueDecoding/glueEpochsPooled"
PART_DIR="${BASE_DIR}/parts"
mkdir -p "${BASE_DIR}" "${PART_DIR}"

LOG_DIR="logs_glue_epochs_${VOX_RES}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo " Pooled Epoch glue Capacity Runner"
echo " VoxRes           : ${VOX_RES}"
echo " Max Parallel     : ${MAX_PARALLEL}  (over (band,roi) cells -- NOT subjects)"
echo " Points/manifold  : ${PPC} (cap, applied after pooling)"
echo " N hyperplanes    : ${N_HYP}"
echo " Force            : ${FORCE_RAW}"
echo " Bands            : ${BANDS[*]}"
echo " ROIs             : ${ROIS[*]}"
echo " Conditions       : ${CONDITIONS[*]}  (ampPhase only for: ${PHASE_ROIS[*]})"
echo " Schemes          : ${SCHEMES[*]}  (4 = quadrants, drops 0/180 deg)"
echo " Epochs           : fixation stimulus early_delay late_delay"
echo " Output           : ${BASE_DIR}/"
echo "========================================================"
echo ""

count=0
for band in "${BANDS[@]}"; do
    for roi in "${ROIS[@]}"; do
        tag="${band}_${roi}"
        echo "[$(date '+%H:%M:%S')] Starting ${tag} ..."
        ( python3 "${CELL_SCRIPT}" \
              --bands "${band}" \
              --rois "${roi}" \
              --conditions "${CONDITIONS[@]}" \
              --phase_rois "${PHASE_ROIS[@]}" \
              --schemes "${SCHEMES[@]}" \
              --voxRes "${VOX_RES}" \
              --points_per_category "${PPC}" \
              --n_hyperplanes "${N_HYP}" \
              --outdir "${PART_DIR}/${tag}" \
              ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"} \
        ) > "${LOG_DIR}/${tag}.log" 2>&1 &
        count=$((count + 1))
        if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Reached max parallel limit. Waiting..."; wait
        fi
    done
done
wait
echo ""
echo "[$(date '+%H:%M:%S')] All (band,roi) jobs finished."

# ── Concatenate the per-cell CSVs into one table ────────────────────────────
python3 - "${PART_DIR}" "${BASE_DIR}/group_task-mgs_glueEpochsPooled_stim_${VOX_RES}.csv" <<'PYEOF'
import sys, glob, os
import pandas as pd
part_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(part_dir, '*', '*.csv')))
if not files:
    print('No per-cell CSVs found -- nothing to concatenate.'); raise SystemExit(0)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out, index=False)
print(f'Combined {len(files)} per-cell CSVs -> {out}  ({len(df)} rows)')
PYEOF

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
