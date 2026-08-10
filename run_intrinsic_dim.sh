#!/usr/bin/env bash
# run_intrinsic_dim.sh
#
# Parallelised intrinsic-dimensionality computation for glue_decoding.
#
# One background job per (subject, lockType, band) cell -- up to
# 21 subjects x 2 lockTypes x 6 bands = 252 cells.  The default
# MAX_PARALLEL of 40 keeps vader comfortable; adjust as needed.
#
# Each cell is handled by glue_decoding/intrinsic_dim_cell.py, which
# saves one .npz file per cell (skip-if-exists is inside the cell
# script, so re-running after a partial run only computes what is
# missing).
#
# After all cells finish, runs intrinsic_dimensionality.py to produce
# the cross-subject mean +/- SEM figures (one call per lockType).
#
# Usage:
#   bash run_intrinsic_dim.sh [voxRes] [max_parallel]
#
# Examples:
#   bash run_intrinsic_dim.sh               # 8mm, up to 40 parallel
#   bash run_intrinsic_dim.sh 8mm 20        # throttle to 20

set -euo pipefail

VOX_RES="${1:-8mm}"
MAX_PARALLEL="${2:-40}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
LOCK_TYPES=("stim" "resp")
BANDS=("broadband" "theta" "alpha" "beta" "lowgamma" "highgamma")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/glue_decoding/intrinsic_dim_cell.py"
PLOT_SCRIPT="${SCRIPT_DIR}/glue_decoding/intrinsic_dimensionality.py"

# Single-threaded BLAS inside every job -- the outer cell grid IS the
# parallelism; oversubscribing cores here would slow each job down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========================================================"
echo " intrinsic_dim Runner (Vader)"
echo " VoxRes       : ${VOX_RES}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Lock types   : ${LOCK_TYPES[*]}"
echo " Bands        : ${BANDS[*]}"
echo "========================================================"

LOG_DIR="logs_intrinsic_dim_${VOX_RES}"
mkdir -p "${LOG_DIR}"
echo "Logging to: ${LOG_DIR}/"
echo ""

# ── Fan out: one job per (subject, lockType, band) ──────────────────────────
count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))          # strip leading zero for Python arg
    for lockType in "${LOCK_TYPES[@]}"; do
        for band in "${BANDS[@]}"; do
            log_file="${LOG_DIR}/sub-${subjID}_${lockType}_${band}.log"
            echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} ${lockType} ${band} ..."

            ( python3 "${CELL_SCRIPT}" "${subj_num}" "${lockType}" "${band}" \
                  --voxRes "${VOX_RES}" ) > "${log_file}" 2>&1 &

            count=$((count + 1))
            if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
                echo "[$(date '+%H:%M:%S')] Reached max parallel limit (${MAX_PARALLEL}). Waiting for batch..."
                wait
                echo "[$(date '+%H:%M:%S')] Batch complete. Continuing..."
            fi
        done
    done
done

# Wait for any remaining jobs
wait
echo ""
echo "[$(date '+%H:%M:%S')] All cell jobs finished."

# ── Plotting: aggregate + plot across all subjects ───────────────────────────
echo ""
echo "========================================================"
echo " Running plotter ..."
echo "========================================================"

python3 "${PLOT_SCRIPT}" \
    --voxRes "${VOX_RES}" \
    --lockTypes stim resp \
    > "${LOG_DIR}/plotter.log" 2>&1

echo "[$(date '+%H:%M:%S')] Plots complete. Log: ${LOG_DIR}/plotter.log"
echo ""
echo "========================================================"
echo " All done!  $(date)"
echo "========================================================"
