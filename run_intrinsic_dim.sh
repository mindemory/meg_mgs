#!/usr/bin/env bash
# run_intrinsic_dim.sh
#
# Runs epoch-level intrinsic-dimensionality analysis for glue_decoding.
#
# Calls intrinsic_dim_epochs.py directly -- no shell-level fan-out needed.
# Parallelism is handled internally by joblib (one worker per subject), so
# this script just launches a single Python process.
#
# Usage:
#   bash run_intrinsic_dim.sh [voxRes] [n_jobs]
#
# Examples:
#   bash run_intrinsic_dim.sh              # 8mm, n_jobs = n_subjects (21)
#   bash run_intrinsic_dim.sh 8mm 8        # cap at 8 parallel workers

set -euo pipefail

VOX_RES="${1:-8mm}"
N_JOBS="${2:-21}"

SUBJ_LIST=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCH_SCRIPT="${SCRIPT_DIR}/glue_decoding/intrinsic_dim_epochs.py"

# Single-threaded BLAS per worker -- joblib already provides the outer
# parallelism (one process per subject).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR="logs_intrinsic_dim_${VOX_RES}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/intrinsic_dim_epochs.log"

echo "========================================================"
echo " intrinsic_dim_epochs Runner"
echo " VoxRes   : ${VOX_RES}"
echo " N jobs   : ${N_JOBS}"
echo " Subjects : ${SUBJ_LIST[*]}"
echo " Epochs   : stim (0.0-0.2 s)  |  delay (0.2-1.7 s)"
echo " Log      : ${LOG_FILE}"
echo "========================================================"
echo ""

python3 "${EPOCH_SCRIPT}" \
    --voxRes   "${VOX_RES}" \
    --subjects "${SUBJ_LIST[@]}" \
    --n_jobs   "${N_JOBS}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================================"
echo " All done!  $(date)"
echo "========================================================"
