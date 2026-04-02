#!/usr/bin/env bash
# run_pac_all.sh
#
# Optimized batch runner for Cross-Regional PAC Suite.
# Each subject runs 48 parallel tasks (3 bands x 16 interaction pairs) internally.
# We run subjects sequentially to avoid oversubscribing the CPU.
#
# Usage:
#   bash megScripts/run_pac_all.sh [voxRes]

set -euo pipefail

VOX_RES="${1:-8mm}"
SCRIPT_DIR=$( \cd "$(dirname "$0")" > /dev/null && pwd )
BIDS_ROOT="$(python3 -c "import socket; h=socket.gethostname(); \
  print('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='zod' \
  else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='vader' \
  else '/scratch/mdd9787/meg_prf_greene/MEG_HPC')")"

# Explicit subject list for the project
SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)

echo "========================================================"
echo " Cross-Regional PAC Batch Runner"
echo " VoxRes   : ${VOX_RES}"
echo " Subjects : ${SUBJ_LIST[*]}"
echo " Host     : $(hostname)"
echo "========================================================"

for subjID in "${SUBJ_LIST[@]}"; do
    OUT_FILE="${BIDS_ROOT}/derivatives/sub-${subjID}/sourceRecon/pac_data/sub-${subjID}_CrossRegional_PAC_${VOX_RES}.pkl"
    
    if [ -f "$OUT_FILE" ]; then
        echo "SKIP sub-${subjID} — already exists"
        continue
    fi

    echo ""
    echo ">>> Running sub-${subjID} | $(date '+%H:%M:%S')"
    # Use conda to ensure environment is correct if needed, or assume caller handled it
    python3 "${SCRIPT_DIR}/compute_pac.py" "$subjID" "$VOX_RES"
    echo "<<< Done    sub-${subjID} | $(date '+%H:%M:%S')"
done

# Run visualization after all subjects are done
echo ""
echo ">>> Generating Master Figures..."
python3 "${SCRIPT_DIR}/visualizePAC.py"

echo ""
echo "========================================================"
echo " All PAC computations and visualizations done! $(date)"
echo "========================================================"
