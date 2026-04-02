#!/usr/bin/env bash
# run_pac_all.sh
#
# Optimized batch runner for Quantified Cross-Regional PAC Suite.
# Each subject runs 96 parallel tasks (3-phase x 2-amp x 16 interaction pairs).
# Each subject utilizes 48 cores on Vader cluster.
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
echo " Quantified Cross-Regional PAC Batch Runner"
echo " VoxRes   : ${VOX_RES}"
echo " Subjects : ${SUBJ_LIST[*]}"
echo " Host     : $(hostname)"
echo "========================================================"

for subjID in "${SUBJ_LIST[@]}"; do
    OUT_FILE="${BIDS_ROOT}/derivatives/sub-${subjID}/sourceRecon/pac_data/sub-${subjID}_CrossRegional_PAC_Quantified_${VOX_RES}.pkl"
    
    if [ -f "$OUT_FILE" ]; then
        echo "SKIP sub-${subjID} — already quantified"
        continue
    fi

    echo ""
    echo ">>> Quantifying sub-${subjID} | $(date '+%H:%M:%S')"
    # Use 48 cores internally
    python3 "${SCRIPT_DIR}/compute_pac.py" "$subjID" "$VOX_RES"
    echo "<<< Done    sub-${subjID} | $(date '+%H:%M:%S')"
done

echo ""
echo ">>> Running Stats and Quantification Aggregation..."
python3 "${SCRIPT_DIR}/quantifyPAC.py"

echo ""
echo ">>> Generating Master Figures..."
python3 "${SCRIPT_DIR}/visualizePAC.py"

echo ""
echo "========================================================"
echo " All PAC quantification and stats done! $(date)"
echo "========================================================"
