#!/usr/bin/env bash
# run_pac_all.sh
#
# Batch runner for compute_pac.py
# Runs sequentially across all valid subjects. The Python script handles both 
# Visual and Frontal regions and all frequency bands internally.
#
# Usage (on Vader/Mac):
#   bash megScripts/run_pac_all.sh [voxRes]
#
# Default voxRes: 10mm

set -euo pipefail

VOX_RES="${1:-10mm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIDS_ROOT="$(python3 -c "import socket; h=socket.gethostname(); \
  print('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='zod' \
  else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='vader' \
  else '/scratch/mdd9787/meg_prf_greene/MEG_HPC')")"

# Discover subjects that have sourceRecon data
SUBJ_LIST=()
for subdir in "${BIDS_ROOT}/derivatives"/sub-*/sourceRecon; do
    subname=$(basename "$(dirname "$subdir")")
    subjID="${subname#sub-}"
    # Check at least one source task-mgs file exists
    if ls "${subdir}/${subname}_task-mgs_sourceSpaceData_"*.mat &>/dev/null 2>&1; then
        SUBJ_LIST+=("$subjID")
    fi
done

echo "========================================================"
echo " PAC Batch Runner (Caching)"
echo " VoxRes   : ${VOX_RES}"
echo " Subjects : ${SUBJ_LIST[*]}"
echo " Host     : $(hostname)"
echo "========================================================"

TOTAL=${#SUBJ_LIST[@]}
COUNT=0

for subjID in "${SUBJ_LIST[@]}"; do
    COUNT=$(( COUNT + 1 ))

    # Check if both Visual and Frontal outputs already exist — skip if so
    VIS_OUT="${BIDS_ROOT}/derivatives/sub-${subjID}/sourceRecon/pac_data/sub-${subjID}_task-mgs_PAC_Visual_${VOX_RES}.pkl"
    FRONT_OUT="${BIDS_ROOT}/derivatives/sub-${subjID}/sourceRecon/pac_data/sub-${subjID}_task-mgs_PAC_Frontal_${VOX_RES}.pkl"
    
    if [ -f "$VIS_OUT" ] && [ -f "$FRONT_OUT" ]; then
        echo "[${COUNT}/${TOTAL}] SKIP sub-${subjID} — already computed"
        continue
    fi

    echo ""
    echo "[${COUNT}/${TOTAL}] Running sub-${subjID} | $(date '+%H:%M:%S')"
    python3 "${SCRIPT_DIR}/compute_pac.py" "$subjID" "$VOX_RES"
    echo "[${COUNT}/${TOTAL}] Done    sub-${subjID} | $(date '+%H:%M:%S')"
done

echo ""
echo "========================================================"
echo " All PAC computations done! $(date)"
echo "========================================================"
