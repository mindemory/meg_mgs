#!/usr/bin/env bash
# run_tgm_all.sh
#
# Batch runner for temporalGeneralizationDecoding.py
# Runs sequentially across subjects and frequency bands — each run is already
# parallelized internally via joblib (48 cores on Vader).
#
# Usage (on Vader):
#   bash megScripts/run_tgm_all.sh [voxRes]
#
# Default voxRes: 8mm

set -euo pipefail

VOX_RES="${1:-8mm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIDS_ROOT="$(python3 -c "import socket; h=socket.gethostname(); \
  print('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='zod' \
  else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h=='vader' \
  else '/scratch/mdd9787/meg_prf_greene/MEG_HPC')")"

FREQ_BANDS=(theta alpha beta lowgamma)

# Discover subjects that have sourceRecon data
SUBJ_LIST=()
for subdir in "${BIDS_ROOT}/derivatives"/sub-*/sourceRecon; do
    subname=$(basename "$(dirname "$subdir")")
    subjID="${subname#sub-}"
    # Check at least one source file exists
    if ls "${subdir}/${subname}_task-mgs_sourceSpaceData_"*.mat &>/dev/null 2>&1; then
        SUBJ_LIST+=("$subjID")
    fi
done

echo "========================================================"
echo " TGM Batch Runner"
echo " VoxRes   : ${VOX_RES}"
echo " Subjects : ${SUBJ_LIST[*]}"
echo " Bands    : ${FREQ_BANDS[*]}"
echo " Host     : $(hostname)"
echo "========================================================"

TOTAL=$(( ${#SUBJ_LIST[@]} * ${#FREQ_BANDS[@]} ))
COUNT=0

for subjID in "${SUBJ_LIST[@]}"; do
    for band in "${FREQ_BANDS[@]}"; do
        COUNT=$(( COUNT + 1 ))

        # Check if output already exists — skip if so
        OUTPUT_FILE="${BIDS_ROOT}/derivatives/sub-${subjID}/sourceRecon/decodingVC/sub-${subjID}_task-mgs_TGM_${band}_${VOX_RES}.pkl"
        if [ -f "$OUTPUT_FILE" ]; then
            echo "[${COUNT}/${TOTAL}] SKIP sub-${subjID} ${band} — already done"
            continue
        fi

        echo ""
        echo "[${COUNT}/${TOTAL}] Running sub-${subjID} | ${band} | $(date '+%H:%M:%S')"
        python3 "${SCRIPT_DIR}/temporalGeneralizationDecoding.py" "$subjID" "$VOX_RES" "$band"
        echo "[${COUNT}/${TOTAL}] Done    sub-${subjID} | ${band} | $(date '+%H:%M:%S')"

    done
done

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
