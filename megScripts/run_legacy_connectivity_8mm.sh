#!/bin/bash
# ==============================================================================
# run_legacy_connectivity_8mm.sh
# Performs 8mm connectivity computation using the original project script
# to ensure 100% fidelity to the "Old Look" (integrated window logic).
# ==============================================================================

# List of 21 subjects
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)
VOXRES="8mm"

echo "========================================================"
echo " Starting Sequential 8mm Connectivity Re-run (BETA)"
echo " Metrics  : Coherence, ImCoh, dPLI (Internal Parallel)"
echo " Script   : inSourceSpaceConnectivity.py"
echo " Band     : Beta (18-30 Hz)"
echo " Start    : $(date)"
echo "========================================================"

for sub in "${subjects[@]}"; do
    SUB_NAME=$(printf "sub-%02d" $sub)
    OUT_DIR="/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/${SUB_NAME}/sourceRecon/connectivity"
    OUT_FILE="${OUT_DIR}/${SUB_NAME}_task-mgs_connectivity_${VOXRES}.pkl"
    
    if [ -f "$OUT_FILE" ]; then
        echo "[-] Skipping ${SUB_NAME} (Already exists)"
        continue
    fi

    echo "--------------------------------------------------------"
    echo "▶ Processing ${SUB_NAME}..."
    python megScripts/inSourceSpaceConnectivity.py $sub $VOXRES
    
    if [ $? -eq 0 ]; then
        echo "  [✓] Completed ${SUB_NAME}"
    else
        echo "  [✗] ERROR on ${SUB_NAME}"
    fi
done

echo "========================================================"
echo "✓ 8mm Legacy Cohort Re-run Complete!"
echo "Finish   : $(date)"
echo "========================================================"
