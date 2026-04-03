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
echo " Starting Legacy 8mm Connectivity Re-run (BETA)"
echo " Metric   : Coherence, ImCoh, PLV, PLI (Monolithic)"
echo " Script   : inSourceSpaceConnectivity.py"
echo " Band     : Beta (18-30 Hz)"
echo " Cohort   : ${#subjects[@]} subjects"
echo " Start    : $(date)"
echo "========================================================"

# Launch all subjects in parallel (Vader has 48 cores, we use 21)
for sub in "${subjects[@]}"; do
    echo "▶ Submitting sub-${sub} to background..."
    python megScripts/inSourceSpaceConnectivity.py $sub $VOXRES > /tmp/sub-${sub}_8mm_beta.log 2>&1 &
done

echo "--------------------------------------------------------"
echo "All 21 subjects submitted! Monitoring progress..."
echo "Estimated time: 45-60 minutes."
wait # Wait for all background processes to finish

echo "========================================================"
echo "✓ 8mm Legacy Cohort Re-run Complete!"
echo "Finish   : $(date)"
echo "========================================================"
