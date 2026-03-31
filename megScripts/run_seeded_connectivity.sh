#!/bin/bash
# ==============================================================================
# run_seeded_connectivity.sh
# Batch execution script to calculate Seeded Source Connectivity measures.
# ==============================================================================

if [ -z "$1" ]; then
    echo "Usage: bash run_seeded_connectivity.sh <VoxRes> <ConnectivityType>"
    echo "Example (Runs standard Coherence): bash run_seeded_connectivity.sh 10mm coh"
    echo "Example (Runs new Directed PLI)  : bash run_seeded_connectivity.sh 10mm dpli"
    exit 1
fi

VOXRES=$1
# Full list of metrics and targets to process in bulk
# (Loading once per subject/band/seed, calculating all combinations in RAM)
METRICS="imcoh,dpli"
TARGETS="left,right"
SEEDS=("left_visual" "right_visual" "left_frontal" "right_frontal")
BANDS=("theta" "alpha" "beta" "lowgamma")

echo "========================================================"
echo " Seeded Connectivity Bulk Runner"
echo " Metrics : $METRICS"
echo " Targets : $TARGETS"
echo " VoxRes  : $VOXRES"
echo " Host    : $(hostname)"
echo "========================================================"

for sub in "${subjects[@]}"; do
    echo "--------------------------------------------------------"
    echo "▶ Bulk processing sub-${sub}"
    
    for band in "${bands[@]}"; do
        for seed in "${seeds[@]}"; do
            
            echo "  -> Processing [${band}] | ${seed} | Bulk: ${METRICS} x ${TARGETS}"
            python megScripts/inSourceSpaceSeededConnectivity.py $sub $VOXRES $seed "$TARGETS" "$METRICS" $band
            
            if [ $? -ne 0 ]; then
                echo "  [✗] Error processing sub-${sub} / ${band} / ${seed}"
            fi
            
        done
    done
done

echo "========================================================"
echo "✓ All Seeded Connectivity computations for metric '$CON_TYPE' complete!"

# Send bulk completion email
python -c "import smtplib, socket
try:
    hostname = socket.gethostname()
    subject = f'Subject: [Batch Complete] {\"$CON_TYPE\"} {\"$VOXRES\"} DONE on {hostname}'
    body = f'The automated 21-subject seeded connectivity batch run for metric {\"$CON_TYPE\"} at {\"$VOXRES\"} across all bands, seeds, and targets has successfully finished processing.'
    msg = f'{subject}\n\n{body}'
    with smtplib.SMTP('localhost') as s:
        s.sendmail('mrugank.dake@nyu.edu', 'mrugank.dake@nyu.edu', msg)
    print('  [✓] Bulk completion email successfully dispatched.')
except Exception as e:
    print(f'  [✗] Email notification skipped: {e}')"
