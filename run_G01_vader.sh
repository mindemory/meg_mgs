#!/usr/bin/env bash
# run_G01_vader.sh
#
# Runs G01_ExtractRespLockedEpochs in parallel across all subjects on Vader.
# Outputs logs for each subject to a logs directory.
#
# Usage:
#   bash run_G01_vader.sh [max_parallel]
#
# Examples:
#   bash run_G01_vader.sh 8
#   bash run_G01_vader.sh 21  # Run all 21 subjects in parallel

set -euo pipefail

MAX_PARALLEL="${1:-8}"

# Subject list
SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)

echo "========================================================"
echo " G01 Resp-Locked Epoch Extraction Runner (Vader)"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo "========================================================"

LOG_DIR="logs_G01"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))

    log_file="${LOG_DIR}/sub-${subjID}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} in background..."

    matlab9.13 -nodisplay -nosplash -nodesktop -r "G01_ExtractRespLockedEpochs(${subj_num}); exit;" > "$log_file" 2>&1 &

    count=$((count + 1))
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit ($MAX_PARALLEL). Waiting for current batch to complete..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Starting next batch..."
    fi
done

wait
echo "========================================================"
echo " All subjects completed! $(date)"
echo "========================================================"
