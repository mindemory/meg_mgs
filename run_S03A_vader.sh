#!/usr/bin/env bash
# run_S03A_vader.sh
#
# Runs S03A_FrequencyPowerInSource in parallel across all subjects on Vader.
# Outputs logs for each subject to a logs directory.
#
# Usage:
#   bash run_S03A_vader.sh [resolution] [band] [max_parallel]
#
# Examples:
#   bash run_S03A_vader.sh 5124 alpha 8
#   bash run_S03A_vader.sh 5124 alpha 21  # Run all 21 subjects in parallel

set -euo pipefail

RESOLUTION="${1:-5124}"
BAND="${2:-alpha}"
MAX_PARALLEL="${3:-8}"

# Subject list
SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)

echo "========================================================"
echo " S03A Parallel Runner (Vader)"
echo " Resolution   : ${RESOLUTION}"
echo " Band         : ${BAND}"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo "========================================================"

# Create logs directory
LOG_DIR="logs_S03A_${BAND}_${RESOLUTION}"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    # Remove leading zero for MATLAB function input
    subj_num=$((10#$subjID))
    
    log_file="${LOG_DIR}/sub-${subjID}.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} in background..."
    
    # Run MATLAB in background
    matlab -nodisplay -nosplash -nodesktop -r "S03A_FrequencyPowerInSource(${subj_num}, ${RESOLUTION}, '${BAND}'); exit;" > "$log_file" 2>&1 &
    
    count=$((count + 1))
    
    # Limit concurrency
    if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Reached max parallel limit ($MAX_PARALLEL). Waiting for current batch to complete..."
        wait
        echo "[$(date '+%H:%M:%S')] Batch complete. Starting next batch..."
    fi
done

# Wait for any remaining background jobs
wait
echo "========================================================"
echo " All subjects completed! $(date)"
echo "========================================================"
