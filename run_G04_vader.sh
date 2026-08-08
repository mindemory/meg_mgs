#!/usr/bin/env bash
# run_G04_vader.sh
#
# Runs G04_BandAmplitudePhaseInSource in parallel across all subjects and
# both lock types on Vader, for a single band per invocation (mirrors
# run_S03A_vader.sh's one-band-per-call convention). No stim/resp
# ordering constraint here (unlike G03) -- both lock types for a subject
# can run concurrently since each only reads that lock type's already-
# computed G03 broadband output.
#
# Usage:
#   bash run_G04_vader.sh [band] [resolution] [max_parallel]
#
# Bands: theta | alpha | beta | lowgamma | highgamma
#
# Examples:
#   bash run_G04_vader.sh beta 8 8
#   bash run_G04_vader.sh highgamma 8 16

set -euo pipefail

BAND="${1:-beta}"
RESOLUTION="${2:-8}"
MAX_PARALLEL="${3:-8}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
LOCK_TYPES=("stim" "resp")

echo "========================================================"
echo " G04 Band Amplitude(+Phase) Extraction Runner (Vader)"
echo " Band         : ${BAND}"
echo " Resolution   : ${RESOLUTION}mm"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Lock types   : ${LOCK_TYPES[*]}"
echo "========================================================"

LOG_DIR="logs_G04_${BAND}_${RESOLUTION}mm"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    for lockType in "${LOCK_TYPES[@]}"; do
        log_file="${LOG_DIR}/sub-${subjID}_${lockType}.log"
        echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (${lockType}) in background..."

        matlab9.13 -nodisplay -nosplash -nodesktop -r "G04_BandAmplitudePhaseInSource(${subj_num}, '${lockType}', '${BAND}', ${RESOLUTION}); exit;" > "$log_file" 2>&1 &

        count=$((count + 1))
        if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Reached max parallel limit ($MAX_PARALLEL). Waiting for current batch to complete..."
            wait
            echo "[$(date '+%H:%M:%S')] Batch complete. Starting next batch..."
        fi
    done
done

wait
echo "========================================================"
echo " All subjects completed! $(date)"
echo "========================================================"
