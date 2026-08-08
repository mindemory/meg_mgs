#!/usr/bin/env bash
# run_G03_vader.sh
#
# Runs G03_SourceLocalizationBroadband in parallel across all subjects on
# Vader. For each subject, 'stim' and 'resp' are run IN SEQUENCE within the
# same backgrounded job -- not because either lockType requires the other
# to run first (both stim- and resp-locked epochs are uniform-length, so
# covariance/LCMV estimation is valid from either), but to avoid two
# parallel MATLAB jobs for the same subject racing to compute and write
# the same cached beamformer filter file at once (see the "Shared
# beamformer filter" note in G03_SourceLocalizationBroadband.m). Different
# subjects still run concurrently up to max_parallel.
#
# Usage:
#   bash run_G03_vader.sh [resolution] [max_parallel]
#
# Examples:
#   bash run_G03_vader.sh 8 8

set -euo pipefail

RESOLUTION="${1:-8}"
MAX_PARALLEL="${2:-8}"

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)

echo "========================================================"
echo " G03 Broadband Source Localization Runner (Vader)"
echo " Resolution   : ${RESOLUTION}mm"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo "========================================================"

LOG_DIR="logs_G03_${RESOLUTION}mm"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))

    stim_log="${LOG_DIR}/sub-${subjID}_stim.log"
    resp_log="${LOG_DIR}/sub-${subjID}_resp.log"
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (stim then resp) in background..."

    (
        matlab -nodisplay -nosplash -nodesktop -r "G03_SourceLocalizationBroadband(${subj_num}, 'stim', ${RESOLUTION}); exit;" > "$stim_log" 2>&1
        matlab -nodisplay -nosplash -nodesktop -r "G03_SourceLocalizationBroadband(${subj_num}, 'resp', ${RESOLUTION}); exit;" > "$resp_log" 2>&1
    ) &

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
