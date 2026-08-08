#!/usr/bin/env bash
# run_G03_vader.sh
#
# Runs G03_SourceLocalizationBroadband in parallel across all subjects on
# Vader. For each subject, 'stim' and 'resp' are run IN SEQUENCE within the
# same backgrounded job. G03 always derives the shared beamformer filter
# from STIM-LOCKED covariance regardless of run order (it self-loads
# stim-locked data internally if 'resp' runs before a cache exists -- see
# the "Shared beamformer filter" note in G03_SourceLocalizationBroadband.m),
# so this ordering isn't required for correctness; it's kept simply to
# avoid two parallel jobs for the same subject both computing and writing
# the same cached filter file at once. Different subjects still run
# concurrently up to max_parallel.
#
# Usage:
#   bash run_G03_vader.sh [resolution] [max_parallel]
#
# Defaults to 21 (all subjects at once) -- vader has ~50 cores, so the
# full cohort fits comfortably. Pass a lower number to throttle. Note
# each subject's background job runs TWO matlab calls in sequence (stim
# then resp), so effective peak concurrency is max_parallel simultaneous
# subjects, not max_parallel simultaneous matlab processes.
#
# Examples:
#   bash run_G03_vader.sh 8       # all 21 subjects in parallel (default)
#   bash run_G03_vader.sh 8 8     # throttle to 8 subjects at a time

set -euo pipefail

RESOLUTION="${1:-8}"
MAX_PARALLEL="${2:-21}"

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
        matlab9.13 -nodisplay -nosplash -nodesktop -r "G03_SourceLocalizationBroadband(${subj_num}, 'stim', ${RESOLUTION}); exit;" > "$stim_log" 2>&1
        matlab9.13 -nodisplay -nosplash -nodesktop -r "G03_SourceLocalizationBroadband(${subj_num}, 'resp', ${RESOLUTION}); exit;" > "$resp_log" 2>&1
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
