#!/usr/bin/env bash
# run_G04_vader.sh
#
# Runs G04_BandAmplitudePhaseInSource across all subjects on Vader. For
# each subject, ALL 5 bands x both lock types are run IN SEQUENCE within
# the same backgrounded job (10 matlab calls per subject: theta/alpha/
# beta/lowgamma/highgamma x stim/resp) -- no need to invoke this script
# separately per band. Different subjects still run concurrently up to
# max_parallel. There's no ordering constraint between bands or lock
# types here (unlike G03's shared-beamformer-filter dependency) -- this
# is purely to bundle a subject's full band x lockType sweep into one job.
#
# Usage:
#   bash run_G04_vader.sh [resolution] [max_parallel] [bands...]
#
# Defaults to 21 (all subjects at once) -- vader has ~50 cores, so the
# full cohort fits comfortably. Pass a lower number to throttle. Extra
# trailing args override the default band list (useful for re-running
# just one or two bands without redoing all five).
#
# Examples:
#   bash run_G04_vader.sh                    # all 21 subjects, all 5 bands, both lock types
#   bash run_G04_vader.sh 8 8                # throttle to 8 subjects at a time
#   bash run_G04_vader.sh 8 21 highgamma     # only re-run highgamma for all subjects

set -euo pipefail

RESOLUTION="${1:-8}"
MAX_PARALLEL="${2:-21}"
shift $(( $# < 2 ? $# : 2 )) || true
if [ "$#" -gt 0 ]; then
    BANDS=("$@")
else
    BANDS=("theta" "alpha" "beta" "lowgamma" "highgamma")
fi

SUBJ_LIST=(01 02 03 04 05 06 07 09 10 12 13 15 17 18 19 23 24 25 29 31 32)
LOCK_TYPES=("stim" "resp")

echo "========================================================"
echo " G04 Band Amplitude(+Phase) Extraction Runner (Vader)"
echo " Resolution   : ${RESOLUTION}mm"
echo " Max Parallel : ${MAX_PARALLEL}"
echo " Subjects     : ${SUBJ_LIST[*]}"
echo " Bands        : ${BANDS[*]}"
echo " Lock types   : ${LOCK_TYPES[*]}"
echo "========================================================"

LOG_DIR="logs_G04_${RESOLUTION}mm"
mkdir -p "$LOG_DIR"
echo "Logging output to: ${LOG_DIR}/"
echo ""

count=0
for subjID in "${SUBJ_LIST[@]}"; do
    subj_num=$((10#$subjID))
    echo "[$(date '+%H:%M:%S')] Starting sub-${subjID} (all bands x lock types) in background..."

    (
        for band in "${BANDS[@]}"; do
            for lockType in "${LOCK_TYPES[@]}"; do
                log_file="${LOG_DIR}/sub-${subjID}_${band}_${lockType}.log"
                matlab9.13 -nodisplay -nosplash -nodesktop -r "G04_BandAmplitudePhaseInSource(${subj_num}, '${lockType}', '${band}', ${RESOLUTION}); exit;" > "$log_file" 2>&1
            done
        done
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
