#!/usr/bin/env bash
# run_G02_vader.sh
#
# Runs G02_WangAtlasParcellation once for a given resolution (this is a
# single template-grid computation, not per-subject -- see the "one file
# per resolution" note in G02_WangAtlasParcellation.m).
#
# Usage:
#   bash run_G02_vader.sh [resolution]
#
# Examples:
#   bash run_G02_vader.sh 8

set -euo pipefail

RESOLUTION="${1:-8}"

echo "========================================================"
echo " G02 Wang Atlas Parcellation Runner (Vader)"
echo " Resolution : ${RESOLUTION}mm"
echo "========================================================"

LOG_DIR="logs_G02"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/rois_${RESOLUTION}mm.log"

matlab -nodisplay -nosplash -nodesktop -r "G02_WangAtlasParcellation(${RESOLUTION}); exit;" > "$log_file" 2>&1

echo "========================================================"
echo " Done! Log: ${log_file}  $(date)"
echo "========================================================"
