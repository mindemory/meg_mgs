#!/bin/bash
# local_pilot_sub01.sh
RES="8mm"

echo "[*] Sub-01 | Initiating Single-Load Optimized Pilot Matrix..."
/users/mrugank/.conda/envs/megAnalyses/bin/python3 -u megScripts/inSourceSpacedirectionalConnectivity.py 1 "$RES"
