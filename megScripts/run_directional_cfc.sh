#!/bin/bash
# run_directional_cfc.sh: Modular batch script for CFC directionality
# Saves to derivatives/sub-*/sourceRecon/CFC_8mm/

RES=$1
if [ -z "$RES" ]; then RES="8mm"; fi

# Subject lists (1-6 for pilot)
SUBJS=(1 2 3 4 5 6)

for SUB in "${SUBJS[@]}"; do
    echo "[*] Sub-${SUB} | Initializing Single-Load CFC Matrix..."
    python3 megScripts/inSourceSpacedirectionalConnectivity.py $SUB $RES
done
