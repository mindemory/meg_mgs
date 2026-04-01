#!/bin/bash
# run_directional_cfc_local_all.sh
RES="8mm"
SUBJS=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

echo "[*] Initiating Full Cohort (21 Subjects) CFC Analysis..."
for SUB in "${SUBJS[@]}"; do
    echo "--------------------------------------------------------"
    echo "[*] Processing Sub-${SUB}..."
    /users/mrugank/.conda/envs/megAnalyses/bin/python3 -u megScripts/inSourceSpacedirectionalConnectivity.py "$SUB" "$RES"
done

echo "[*] All subjects completed. Generating final group visualization..."
/users/mrugank/.conda/envs/megAnalyses/bin/python3 -u megScripts/visualizeDirectionalCFC.py
