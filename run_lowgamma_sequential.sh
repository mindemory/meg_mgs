#!/bin/bash
# Sequential Batch Processor for Multi-Frequency CFC (Low-Gamma Expansion)
# Cohort: 21 Subjects
# Running one by one to avoid resource exhaustion and "getting stuck"

subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)
voxRes="8mm"

echo "[*] Launching 21-Subject Low-Gamma Expansion (SEQUENTIAL MODE)..."
echo "[*] Engine: inSourceSpacedirectionalConnectivity.py"
echo "[*] Environment: megAnalyses"

for sub in "${subjects[@]}"; do
    printf -v sub_pad "%02d" $sub
    echo "--------------------------------------------------------"
    echo "  [+] Starting sub-${sub_pad} at $(date)"
    
    # Run sequentially using conda run to ensure environment activation
    conda run -n megAnalyses python megScripts/inSourceSpacedirectionalConnectivity.py $sub $voxRes
    
    if [ $? -eq 0 ]; then
        echo "  [+] sub-${sub_pad} complete."
    else
        echo "  [!] ERROR: sub-${sub_pad} failed."
    fi
done

echo "--------------------------------------------------------"
echo "[+] ALL SUBJECTS PROCESSED. Proceeding to consolidation if needed."
