#!/bin/bash
# local_pilot_sub01.sh
RES="8mm"
LOW_BANDS=("4 8" "8 13" "13 30") 
HIGH_BANDS=("4 8" "8 13" "13 30") 
SEEDS=("left_frontal" "right_frontal")
LOCS=("left" "right")

for LB in "${LOW_BANDS[@]}"; do
    for HB in "${HIGH_BANDS[@]}"; do
        read sf_low sf_high <<< "$LB"
        read tf_low tf_high <<< "$HB"
        for SEED in "${SEEDS[@]}"; do
            for LOC in "${LOCS[@]}"; do
                echo "[*] Sub-01 | ${SEED} -> ${LOC} | Phase(${sf_low}-${sf_high}) -> Env(${tf_low}-${tf_high})"
                python3 megScripts/inSourceSpacedirectionalConnectivity.py 1 "$RES" "$SEED" "$LOC" "$sf_low" "$sf_high" "$tf_low" "$tf_high"
            done
        done
    done
done
