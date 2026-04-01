#!/bin/bash
# run_directional_cfc.sh: Modular batch script for CFC directionality
# Saves to derivatives/sub-*/sourceRecon/CFC_8mm/

RES=$1
if [ -z "$RES" ]; then RES="8mm"; fi

# Frequencies for Phase (Frontal Seed) and Envelope (Visual Target)
LOW_BANDS=("4 8" "8 13" "13 30") 
HIGH_BANDS=("4 8" "8 13" "13 30") 

# Subject lists (1-6 for pilot)
SUBJS=(1 2 3 4 5 6)

SEEDS=("left_frontal" "right_frontal")
LOCS=("left" "right")

for SUB in "${SUBJS[@]}"; do
    for LB in "${LOW_BANDS[@]}"; do
        for HB in "${HIGH_BANDS[@]}"; do
            read sf_low sf_high <<< "$LB"
            read tf_low tf_high <<< "$HB"
            
            for SEED in "${SEEDS[@]}"; do
                for LOC in "${LOCS[@]}"; do
                    echo "[*] Sub-${SUB} | ${SEED} -> ${LOC} | Phase(${sf_low}-${sf_high}) -> Env(${tf_low}-${tf_high} AM)"
                    python3 megScripts/inSourceSpacedirectionalConnectivity.py $SUB $RES "$SEED" "$LOC" $sf_low $sf_high $tf_low $tf_high
                done
            done
        done
    done
done
