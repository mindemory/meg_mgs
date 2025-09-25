#!/bin/bash
# HPC Batch Script for S02A Volumetric Reverse Model Analysis
# Usage: sbatch S02AOnHPC.sh

#SBATCH --job-name=s02a_volumetric
#SBATCH --output=slurmOutput/s02a_volumetric_sub-%02a_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=END
#SBATCH --array=1-21

# Load MATLAB module (adjust as needed for your HPC)
module load matlab/2024a

# Create output directory if it doesn't exist
mkdir -p slurmOutput

# Define subjects array
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get subject ID from array
subjID=${subjects[$((SLURM_ARRAY_TASK_ID-1))]}

# Zero-pad subject ID for BIDS naming
subjID_padded=$(printf "%02d" $subjID)

# Define volumetric resolutions to process
volumetric_resolutions=(5 8 10)

echo "Processing Subject $subjID (sub-$subjID_padded)"

for volumetric_resolution in "${volumetric_resolutions[@]}"
do
    echo "Processing resolution ${volumetric_resolution}mm for subject $subjID"

    # Define paths for dependency checks
    volumetric_source_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_volumetricSources_${volumetric_resolution}mm.mat"
    forward_model_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_forwardModel.mat"
    stimlocked_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/meg/sub-${subjID_padded}_task-mgs_stimlocked_lineremoved.mat"

    # Check dependencies
    if [ ! -f "$volumetric_source_file" ]; then
        echo "Volumetric source file $volumetric_source_file not found, skipping..."
        continue
    fi
    if [ ! -f "$forward_model_file" ]; then
        echo "Forward model file $forward_model_file not found, skipping..."
        continue
    fi
    if [ ! -f "$stimlocked_file" ]; then
        echo "Stimlocked file $stimlocked_file not found, skipping..."
        continue
    fi

    # Run S02A analysis
    matlab -nodisplay -nosplash -r "S02A_ReverseModelMNIVolumetric($subjID, $volumetric_resolution); exit;"

    echo "Completed resolution ${volumetric_resolution}mm for subject $subjID"
done

echo "All volumetric processing complete for subject $subjID"
