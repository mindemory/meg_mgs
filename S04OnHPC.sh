#!/bin/bash
# HPC Batch Script for S04 Beta Power Analysis in MNI Space
# Usage: sbatch S04OnHPC.sh

#SBATCH --job-name=s04_beta
#SBATCH --output=slurmOutput/s04_beta_sub-%02a_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
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

# Define surface resolutions to process
surface_resolutions=(5124 8196)

echo "Processing Subject $subjID (sub-$subjID_padded)"

# Process each surface resolution
for surface_resolution in "${surface_resolutions[@]}"
do
    echo "Processing resolution $surface_resolution for subject $subjID"
    
    # Check if complex beta data exists (from S03)
    beta_data_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_complexBeta_allTargets_${surface_resolution}.mat"
    if [ ! -f "$beta_data_file" ]; then
        echo "Complex beta data file $beta_data_file not found, skipping..."
        continue
    fi
    
    # Check if forward model exists
    forward_model_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_forwardModel.mat"
    if [ ! -f "$forward_model_file" ]; then
        echo "Forward model file $forward_model_file not found, skipping..."
        continue
    fi
    
    # Run S04 analysis
    matlab -nodisplay -nosplash -r "S04_betaPowerInMNI($subjID, $surface_resolution, false); exit;"
    
    echo "Completed resolution $surface_resolution for subject $subjID"
done

echo "All processing complete for subject $subjID"
