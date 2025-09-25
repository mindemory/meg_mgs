#!/bin/bash
# HPC Batch Script for S03A Frequency Power Analysis
# Usage: sbatch executeMatlabOnHPC.sh

#SBATCH --job-name=s03a_freq
#SBATCH --output=slurmOutput/s03a_freq_sub-%02a_%j.out
#SBATCH --time=06:00:00
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

# Define surface resolutions to process
surface_resolutions=(5 8 10)

# Define frequency bands to process
frequency_bands=("theta" "alpha" "beta" "lowgamma")
# frequency_bands=("beta")


echo "Processing Subject $subjID (sub-$subjID_padded)"

for frequency_band in "${frequency_bands[@]}"
do
    # Process each surface resolution
    for surface_resolution in "${surface_resolutions[@]}"
    do
        echo "Processing resolution $surface_resolution for subject $subjID"
        
        # Check if source data exists
        source_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_sourceSpaceData_${surface_resolution}.mat"
        if [ ! -f "$source_file" ]; then
            echo "Source file $source_file not found, skipping..."
            continue
        fi
        
        # Run S03 analysis
        matlab -nodisplay -nosplash -r "S03A_FrequencyPowerInSource($subjID, $surface_resolution, '$frequency_band'); exit;"
        
        echo "Completed resolution $surface_resolution for subject $subjID"
    done
done

echo "All processing complete for subject $subjID"
