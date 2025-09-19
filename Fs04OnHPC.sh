#!/bin/bash
# HPC Batch Script for Fs04 Beta Power Topography Visualization
# Usage: sbatch Fs04OnHPC.sh
#        sbatch --array=0 Fs04OnHPC.sh  (for group average only)

#SBATCH --job-name=fs04_viz
#SBATCH --output=slurmOutput/fs04_viz_sub-%02a_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=END
#SBATCH --array=0-21

# Load MATLAB module (adjust as needed for your HPC)
module load matlab/2024a

# Create output directory if it doesn't exist
mkdir -p slurmOutput

# Define subjects array
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Define surface resolutions to process
surface_resolutions=(5124 8196)

# Check if this is group average job (array task 0)
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Creating group average visualizations"
    
    # Process each surface resolution for group average
    for surface_resolution in "${surface_resolutions[@]}"
    do
        echo "Processing group average for resolution $surface_resolution"
        
        # Run Fs04 group average visualization
        matlab -nodisplay -nosplash -r "Fs04_visualizeBetaTopographyByLoc('all', $surface_resolution); exit;"
        
        echo "Completed group average for resolution $surface_resolution"
    done
    
    echo "All group average visualizations complete"
    
else
    # Individual subject processing
    subjID=${subjects[$((SLURM_ARRAY_TASK_ID-1))]}
    subjID_padded=$(printf "%02d" $subjID)
    
    echo "Processing Subject $subjID (sub-$subjID_padded)"
    
    # Process each surface resolution
    for surface_resolution in "${surface_resolutions[@]}"
    do
        echo "Processing resolution $surface_resolution for subject $subjID"
        
        # Check if relative power data exists (from S04)
        power_data_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_relativePowerBetaBand_${surface_resolution}.mat"
        if [ ! -f "$power_data_file" ]; then
            echo "Relative power data file $power_data_file not found, skipping..."
            continue
        fi
        
        # Check if forward model exists
        forward_model_file="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-${subjID_padded}/sourceRecon/sub-${subjID_padded}_task-mgs_forwardModel.mat"
        if [ ! -f "$forward_model_file" ]; then
            echo "Forward model file $forward_model_file not found, skipping..."
            continue
        fi
        
        # Run Fs04 visualization
        matlab -nodisplay -nosplash -r "Fs04_visualizeBetaTopographyByLoc($subjID, $surface_resolution); exit;"
        
        echo "Completed resolution $surface_resolution for subject $subjID"
    done
    
    echo "All visualization complete for subject $subjID"
fi
