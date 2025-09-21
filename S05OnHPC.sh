#!/bin/bash
#SBATCH --job-name=s05_tafkap_parallel
#SBATCH --output=slurmOutput/s05_tafkap_sub-%02a_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-2

# Load MATLAB module
module load matlab/2024a

# Set up paths
PROJECT_DIR="/scratch/mdd9787/meg_prf_greene"
MEG_SCRIPTS_DIR="$PROJECT_DIR/megScripts"
OUTPUT_DIR="$PROJECT_DIR/MEG_HPC/derivatives"

# Change to the scripts directory
cd $MEG_SCRIPTS_DIR

# Get subject ID from command line argument (default to 1)
SUBJ_ID=${1:-1}
SURFACE_RES=${2:-5124}

# Define time points for each batch (from -0.5 to 1.5 in steps of 0.2)
# Batch 1: -0.5, -0.3, -0.1, 0.1, 0.3, 0.5
# Batch 2: 0.7, 0.9, 1.1, 1.3, 1.5

# case $SLURM_ARRAY_TASK_ID in
#     1) time_points=(-0.5 -0.3 -0.1 0.1 0.3 0.5) ;;
#     2) time_points=(0.7 0.9 1.1 1.3 1.5) ;;
# esac

case $SLURM_ARRAY_TASK_ID in
    1) time_points=(0.1 0.3 0.5 0.7) ;;
    2) time_points=(0.9 1.1 1.3 1.5) ;;
esac

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_decoding

echo "Starting TAFKAP decoding for Subject $SUBJ_ID with surface resolution $SURFACE_RES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Time points in this batch: ${time_points[@]}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run TAFKAP for each time point in this batch
for INPUT_TIME in "${time_points[@]}"; do
    echo "Processing time point: $INPUT_TIME (analysis window: $INPUT_TIME-$(echo "$INPUT_TIME + 0.2" | bc)s)"
    
    # Run the MATLAB script with TAFKAP
    matlab -nodisplay -nosplash -r "S05_TAFKAP_Decoding($SUBJ_ID, $SURFACE_RES, 'TAFKAP', $INPUT_TIME); exit"
    
    # Check if this time point completed successfully
    if [ $? -eq 0 ]; then
        echo "TAFKAP decoding completed successfully for Subject $SUBJ_ID at time $INPUT_TIME"
        
        # List output files for this time point
        echo "Output files created for time $INPUT_TIME:"
        ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_decoding/*t${INPUT_TIME}*
    else
        echo "TAFKAP decoding failed for Subject $SUBJ_ID at time $INPUT_TIME"
        echo "Continuing with next time point..."
    fi
done

echo "All processing complete for subject $SUBJ_ID in batch $SLURM_ARRAY_TASK_ID"
echo "End time: $(date)"
