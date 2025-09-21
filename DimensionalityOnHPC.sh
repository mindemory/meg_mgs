#!/bin/bash
#SBATCH --job-name=s05_dimensionality
#SBATCH --output=slurmOutput/s05_dimensionality_sub-%02a_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-21

# Load MATLAB module
module load matlab/2024a

# Set up paths
PROJECT_DIR="/scratch/mdd9787/meg_prf_greene"
MEG_SCRIPTS_DIR="$PROJECT_DIR/megScripts"
OUTPUT_DIR="$PROJECT_DIR/MEG_HPC/derivatives"

# Change to the scripts directory
cd $MEG_SCRIPTS_DIR

# Define subjects array (20 subjects)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get subject ID from array task ID
SUBJ_ID=${subjects[$((SLURM_ARRAY_TASK_ID-1))]}
SURFACE_RES=5124

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis

echo "Starting dimensionality analysis for Subject $SUBJ_ID with surface resolution $SURFACE_RES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject ID: $SUBJ_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run dimensionality analysis for 5124 surface resolution
echo "Running dimensionality analysis for 5124 surface resolution..."
matlab -nodisplay -nosplash -r "S05_DimensionalityAnalysis($SUBJ_ID, $SURFACE_RES); exit"

# Check if 5124 completed successfully
if [ $? -eq 0 ]; then
    echo "Dimensionality analysis completed successfully for Subject $SUBJ_ID with 5124 surface resolution"
    
    # List output files for 5124
    echo "Output files created for 5124:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis/*5124*
else
    echo "Dimensionality analysis failed for Subject $SUBJ_ID with 5124 surface resolution"
fi

# Run dimensionality analysis for 8196 surface resolution
echo "Running dimensionality analysis for 8196 surface resolution..."
SURFACE_RES=8196
matlab -nodisplay -nosplash -r "S05_DimensionalityAnalysis($SUBJ_ID, $SURFACE_RES); exit"

# Check if 8196 completed successfully
if [ $? -eq 0 ]; then
    echo "Dimensionality analysis completed successfully for Subject $SUBJ_ID with 8196 surface resolution"
    
    # List output files for 8196
    echo "Output files created for 8196:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis/*8196*
else
    echo "Dimensionality analysis failed for Subject $SUBJ_ID with 8196 surface resolution"
fi

echo "All processing complete for subject $SUBJ_ID"
echo "End time: $(date)"
