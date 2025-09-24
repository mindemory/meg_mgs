#!/bin/bash
#SBATCH --job-name=s05_dimensionality
#SBATCH --output=slurmOutput/s05_dimensionality_sub-%02a_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-20

# Load MATLAB module
module load matlab/2024a

# Set up paths
PROJECT_DIR="/scratch/mdd9787/meg_prf_greene"
MEG_SCRIPTS_DIR="$PROJECT_DIR/megScripts"
OUTPUT_DIR="$PROJECT_DIR/MEG_HPC/derivatives"

# Change to the scripts directory
cd $MEG_SCRIPTS_DIR

# Define subjects array (21 subjects)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get subject ID from array task ID
SUBJ_ID=${subjects[$SLURM_ARRAY_TASK_ID]}
SURFACE_RES=5124

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis

echo "Starting dimensionality analysis for Subject $SUBJ_ID with surface resolution $SURFACE_RES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject ID: $SUBJ_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Initialize success flags
SUCCESS_5124=false
SUCCESS_8196=false

# Run dimensionality analysis for 5124 surface resolution
echo "Running dimensionality analysis for 5124 surface resolution..."
matlab -nodisplay -nosplash -r "S05_DimensionalityAnalysis($SUBJ_ID, $SURFACE_RES); exit"
EXIT_CODE_5124=$?

# Check if 5124 completed successfully
if [ $EXIT_CODE_5124 -eq 0 ]; then
    echo "✓ Dimensionality analysis completed successfully for Subject $SUBJ_ID with 5124 surface resolution"
    SUCCESS_5124=true
    
    # List output files for 5124
    echo "Output files created for 5124:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis/*5124*
else
    echo "✗ Dimensionality analysis failed for Subject $SUBJ_ID with 5124 surface resolution (exit code: $EXIT_CODE_5124)"
fi

echo ""
echo "=========================================="
echo ""

# Run dimensionality analysis for 8196 surface resolution
echo "Running dimensionality analysis for 8196 surface resolution..."
SURFACE_RES=8196
matlab -nodisplay -nosplash -r "S05_DimensionalityAnalysis($SUBJ_ID, $SURFACE_RES); exit"
EXIT_CODE_8196=$?

# Check if 8196 completed successfully
if [ $EXIT_CODE_8196 -eq 0 ]; then
    echo "✓ Dimensionality analysis completed successfully for Subject $SUBJ_ID with 8196 surface resolution"
    SUCCESS_8196=true
    
    # List output files for 8196
    echo "Output files created for 8196:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/dimensionality_analysis/*8196*
else
    echo "✗ Dimensionality analysis failed for Subject $SUBJ_ID with 8196 surface resolution (exit code: $EXIT_CODE_8196)"
fi

echo ""
echo "=========================================="
echo "FINAL SUMMARY for Subject $SUBJ_ID:"
echo "5124 vertices: $([ "$SUCCESS_5124" = true ] && echo "SUCCESS" || echo "FAILED")"
echo "8196 vertices: $([ "$SUCCESS_8196" = true ] && echo "SUCCESS" || echo "FAILED")"

# Determine overall success
if [ "$SUCCESS_5124" = true ] && [ "$SUCCESS_8196" = true ]; then
    echo "Overall status: ALL SUCCESSFUL"
    FINAL_EXIT_CODE=0
elif [ "$SUCCESS_5124" = true ] || [ "$SUCCESS_8196" = true ]; then
    echo "Overall status: PARTIAL SUCCESS"
    FINAL_EXIT_CODE=1
else
    echo "Overall status: ALL FAILED"
    FINAL_EXIT_CODE=2
fi

echo "All processing complete for subject $SUBJ_ID"
echo "End time: $(date)"
echo "Final exit code: $FINAL_EXIT_CODE"

exit $FINAL_EXIT_CODE
