#!/bin/bash
#SBATCH --job-name=s05_tafkap_pca_subjects
#SBATCH --output=slurmOutput/s05_tafkap_pca_sub-%02a_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
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

# Define subjects array (21 subjects)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get subject ID from array task ID
SUBJ_ID=${subjects[$((SLURM_ARRAY_TASK_ID - 1))]}
SURFACE_RES=8196

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_pca_decoding

echo "Starting TAFKAP PCA time series analysis for Subject $SUBJ_ID with surface resolution $SURFACE_RES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject ID: $SUBJ_ID"
echo "Surface resolution: $SURFACE_RES"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run TAFKAP PCA time series analysis using callerOfS05
echo "Running TAFKAP PCA time series analysis..."
matlab -nodisplay -nosplash -r "callerOfS05($SUBJ_ID, $SURFACE_RES, 'TAFKAP'); exit"

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "TAFKAP PCA time series analysis completed successfully for Subject $SUBJ_ID"
    
    # List output files
    echo "Output files created:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_pca_decoding/*timeseries*
else
    echo "TAFKAP PCA time series analysis failed for Subject $SUBJ_ID"
fi

echo "Processing complete for subject $SUBJ_ID"
echo "End time: $(date)"
