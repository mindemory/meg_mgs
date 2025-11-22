#!/bin/bash
#SBATCH --job-name=s05_tafkap_subjects
#SBATCH --output=slurmOutput/s05_tafkap_subj_%a_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20
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

# Define subject list (21 subjects)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get subject ID from array task ID
ARRAY_ID=$SLURM_ARRAY_TASK_ID
SUBJ_ID=${subjects[$((ARRAY_ID - 1))]}  # Convert array ID to actual subject ID
VOL_RES=8  # Volume resolution in mm
ALGORITHM="TAFKAP"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_decoding

echo "Starting TAFKAP decoding for Subject $SUBJ_ID with volume resolution ${VOL_RES}mm"
echo "Algorithm: $ALGORITHM"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run TAFKAP for this subject
echo "Processing subject $SUBJ_ID..."

# Run the MATLAB script with TAFKAP for this subject
matlab -nodisplay -nosplash -r "S05_TAFKAP_Decoding($SUBJ_ID, $VOL_RES, '$ALGORITHM'); exit"

# Check if this subject completed successfully
if [ $? -eq 0 ]; then
    echo "TAFKAP decoding completed successfully for subject $SUBJ_ID"
    
    # List output files for this subject
    echo "Output files created for subject $SUBJ_ID:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_decoding/*avgdelay*
else
    echo "TAFKAP decoding failed for subject $SUBJ_ID"
fi

echo "Processing complete for subject $SUBJ_ID"
echo "End time: $(date)"
