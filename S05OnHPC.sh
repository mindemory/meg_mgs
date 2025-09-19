#!/bin/bash
#SBATCH --job-name=s05_tafkap
#SBATCH --output=slurmOutput/s05_tafkap_sub-%02a_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=END

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

echo "Starting TAFKAP decoding for Subject $SUBJ_ID with surface resolution $SURFACE_RES"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run the MATLAB script
matlab -nodisplay -nosplash -r "S05_TAFKAP_Decoding($SUBJ_ID, $SURFACE_RES); exit"

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "TAFKAP decoding completed successfully for Subject $SUBJ_ID"
    echo "End time: $(date)"
    
    # List output files
    echo "Output files created:"
    ls -la $OUTPUT_DIR/sub-$(printf "%02d" $SUBJ_ID)/sourceRecon/tafkap_decoding/
else
    echo "TAFKAP decoding failed for Subject $SUBJ_ID"
    echo "End time: $(date)"
    exit 1
fi
