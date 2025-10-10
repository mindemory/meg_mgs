#!/bin/bash
#SBATCH --job-name=sourceDecoding              # The name of the job
#SBATCH --nodes=1                               # Request 1 compute node per job instance
#SBATCH --cpus-per-task=10                     # Request 10 CPU per job instance
#SBATCH --mem=100GB                            # Request 50GB of RAM per job instance
#SBATCH --time=08:00:00                        # Request 4 hours per job instance
#SBATCH --array=1-21                          # Array job with 21 tasks (one per subject)
#SBATCH --output=slurmOut/slurm_%A_%a.out             # Output file: %A = job ID, %a = array task ID
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

mkdir -p slurmOut
module purge
root_dir='/scratch/mdd9787/meg_prf_greene/megScripts/megScripts'
cd $root_dir
chmod 755 activators/activate_conda.bash

# Define subject list (21 subjects total)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Get the subject ID for this array task
# SLURM_ARRAY_TASK_ID is 1-based, so subtract 1 to get 0-based index
subjID=${subjects[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Starting processing for subject $subjID (Array task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT)"

# Run the source decoding for this subject
activators/activate_conda.bash python inSourceSpaceDecoding.py $subjID 5mm

echo "Completed processing for subject $subjID"
