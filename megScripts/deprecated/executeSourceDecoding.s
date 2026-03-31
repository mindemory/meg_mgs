#!/bin/bash
#SBATCH --job-name=sourceDecoding              # The name of the job
#SBATCH --nodes=1                               # Request 1 compute node per job instance
#SBATCH --cpus-per-task=20                     # Request 20 CPU per job instance
#SBATCH --mem=32GB                            # Request 32GB of RAM per job instance
#SBATCH --time=01:00:00                        # Request 1 hour per job instance
#SBATCH --array=1-84                          # Array job with 84 tasks (21 subjects × 4 frequency bands)
#SBATCH --output=slurmOutDecoding/slurm_%A_%a.out             # Output file: %A = job ID, %a = array task ID
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

mkdir -p slurmOutDecoding
module purge
root_dir='/scratch/mdd9787/meg_prf_greene/megScripts/megScripts'
cd $root_dir
chmod 755 activators/activate_conda.bash

# Define subject list (21 subjects total)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Define frequency bands
freq_bands=(theta alpha beta lowgamma)

# Calculate subject and frequency band indices from array task ID
# Array task ID is 1-based, so subtract 1 for 0-based indexing
# Each subject has 4 frequency bands, so:
#   subject_index = (task_id - 1) / 4
#   freq_band_index = (task_id - 1) % 4
task_idx=$((SLURM_ARRAY_TASK_ID - 1))
n_freq_bands=${#freq_bands[@]}
n_subjects=${#subjects[@]}

subject_idx=$((task_idx / n_freq_bands))
freq_band_idx=$((task_idx % n_freq_bands))

subjID=${subjects[$subject_idx]}
freq_band=${freq_bands[$freq_band_idx]}

echo "=========================================="
echo "Starting processing:"
echo "  Subject: $subjID"
echo "  Frequency band: $freq_band"
echo "  Array task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "  (Subject index: $subject_idx, Frequency band index: $freq_band_idx)"
echo "=========================================="

# Run the source decoding for this subject and frequency band
activators/activate_conda.bash python inSourceSpaceDecodingWithBehav.py $subjID 8mm $freq_band

echo "=========================================="
echo "Completed processing for subject $subjID, frequency band $freq_band"
echo "=========================================="
