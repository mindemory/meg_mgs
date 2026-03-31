#!/bin/bash
#SBATCH --job-name=sourceSeededConnectivity            # The name of the job
#SBATCH --nodes=1                               # Request 1 compute node per job instance
#SBATCH --cpus-per-task=20                     # Request 20 CPU per job instance
#SBATCH --mem=32GB                            # Request 32GB of RAM per job instance
#SBATCH --time=24:00:00                        # Request 12 hours per job instance
#SBATCH --array=1-672                          # Array job: 21 subjects × 4 rois × 2 targets × 1 connectivityTypes × 4 frequency bands = 672 tasks
#SBATCH --output=slurmOutConnectivity/slurm_%A_%a.out             # Output file: %A = job ID, %a = array task ID
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

mkdir -p slurmOutConnectivity
module purge
root_dir='/scratch/mdd9787/meg_prf_greene/megScripts/megScripts'
cd $root_dir
chmod 755 activators/activate_conda.bash

# Define arrays
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)
rois=(left_visual right_visual left_frontal right_frontal)
targets=(left right)
connectivityTypes=(imcoh)
freqBands=(theta alpha beta lowgamma)

# Calculate dimensions
n_subjects=${#subjects[@]}
n_rois=${#rois[@]}
n_targets=${#targets[@]}
n_connectivityTypes=${#connectivityTypes[@]}
n_freqBands=${#freqBands[@]}

# Convert 1-based array task ID to 0-based index
task_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate indices for each dimension
# Order: subjects, rois, targets, connectivityTypes, freqBands
# Formula: task_idx = subj_idx * (n_rois * n_targets * n_connectivityTypes * n_freqBands) + 
#                    roi_idx * (n_targets * n_connectivityTypes * n_freqBands) + 
#                    target_idx * (n_connectivityTypes * n_freqBands) + 
#                    connectivityType_idx * n_freqBands + 
#                    freqBand_idx

roi_target_conn_freq_size=$((n_targets * n_connectivityTypes * n_freqBands))
target_conn_size=$((n_connectivityTypes * n_freqBands))
connectivityType_freq_size=$n_freqBands

subj_idx=$((task_idx / (n_rois * roi_target_conn_freq_size)))
remainder=$((task_idx % (n_rois * roi_target_conn_freq_size)))

roi_idx=$((remainder / roi_target_conn_freq_size))
remainder=$((remainder % roi_target_conn_freq_size))

target_idx=$((remainder / target_conn_size))
remainder=$((remainder % target_conn_size))

connectivityType_idx=$((remainder / connectivityType_freq_size))
freqBand_idx=$((remainder % connectivityType_freq_size))

# Get values
subjID=${subjects[$subj_idx]}
roi=${rois[$roi_idx]}
target=${targets[$target_idx]}
connectivityType=${connectivityTypes[$connectivityType_idx]}
freqBand=${freqBands[$freqBand_idx]}
echo "Starting connectivity processing (Task $SLURM_ARRAY_TASK_ID of 672)"
echo "Subject: $subjID, ROI: $roi, Target: $target, Connectivity Type: $connectivityType, Frequency Band: $freqBand"

# Run the source seeded connectivity
activators/activate_conda.bash python inSourceSpaceSeededConnectivity.py $subjID 10mm $roi $target $connectivityType $freqBand

echo "Completed connectivity processing for subject $subjID with ROI $roi, target $target, connectivity type $connectivityType, and frequency band $freqBand"
