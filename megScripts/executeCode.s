#!/bin/bash
#SBATCH --account=torch_pr_282_general
#SBATCH --job-name=PAC_Coupling                # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=10                   # Request 10 CPU per job instance
#SBATCH --mem=8GB                           # Request 50GB of RAM per job instance
#SBATCH --time=01:00:00                      # Request 4 hours per job instance
#SBATCH --output=slurm_PAC_%j.out
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

module purge
module load anaconda3/2025.06 
source activate ./penv 
root_dir='/scratch/mdd9787/meg_mgs'
cd $root_dir
python megScripts/plot_phase_power_coupling.py 1 8mm