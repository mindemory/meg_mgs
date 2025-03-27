#!/bin/bash
#SBATCH --job-name=TGAdecoding               # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=10                   # Reqest 20 CPU per job instance
#SBATCH --mem=50GB                           # Request 50GB of RAM per job instance
#SBATCH --time=04:30:00                      # Request 2 hours per job instance
#SBATCH --output=slurm%j.out
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

module purge
root_dir='/scratch/mdd9787/meg_prf_greene/megScripts'
cd $root_dir
chmod 755 activators/activate_conda.bash
activators/activate_conda.bash python runDecoding.py