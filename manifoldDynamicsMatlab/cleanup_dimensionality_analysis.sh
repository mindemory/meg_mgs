#!/bin/bash
#SBATCH --job-name=cleanup_dimensionality
#SBATCH --output=slurmOutput/cleanup_dimensionality_%j.out
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

# Script to clean up dimensionality analysis folders for all subjects
# This removes existing dimensionality_analysis directories to allow fresh analysis

echo "=== Cleaning up dimensionality analysis folders ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Define subjects array (same as in the analysis scripts)
subjects=(1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32)

# Set base path for HPC
base_path="/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives"

echo "Base path: $base_path"
echo "Subjects: ${subjects[*]}"
echo ""

# Counter for tracking
deleted_count=0
not_found_count=0

# Loop through each subject
for subj in "${subjects[@]}"; do
    # Format subject ID with zero padding
    subj_formatted=$(printf "sub-%02d" $subj)
    
    # Define the dimensionality analysis directory path
    dim_analysis_dir="$base_path/$subj_formatted/sourceRecon/dimensionality_analysis"
    
    echo "Processing $subj_formatted..."
    
    # Check if directory exists
    if [ -d "$dim_analysis_dir" ]; then
        echo "  Found dimensionality_analysis directory"
        echo "  Contents before deletion:"
        ls -la "$dim_analysis_dir" 2>/dev/null | head -5
        
        # Remove the entire directory and its contents
        rm -rf "$dim_analysis_dir"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully deleted $dim_analysis_dir"
            deleted_count=$((deleted_count + 1))
        else
            echo "  ✗ Failed to delete $dim_analysis_dir"
        fi
    else
        echo "  ✗ Directory not found: $dim_analysis_dir"
        not_found_count=$((not_found_count + 1))
    fi
    echo ""
done

# Summary
echo "=== Cleanup Summary ==="
echo "Total subjects processed: ${#subjects[@]}"
echo "Directories deleted: $deleted_count"
echo "Directories not found: $not_found_count"
echo ""

# Verify cleanup by checking a few subjects
echo "=== Verification ==="
echo "Checking a few subjects to verify cleanup:"
for subj in 1 2 3; do
    subj_formatted=$(printf "sub-%02d" $subj)
    dim_analysis_dir="$base_path/$subj_formatted/sourceRecon/dimensionality_analysis"
    if [ -d "$dim_analysis_dir" ]; then
        echo "  $subj_formatted: Directory still exists (cleanup may have failed)"
    else
        echo "  $subj_formatted: Directory successfully removed"
    fi
done

echo ""
echo "Cleanup completed at: $(date)"
echo "You can now run the dimensionality analysis fresh on HPC."
