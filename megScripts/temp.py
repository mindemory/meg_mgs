#!/usr/bin/env python3
"""
Temporary script to check SLURM output files and identify failed jobs
"""

import os
import re
import glob

# Directory containing SLURM output files
output_dir = 'slurmOutConnectivity'

if not os.path.exists(output_dir):
    print(f"Directory {output_dir} does not exist")
    exit(1)

# Get all SLURM output files
slurm_files = glob.glob(os.path.join(output_dir, 'slurm_*.out'))
slurm_files.sort()

print(f"Found {len(slurm_files)} SLURM output files\n")

failed_jobs = []
successful_jobs = []
incomplete_jobs = []
oom_killed_jobs = []
time_limit_jobs = []

# Patterns to look for
error_patterns = [
    r'error',
    r'Error',
    r'ERROR',
    r'Traceback',
    r'Exception',
    r'Failed',
    r'FAILED',
    r'exit code [1-9]',
    r'exit status [1-9]',
    r'oom_kill',
    r'OOM Killed',
    r'Killed'
]

success_patterns = [
    r'Completed connectivity processing',
    r'saved to',
    r'Connectivity analysis completed'
]

for slurm_file in slurm_files:
    # Extract job ID and task ID from filename (format: slurm_<jobID>_<taskID>.out)
    filename = os.path.basename(slurm_file)
    match = re.match(r'slurm_(\d+)_(\d+)\.out', filename)
    if match:
        job_id = match.group(1)
        task_id = match.group(2)
    else:
        # Try format without task ID: slurm_<jobID>.out
        match = re.match(r'slurm_(\d+)\.out', filename)
        if match:
            job_id = match.group(1)
            task_id = 'N/A'
        else:
            job_id = 'unknown'
            task_id = 'unknown'
    
    try:
        with open(slurm_file, 'r') as f:
            content = f.read()
            
        # Check for OOM kills specifically
        has_oom = re.search(r'oom_kill|OOM Killed', content, re.IGNORECASE) is not None
        
        # Check for time limit exceeded
        has_time_limit = re.search(r'TIME LIMIT|time limit|DUE TO TIME', content, re.IGNORECASE) is not None
        
        # Check for errors
        has_error = any(re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns)
        
        # Check for success indicators
        has_success = any(re.search(pattern, content, re.IGNORECASE) for pattern in success_patterns)
        
        # Check if file is empty or very short (might be incomplete)
        is_incomplete = len(content.strip()) < 100
        
        # Extract combination details from output
        subj_match = re.search(r'Subject: (\d+)', content)
        roi_match = re.search(r'ROI: (\w+)', content)
        target_match = re.search(r'Target: (\w+)', content)
        conn_match = re.search(r'Connectivity Type: (\w+)', content)
        
        subj = subj_match.group(1) if subj_match else 'unknown'
        roi = roi_match.group(1) if roi_match else 'unknown'
        target = target_match.group(1) if target_match else 'unknown'
        conn = conn_match.group(1) if conn_match else 'unknown'
        
        combo_info = f"subj{subj}_{roi}_{target}_{conn}"
        
        if has_oom:
            oom_killed_jobs.append((job_id, task_id, slurm_file, combo_info))
            failed_jobs.append((job_id, task_id, slurm_file, combo_info))
        elif has_time_limit:
            time_limit_jobs.append((job_id, task_id, slurm_file, combo_info))
            failed_jobs.append((job_id, task_id, slurm_file, combo_info))
        elif has_error:
            failed_jobs.append((job_id, task_id, slurm_file, combo_info))
        elif has_success and not has_error:
            successful_jobs.append((job_id, task_id, slurm_file, combo_info))
        elif is_incomplete:
            incomplete_jobs.append((job_id, task_id, slurm_file, combo_info))
        else:
            # Unknown status
            failed_jobs.append((job_id, task_id, slurm_file, combo_info))
            
    except Exception as e:
        print(f"Error reading {slurm_file}: {e}")
        failed_jobs.append((job_id, task_id, slurm_file, 'unknown'))

# Print summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Total jobs: {len(slurm_files)}")
print(f"Successful: {len(successful_jobs)}")
print(f"Failed: {len(failed_jobs)}")
print(f"  - OOM Killed: {len(oom_killed_jobs)}")
print(f"  - Time Limit Exceeded: {len(time_limit_jobs)}")
print(f"Incomplete: {len(incomplete_jobs)}")
print()

# Print OOM killed jobs
if oom_killed_jobs:
    print("="*70)
    print("OOM KILLED JOBS (Out of Memory):")
    print("="*70)
    for job_id, task_id, slurm_file, combo_info in oom_killed_jobs:
        print(f"Task {task_id}: {combo_info}")
    print()

# Print time limit exceeded jobs
if time_limit_jobs:
    print("="*70)
    print("TIME LIMIT EXCEEDED JOBS:")
    print("="*70)
    for job_id, task_id, slurm_file, combo_info in time_limit_jobs:
        print(f"Task {task_id}: {combo_info}")
    print()

# Print failed jobs (non-OOM, non-time-limit)
non_oom_time_failed = [f for f in failed_jobs if f not in oom_killed_jobs and f not in time_limit_jobs]
if non_oom_time_failed:
    print("="*70)
    print("FAILED JOBS (Other errors):")
    print("="*70)
    for job_info in non_oom_time_failed:
        if len(job_info) == 4:
            job_id, task_id, slurm_file, combo_info = job_info
            print(f"Task {task_id}: {combo_info} - {slurm_file}")
        else:
            job_id, task_id, slurm_file = job_info
            print(f"Job {job_id}, Task {task_id}: {slurm_file}")
    print()

# Print incomplete jobs
if incomplete_jobs:
    print("="*70)
    print("INCOMPLETE JOBS (empty or very short files):")
    print("="*70)
    for job_info in incomplete_jobs:
        if len(job_info) == 4:
            job_id, task_id, slurm_file, combo_info = job_info
            print(f"Task {task_id}: {combo_info} - {slurm_file}")
        else:
            job_id, task_id, slurm_file = job_info
            print(f"Job {job_id}, Task {task_id}: {slurm_file}")
    print()

# Print successful jobs count by task ID range
if successful_jobs:
    print("="*70)
    print(f"SUCCESSFUL JOBS: {len(successful_jobs)}")
    print("="*70)
    # Group by task ID ranges
    task_ids = []
    for job_info in successful_jobs:
        if len(job_info) >= 3:
            task_id = job_info[1]
            if task_id != 'N/A':
                try:
                    task_ids.append(int(task_id))
                except:
                    pass
    if task_ids:
        print(f"Task ID range: {min(task_ids)} - {max(task_ids)}")
        print(f"Unique task IDs: {len(set(task_ids))}")

