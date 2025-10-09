#!/bin/bash

### TC1 Job Script ###
 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=64G

### Specify number of core (CPU) to allocate to per node ###
#SBATCH --ntasks-per-node=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC1N07

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=60

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=sfiad_sanity
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

### Your script for computation ###
module load cuda/12.2
module load anaconda
source activate df-env

# Debug: Show the Python version and path
python --version
python -m site

# Run your script
python src/models/sfiad.py
