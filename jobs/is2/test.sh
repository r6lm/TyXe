#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=01:00:00
#$ -N test
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU: 
#$ -pe gpu-titanx 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=16G
#$ -M ro6lm@outlook.com
#$ -m beas
 

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
#module load cuda/11.0.2 
module load anaconda
source activate tyxe

# Run the executable
python hello.py
