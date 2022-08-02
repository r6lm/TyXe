#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N is-2              
#$ -cwd
#$ -pe gpu-titanx 1                  
# -l h_rt=06:00:00 
#$ -l h_vmem=32G
#$ -o my_stdout.log
#$ -e my_stderr.log
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda environment
module load anaconda
source activate tyxe

# Run the program
cd /exports/eddie/scratch/s2110626/diss/others/TyXe/notebooks
python -u ./vlc-MF.py --init-scale 1e-2 --seed 6002
