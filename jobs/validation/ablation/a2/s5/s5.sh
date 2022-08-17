#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N BIU-a2-seed5     
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -o my_stdout.log
#$ -e my_stderr.log
#$ -M ro6lm@outlook.com
#$ -m beas
#$ -pe gpu-titanx 1

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda environment
module load anaconda
source activate tyxe

# Run the program
cd /home/s2110626/diss/TyXe/notebooks 
python -u ./vcl-MF.py --seed 5 --val 1 -a 2
