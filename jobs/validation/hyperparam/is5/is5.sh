#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N is-5     
#$ -cwd
#$ -pe gpu-titanx 1
# -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -o my_stdout.log
#$ -e my_stderr.log
#$ -M ro6lm@outlook.com
#$ -m beas

# Initialise the environment modules
. /etc/profile.d/modules.sh

# close gap with qlogin
. /etc/bashrc

# Load conda environment
module load anaconda
source activate tyxe

# Run the program
cd /exports/eddie/scratch/s2110626/diss/others/TyXe/notebooks
python -u ./vcl-MF.py --init-scale 1e-55555
