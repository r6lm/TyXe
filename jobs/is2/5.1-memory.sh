#!/bin/sh
#
#                                                                             #
###############################################################################
 
# Grid Engine options
#$ -N memory
#$ -cwd
#$ -l h_vmem=8G
#$ -pe gpu-titanx 1
#$ -l h_rt=00:02:00
 
# Initialise the modules framework and load required modules
. /etc/profile.d/modules.sh
 
# Preamble
echo '========================================================================'
echo $(ulimit -v) KB virtual memory is usable.
echo Stack size is $(ulimit -s)
echo '========================================================================'
 
# Run the program
echo '========================================================================'
./memorycheck
echo '========================================================================'
