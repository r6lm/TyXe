#!/bin/bash

# this will run a R job 5 times and save output to a specified file per run
# there are many ways to specify loops, read the Bash (or your shell's) documentation!
for RUN in $(seq 1 5); do
    cd s$RUN
    qsub s$RUN.sh
    cd ..
done