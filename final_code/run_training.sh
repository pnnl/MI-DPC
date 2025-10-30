#!/bin/bash
# nsteps=${1:-20}

env=/home/desktop309/git/.venv/bin/python
# Fixed parameter
Ts=180

# Log file

# LOGFILE="logs/MIDPC_training_N$nsteps.log"
LOGFILE="logs/MIDPC_training.log"
# Overwrite the log file at start
echo "===== Starting runs at $(date) =====" > "$LOGFILE"

# Loop over different nsteps values
for M in 2 3; do
# for nsteps in 20 40 60; do
for nsteps in 5 10 15; do
    echo "Running with Ts=$Ts, nsteps=$nsteps, and M=$M" | tee -a "$LOGFILE"
    $env -u MIDPC.py -Ts $Ts -nsteps $nsteps -M $M >> "$LOGFILE" 2>&1
    echo "Finished run with nsteps=$nsteps at $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------" >> "$LOGFILE"
done
done
echo "===== All runs completed at $(date) =====" >> "$LOGFILE"