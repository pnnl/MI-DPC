#!/bin/bash
M=${1:-2}

# env=/home/bold914/miniconda3/envs/neuromancer/bin/python
env=/home/desktop309/miniconda3/envs/neuromancer/bin/python
# Fixed parameter
Ts=180

# Log file
LOGFILE="logs/MIDPC_training_M$M.log"

# Overwrite the log file at start
echo "===== Starting runs at $(date) =====" > "$LOGFILE"

# Loop over different nsteps values
for nsteps in 5 10 15; do
    echo "Running with Ts=$Ts, nsteps=$nsteps, and M=$M" | tee -a "$LOGFILE"
    $env -u MIDPC.py -Ts $Ts -nsteps $nsteps -M $M >> "$LOGFILE" 2>&1
    echo "Finished run with nsteps=$nsteps at $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------" >> "$LOGFILE"
done
# done
echo "===== All runs completed at $(date) =====" >> "$LOGFILE"