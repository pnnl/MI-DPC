#!/bin/bash
env=/home/desktop309/git/.venv/bin/python
# Fixed parameter
Ts=180

# Log file
LOGFILE="logs/MIDPC_training.log"

# Overwrite the log file at start
echo "===== Starting runs at $(date) =====" > "$LOGFILE"

# Loop over different nsteps values
for nsteps in 10 20 30 40 50 60 70 80
# for nsteps in 30
do
    echo "Running with Ts=$Ts and nsteps=$nsteps" | tee -a "$LOGFILE"
    $env -u MIDPC.py -Ts $Ts -nsteps $nsteps >> "$LOGFILE" 2>&1
    echo "Finished run with nsteps=$nsteps at $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------" >> "$LOGFILE"
done

echo "===== All runs completed at $(date) =====" >> "$LOGFILE"