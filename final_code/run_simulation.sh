#!/bin/bash
# Control strategy to use in simulation options [MIMPC, MIDPC, RBC]
POLICY="MIMPC"

# Python env
env=/home/desktop309/git/.venv/bin/python
# Sampling time
Ts=180

# Log file
LOGFILE="logs/${POLICY}_simulation.log"

# Overwrite the log file at start
echo "===== Starting runs at $(date) =====" > "$LOGFILE"

# Loop over different nsteps values
# for nsteps in 10 20 30 40 50 60 70 80
for nsteps in 20
do
    echo "Running with Ts=$Ts and nsteps=$nsteps" | tee -a "$LOGFILE"
    $env -u simulate_chiller.py -Ts $Ts -nsteps $nsteps -policy $POLICY >> "$LOGFILE" 2>&1
    echo "Finished run with nsteps=$nsteps at $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------" >> "$LOGFILE"
done

echo "===== All runs completed at $(date) =====" >> "$LOGFILE"