#!/bin/bash
# Control strategy to use in simulation options [MIMPC, MIDPC, RBC]
POLICY=${1:-"MIDPC"}
n_days=7
# Python env
env=/home/desktop309/git/.venv/bin/python
# Sampling time
Ts=180

# Log file
LOGFILE="logs/${POLICY}_simulation.log"

# Overwrite the log file at start
echo "===== Starting runs at $(date) =====" > "$LOGFILE"

# Loop over different nsteps values
# for nsteps in 10
if [ "$POLICY" = "RBC" ]; then
  nsteps_vals=(20)
else
  # nsteps_vals=(20 40 60)
  nsteps_vals=(5 10 15)
fi

# for M in 2 3; do
for M in 2; do
    for nsteps in  "${nsteps_vals[@]}"; do
        echo "Running $POLICY with Ts=$Ts, nsteps=$nsteps, and M=$M" | tee -a "$LOGFILE"
        $env -u simulate_chiller.py -Ts $Ts -nsteps $nsteps -policy $POLICY -M $M -n_days $n_days >> "$LOGFILE" 2>&1
        echo "Finished run with nsteps=$nsteps at $(date)" | tee -a "$LOGFILE"
        echo "-----------------------------------" >> "$LOGFILE"
    done
done
echo "===== All runs completed at $(date) =====" >> "$LOGFILE"