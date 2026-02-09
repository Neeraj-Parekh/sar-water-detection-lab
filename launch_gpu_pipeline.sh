#!/bin/bash

# GPU Pipeline Launcher Script
# ============================
# This script handles cleaning previous runs, activating the environment,
# and launching the GPU equation search in the background.

# Log setup
LOG_FILE="pipeline.log"
echo "Starting launch sequence at $(date)" > setup_launch.log

# Kill old process if running
echo "Killing old processes..." >> setup_launch.log
pkill -f gpu_equation_search.py
sleep 2

# Cleanup results
echo "Cleaning results directory..." >> setup_launch.log
cd ~/sar_water_detection || exit 1
rm -rf results
mkdir -p results

# Activate Environment (Robust Method)
echo "Activating conda environment..." >> setup_launch.log
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_env

# Check python and GPU
echo "Python path: $(which python3)" >> setup_launch.log
# launch verification in background
nohup python3 gpu_equation_search.py \
    --chip-dir ./chips \
    --output-dir ./results \
    --regimes large_lake wide_river narrow_river wetland arid reservoir urban_flood \
    --max-candidates 500 \
    --extract-rules > "$LOG_FILE" 2>&1 &

PID=$!
echo "Pipeline launched with PID: $PID" >> setup_launch.log
echo "Logs redirected to $LOG_FILE" >> setup_launch.log

# Ensure it doesn't die when ssh closes
disown $PID
echo "Process disowned. Exiting." >> setup_launch.log
