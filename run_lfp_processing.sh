#!/bin/bash
# Script to process O'Doherty-Sabes sessions with LFP band extraction

# Source the profile to get module command
source /etc/profile.d/modules.sh 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Load miniconda and activate environment
module load miniconda/3
conda activate ng

PIPELINE="/home/mila/h/hee-woon.ryoo/projects/brainsets/brainsets_pipelines/odoherty_sabes_nonhuman_2017/pipeline.py"
RAW_DIR="/network/projects/neuro-galaxy/data/raw"
BROADBAND_DIR="/network/projects/neuro-galaxy/data/raw/odoherty_sabes_nonhuman_2017/broadband"

SESSIONS=("indy_20160916_01" "indy_20160921_01" "indy_20160927_04" "indy_20160927_06" "indy_20160930_02")

echo "Starting LFP processing at $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

for session in "${SESSIONS[@]}"; do
    echo ""
    echo "Processing $session at $(date)"
    python -m brainsets.runner \
        "$PIPELINE" \
        --raw-dir="$RAW_DIR" \
        --processed-dir="/network/scratch/h/hee-woon.ryoo/data/processed" \
        --single="$session" \
        --broadband-dir="$BROADBAND_DIR"
    echo "Finished $session at $(date)"
done

echo ""
echo "All processing complete at $(date)"
