#!/bin/bash
# Script to process 5 sessions with both v1 and v2 signal processing

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

# Process v1
echo ""
echo "========================================="
echo "Processing with v1 (original extract_bands)"
echo "========================================="
for session in "${SESSIONS[@]}"; do
    echo ""
    echo "Processing $session with v1 at $(date)"
    python -m brainsets.runner \
        "$PIPELINE" \
        --raw-dir="$RAW_DIR" \
        --processed-dir="/network/scratch/h/hee-woon.ryoo/data/processed_v1" \
        --single="$session" \
        --signal-version=v1 \
        --broadband-dir="$BROADBAND_DIR"
    echo "Finished $session with v1 at $(date)"
done

# Process v2
echo ""
echo "========================================="
echo "Processing with v2 (fixed extract_bands)"
echo "========================================="
for session in "${SESSIONS[@]}"; do
    echo ""
    echo "Processing $session with v2 at $(date)"
    python -m brainsets.runner \
        "$PIPELINE" \
        --raw-dir="$RAW_DIR" \
        --processed-dir="/network/scratch/h/hee-woon.ryoo/data/processed_v2" \
        --single="$session" \
        --signal-version=v2 \
        --broadband-dir="$BROADBAND_DIR"
    echo "Finished $session with v2 at $(date)"
done

echo ""
echo "All processing complete at $(date)"
