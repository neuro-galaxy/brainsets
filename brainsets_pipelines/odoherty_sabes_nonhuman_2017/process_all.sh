#!/bin/bash
#SBATCH --job-name=ods_process
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --array=0-46

# Process all O'Doherty-Sabes sessions with LFP band extraction.
#
# Usage:
#   cd <brainsets_repo>
#   pip install -e .
#   pip install mne zenodo_get
#   sbatch brainsets_pipelines/odoherty_sabes_nonhuman_2017/process_all.sh
#
# Raw .mat files are auto-downloaded from Zenodo if not already present.
# Broadband .nwb files must already exist for LFP extraction; sessions
# without a matching .nwb will still be processed (spikes + behavior only).

RAW_DIR="/network/projects/neuro-galaxy/data/raw"
BROADBAND_DIR="/network/projects/neuro-galaxy/data/raw/odoherty_sabes_nonhuman_2017/broadband"
PROCESSED_DIR="$SCRATCH/data/processed"
BRAINSETS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PIPELINE="$BRAINSETS_DIR/brainsets_pipelines/odoherty_sabes_nonhuman_2017/pipeline.py"

SESSIONS=(
    indy_20160407_02
    indy_20160411_01
    indy_20160411_02
    indy_20160418_01
    indy_20160419_01
    indy_20160420_01
    indy_20160426_01
    indy_20160622_01
    indy_20160624_03
    indy_20160627_01
    indy_20160630_01
    indy_20160915_01
    indy_20160916_01
    indy_20160921_01
    indy_20160927_04
    indy_20160927_06
    indy_20160930_02
    indy_20160930_05
    indy_20161005_06
    indy_20161006_02
    indy_20161007_02
    indy_20161011_03
    indy_20161013_03
    indy_20161014_04
    indy_20161017_02
    indy_20161024_03
    indy_20161025_04
    indy_20161026_03
    indy_20161027_03
    indy_20161206_02
    indy_20161207_02
    indy_20161212_02
    indy_20161220_02
    indy_20170123_02
    indy_20170124_01
    indy_20170127_03
    indy_20170131_02
    loco_20170210_03
    loco_20170213_02
    loco_20170214_02
    loco_20170215_02
    loco_20170216_02
    loco_20170217_02
    loco_20170227_04
    loco_20170228_02
    loco_20170301_05
    loco_20170302_02
)

SESSION=${SESSIONS[$SLURM_ARRAY_TASK_ID]}

module load miniconda/3
conda activate ng

echo "Processing $SESSION (task $SLURM_ARRAY_TASK_ID)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Brainsets dir: $BRAINSETS_DIR"

mkdir -p "$PROCESSED_DIR"

python -m brainsets.runner \
    "$PIPELINE" \
    --raw-dir="$RAW_DIR" \
    --processed-dir="$PROCESSED_DIR" \
    --single="$SESSION" \
    --broadband-dir="$BROADBAND_DIR" \
    --reprocess \
    -c 4

echo "Exit code: $?"
echo "End: $(date)"
