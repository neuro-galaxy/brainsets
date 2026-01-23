#!/bin/bash

# Submit SLURM jobs for processing O'Doherty-Sabes sessions

RAW_DIR="/network/projects/neuro-galaxy/data/raw"
BROADBAND_DIR="/network/projects/neuro-galaxy/data/raw/odoherty_sabes_nonhuman_2017/broadband"
PIPELINE_FILE="/home/mila/h/hee-woon.ryoo/projects/brainsets/brainsets_pipelines/odoherty_sabes_nonhuman_2017/pipeline.py"
PYTHON="/home/mila/h/hee-woon.ryoo/.conda/envs/ng/bin/python"

# Get all sessions from .mat files
SESSIONS=($(ls ${RAW_DIR}/odoherty_sabes_nonhuman_2017/*.mat | xargs -n1 basename | sed 's/.mat//'))

# Create log directory
mkdir -p $SCRATCH/logs

PROCESSED_DIR="$SCRATCH/data/processed"
mkdir -p ${PROCESSED_DIR}

for session in "${SESSIONS[@]}"; do
    JOB_NAME="ods_${session}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$SCRATCH/logs/${JOB_NAME}_%j.out
#SBATCH --error=$SCRATCH/logs/${JOB_NAME}_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long

# Set PATH for conda env binaries
export PATH="/home/mila/h/hee-woon.ryoo/.conda/envs/ng/bin:\$PATH"

echo "Processing ${session}"
echo "Start time: \$(date)"
echo "Node: \$(hostname)"

${PYTHON} -m brainsets.runner \\
    ${PIPELINE_FILE} \\
    --raw-dir=${RAW_DIR} \\
    --processed-dir=${PROCESSED_DIR} \\
    --single=${session} \\
    --broadband-dir=${BROADBAND_DIR} \\
    --reprocess \\
    -c 4

echo "End time: \$(date)"
echo "Exit code: \$?"
EOF

    echo "Submitted ${JOB_NAME}"
done

echo "All jobs submitted!"
