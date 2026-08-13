#!/bin/bash
#SBATCH --job-name=process
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure relative paths such as data/... use the submission directory
cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
source activate graph

echo "Node: $(hostname)"

python -u deidentify_data.py
python -u organize_data.py

