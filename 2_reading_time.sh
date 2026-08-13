#!/bin/bash
#SBATCH --job-name=rt
#SBATCH --partition=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure relative paths such as data/... use the submission directory
cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
source activate graph

echo "Node: $(hostname)"

python -u analysis_residual_mixedlm.py
python -u analysis_residual_decay.py

