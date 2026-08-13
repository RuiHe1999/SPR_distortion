#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure relative paths such as data/... use the submission directory
cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
source activate graph

echo "Node: $(hostname)"

python -u plot_figures.py

