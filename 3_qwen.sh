#!/bin/bash
#SBATCH --job-name=qwen
#SBATCH --partition=high          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1 
#SBATCH --mem=50G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
source activate graph

echo "Node: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi

printf "\nExtract features\n"
python -u analysis_qwen_ols.py

printf "\nSurprisal decay\n"
python -u analysis_qwen_surprisal_decay.py

printf "\nCosine decay\n"
python -u analysis_qwen_cosine_decay.py

printf "\nFinished: %s\n" "$(date)"
