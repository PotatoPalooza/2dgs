#!/bin/bash
#SBATCH -J initnet2-build         # Job name
#SBATCH -o initnet2_build_%j.out  # Standard output log
#SBATCH -e initnet2_build_%j.err  # Standard error log
#SBATCH -p gpu-a100-small         # Queue (partition)
#SBATCH -N 1                      # Nodes
#SBATCH -n 1                      # Tasks
#SBATCH -t 12:00:00               # Time limit
#SBATCH -A ASC25079               # Allocation / account

set -euo pipefail

module load cuda/12.4
module load anaconda/2023.09

PROJECT_ROOT="$WORK/2dgs/Instant-GI"
INSTALL_SCRIPT="$PROJECT_ROOT/install.sh"
CONDA_BASE="$(conda info --base)"
ENV_NAME="instant-gi-new"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "[SLURM] Conda env '$ENV_NAME' not found. Running install.sh to build it..."
    bash "$INSTALL_SCRIPT"
else
    echo "[SLURM] Found existing env '$ENV_NAME'. Skipping install."
fi

conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"

export WANDB_MODE=online

python train_init_net_2.py --config datasets/div2k_train_init_net.yaml
