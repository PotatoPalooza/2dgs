#!/bin/bash
#SBATCH -J initnet2           # Job name
#SBATCH -o initnet2_%j.out       # Name of stdout output file
#SBATCH -e initnet2_%j.err          # stderr
#SBATCH -p gpu-a100-small          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1               # Total # of mpi tas
#SBATCH -t 12:00:00        # Run time (hh:mm:ss)
#SBATCH -A ASC25079       # Allocation name (req'd if you have more than 1)

# Setup env
module load cuda/12.0
source $SCRATCH/miniforge3/etc/profile.d/conda.sh
conda activate instant-gi


cd $WORK/2dgs/Instant-GI

export WANDB_MODE=online

# Run training
python train_init_net_2.py --config datasets/div2k_train_init_net.yaml
