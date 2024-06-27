#!/bin/bash

#SBATCH --job-name=esm-model-download       # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of cores per node
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:5:00                     # Time limit hrs:min:sec
#SBATCH --output=./logs/esm-model-download.%j.out  # Standard output and error log
#SBATCH --error=./logs/esm-model-download.%j.err   # Error log
#SBATCH --partition=gpu_test                  # Partition name, if applicable


# Install necessary Python packages if not already available in the environment
# module load Mambaforge/23.3.1-fasrc01

source ~/.bashrc
source activate myenv

echo "Using Python version:"
srun python --version
srun python explain_esm.py
