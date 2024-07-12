#!/bin/bash

#SBATCH --job-name=combinatorial_strategies              # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of cores per node
#SBATCH --mem=64G                           # Memory per node
#SBATCH --time=12:00:00                     # Time limit
#SBATCH --output=./logs/combinatorial_strategies.%j.out  # Standard output log
#SBATCH --error=./logs/combinatorial_strategies.%j.err   # Standard error log
#SBATCH --partition=test                # Partition name, if applicable

# Loading the environment or modules necessary for running the script
# Assuming 'myenv' is the name of your Python environment
source ~/.bashrc
source activate myenv

echo "Starting the fine-tuning of the model on sequences"
echo "Using Python version:"
srun python --version

# Execute the Python script with command-line arguments passed to this script
srun python combinatorial_strategies.py
