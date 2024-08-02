#!/bin/bash

#SBATCH --job-name=perf_by_size              # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=112   # Number of cores per node
#SBATCH --mem=128G                           # Memory per node
#SBATCH --time=12:00:00                     # Time limit
#SBATCH --output=./logs/perf_by_size.%j.out  # Standard output log
#SBATCH --error=./logs/perf_by_size.%j.err   # Standard error log
#SBATCH --partition=test                # Partition name, if applicable

# Loading the environment or modules necessary for running the script
# Assuming 'myenv' is the name of your Python environment
source ~/.bashrc
source activate myenv

echo "Starting the fine-tuning of the model on sequences"
echo "Using Python version:"
srun python --version

# Execute the Python script with command-line arguments passed to this script
python perf_by_size.py
