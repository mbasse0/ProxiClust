#!/bin/bash

#SBATCH --job-name=al_exp_outcome_lin_rf             # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=112               # Number of cores per node
#SBATCH --mem=256G                           # Memory per node
#SBATCH --time=06:00:00                     # Time limit
#SBATCH --output=./logs/al_exp_outcome_lin_rf.%j.out  # Standard output log
#SBATCH --error=./logs/al_exp_outcome_lin_rf.%j.err   # Standard error log
#SBATCH --partition=sapphire                # Partition name, if applicable

# Loading the environment or modules necessary for running the script
# Assuming 'myenv' is the name of your Python environment
source ~/.bashrc
source activate myenv

echo "Starting the fine-tuning of the model on sequences"
echo "Using Python version:"
srun python --version

# Execute the Python script with command-line arguments passed to this script
python data_acquisition/al_exp_outcome_lin_rf.py
