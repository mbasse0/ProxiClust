

# Environment

mamba activate myenv


# Slurm


# To run with hyperparameters directly set in the code
sbatch finet_tune.sh

# To argparse the hyperparameters

sbatch main.sh --epochs 10 --samples 550 --test_prop 0.90 --layers_trained 0 --batch_size 4 --base_lr 0 --decoder_lr 0.00001 --decoder_type mlp --plot_loss


sbatch main.sh --epochs 20 --samples 550 --test_prop 0.90 --layers_trained 1 --batch_size 4 --base_lr 0.00001 -
-decoder_lr 0.00001 --decoder_type cnn --plot_loss --save_model


## Active learning commands

sbatch main_embeddings_gp.sh --epochs 3 --samples 110 --test_prop 0.5 --layers_trained 1 --batch_size 8 --base_lr 0.00001 --decoder_lr 0.0001 --decoder_type mlp --plot_loss --save_model

sbatch main_embeddings.sh --epochs 5 --samples 1151 --test_prop 0.9565595 --layers_trained 1 --batch_size 8 --base_lr 0.00001 --decoder_lr 0.0001 --decoder_type mlp --plot_loss --save_model

sbatch main_embeddings_al.sh --epochs 5 --batch_size 8 --decoder_lr 0.0001 --decoder_type mlp

## Jupyter on compute nodes

salloc --partition gpu_test --gpus=1 --time 60 --mem 4000
jupyter-notebook --no-browser --port=$myport --ip='0.0.0.0'

# Then copy the url into the kernel of the ipynb file on VSCode 

