
# Project: ProxiClust

This repository contains Jupyter notebooks and cluster-ready Python code from three different research projects focused on improving data acquisition and predictive modeling for binding affinity prediction using state-of-the-art machine learning techniques. 

### Key Research Projects:

1. **Data Acquisition Strategies**: 
   - A comparison of different data acquisition strategies, including active learning techniques, to optimize the selection of training data.
   
2. **Combinatorial Strategies**:
   - Developed a clustering-based approach for selecting training sets for binding affinity predictive models. This approach helps in reducing the model training time while maintaining high predictive performance.

3. **ESM and ML**:
   - Leveraged state-of-the-art models such as ESM (Evolutionary Scale Modeling) to predict binding affinity from the embedding of amino acid sequences. These models are trained using embeddings generated from sequence data, providing highly accurate predictions for protein-ligand interactions.

---

# Environment Setup

To replicate the environment and run the code in this repository, activate the conda environment using the following command:

```bash
mamba activate myenv
```

Ensure you have all the necessary dependencies installed as listed in the `requirements.txt`.

---

# SLURM Job Submission

You can submit jobs to a SLURM cluster using various scripts provided in the repository. Below are examples of commands to run the training scripts with or without hyperparameters specified directly in the code.

### To run with hyperparameters directly set in the code:
```bash
sbatch finet_tune.sh
```

### To run with argparse for hyperparameters:
```bash
sbatch esm_and_ml/main_embeddings.sh --epochs 20 --samples 1000 --test_prop 0.9 --layers_trained 4 --batch_size 8 --decoder_type mlp --save_model --decoder_lr 0.000001 --base_lr 0.0000001
```

Other examples of running different models:
```bash
sbatch esm_and_ml/main_embeddings.sh --epochs 10 --samples 1000 --test_prop 0.9 --layers_trained 4 --batch_size 32 --decoder_type mlp --base_lr 0.00001 --decoder_lr 0.001 --save_model --antibody log10Kd_ACE2
```

```bash
sbatch main.sh --epochs 10 --samples 550 --test_prop 0.90 --layers_trained 0 --batch_size 4 --base_lr 0 --decoder_lr 0.00001 --decoder_type mlp --plot_loss
```

```bash
sbatch main.sh --epochs 20 --samples 550 --test_prop 0.90 --layers_trained 1 --batch_size 4 --base_lr 0.00001 --decoder_lr 0.00001 --decoder_type cnn --plot_loss --save_model
```

---

## Active Learning Commands

The repository also includes commands to implement active learning strategies. Below are examples of how to run these commands on SLURM:

```bash
sbatch main_embeddings_gp.sh --epochs 3 --samples 110 --test_prop 0.5 --layers_trained 1 --batch_size 8 --base_lr 0.00001 --decoder_lr 0.0001 --decoder_type mlp --plot_loss --save_model
```

```bash
sbatch main_embeddings.sh --epochs 5 --samples 1151 --test_prop 0.9565595 --layers_trained 1 --batch_size 8 --base_lr 0.00001 --decoder_lr 0.0001 --decoder_type mlp --plot_loss --save_model
```

```bash
sbatch main_embeddings_al.sh --epochs 5 --batch_size 8 --decoder_lr 0.0001 --decoder_type mlp
```

---

---
