import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import torch
from plot_predictions_vs_actuals import plot_predictions_vs_actuals
from goppy import OnlineGP, SquaredExponentialKernel
import numpy as np
import random
from sequence_dataset import SequenceDataset
from sklearn.metrics import r2_score
import os
import datetime



# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}'

folder_path = f'runs/{file_names}'
# Check if the directory exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Set random seed for reproducibility
torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')



# dataset_path = 'df_bloom_complete.csv'
# dataset_path = 'df_desai_new.csv'
dataset_path = 'df_Desai_15loci_complete.csv'

print(dataset_path)
df = pd.read_csv(dataset_path)

df.drop(columns=['Unnamed: 0'])
sequences = df["mutant_sequence"].tolist()

antibody = 'log10Kd_ACE2'
print("antibody", antibody)
targets_original = df[antibody].values


def normalize_targets(targets):
    min_val = np.min(targets)
    max_val = np.max(targets)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
targets = normalize_targets(targets_original)

# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309'].values

esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
tokenizer = AutoTokenizer.from_pretrained('models')

def batch_process_embeddings(sequences, batch_size=4):
    # Create DataLoader
    dataset = sequences
    loader = DataLoader(dataset, batch_size=batch_size)

    # Collect all embeddings
    all_embeddings = []
    for batch in loader:
        input_ids, attention_mask = tokenizer(batch, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_hidden_states = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        pooled_hidden_states = sum_hidden_states / sum_mask
        all_embeddings.append(pooled_hidden_states.detach().cpu())


    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


# Tokenize the training and validation sequences

embeddings = batch_process_embeddings(sequences)
# embeddings = torch.load(f'embeddings_{dataset_path}.pt')
# targets = torch.load(f'targets_{dataset_path}.pt')


# save the embeddings
torch.save(embeddings, f'embeddings_{dataset_path}.pt')
torch.save(targets, f'targets_{dataset_path}.pt')
WILDTYPE = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

n_train = 201

epistasis_seq_ind = list(range(16384,16400))

df['diff'] = df['mutant_sequence'].apply(
    lambda x: [(WILDTYPE[i], i+331, x[i]) for i in range(len(x)) if x[i] != WILDTYPE[i]])

print("df diff of first one", df['diff'][24576])
labelled_idx = epistasis_seq_ind + [24576, 20480, 18432, 17408, 16896, 16640, 16512, 16448, 0, 16416, 16400]
labelled_idx = [el - 331 for el in labelled_idx]
unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]


# Calculate how many more remaining indexes are needed
n_needed = max(0, n_train - len(labelled_idx)) 
if n_needed > 0:
    additional_idx = list(random.sample(unlabelled_idx, n_needed))
    labelled_idx.extend(additional_idx)

unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

print("len labelled", len(labelled_idx))
print("len unlabelled", len(unlabelled_idx))
embeddings_train, embeddings_val, targets_train, targets_val = [embeddings[i] for i in labelled_idx], [embeddings[i] for i in unlabelled_idx], [targets[i] for i in labelled_idx], [targets[i] for i in unlabelled_idx]

print(f"{n_train} random")


# If choosing random samples
prop_test = 1 - n_train/len(embeddings)
embeddings_train, embeddings_val, targets_train, targets_val = train_test_split(embeddings, targets, test_size=prop_test, random_state=42)
print("len embeddings train", len(embeddings_train))


embeddings_train_np = np.array(embeddings_train)
embeddings_val_np = np.array(embeddings_val)
targets_train_np = np.expand_dims(targets_train, axis=1)
targets_val_np = np.expand_dims(targets_val, axis=1)


gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)


print('Create datasets and dataloaders')
# Training and validation datasets

train_dataset = SequenceDataset(embeddings_train, targets_train, device)
test_dataset = SequenceDataset(embeddings_val, targets_val, device)

# Training is not done in a train loop for the gp
print("Begin training the GP model")

gp.add(embeddings_train_np, targets_train_np)
print("Done training the GP model")
results = gp.predict(embeddings_val_np, what=("mean", "mse"))
predictions_np = results["mean"]
variance_np = results["mse"]


def evaluate_model(gp, unlabelled_seq, labels):
    pred = gp.predict(unlabelled_seq, what=("mean", "mse"))
    predictions_np = np.squeeze(pred["mean"])
    actuals_np = np.array(labels)
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predictions_np, actuals_np)
    # Spearman correlation
    spearman_corr, _ = spearmanr(predictions_np, actuals_np)
    # Compute R-squared
    r_squared = r2_score(actuals_np, predictions_np)
    return predictions_np, actuals_np, pearson_corr, spearman_corr, r_squared

# Usage example with the above function
predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(gp, embeddings_val_np, targets_val)
print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
print("Spearman Correlation Coefficient on Test Set:", spearman_correlation)
print("R-squared on Test Set:", r_squared)


plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
print("Correlation figure saved")

