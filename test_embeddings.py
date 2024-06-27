


import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset

from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr,spearmanr
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from custom_model import ESMWithMLP
from custom_model_2 import ESMWithTransformer
from embedding_model import EmbeddingESMWithMLP
from custom_model3 import ESMWithCNN
from cnn_embedding import EmbeddingESMWithCNN
from plot_predictions_vs_actuals import plot_predictions_vs_actuals
from sequence_dataset import SequenceDataset
from tokenized_dataset import TokenizedDataset
from sklearn.metrics import r2_score
import datetime
import numpy as np
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M")

# Set random seed for reproducibility
torch.manual_seed(42)


#################################################################
# Hyperparams

n_epochs = 1
n_samples = 32000
prop_test = 0.95

n_layers_trained = 0
batch_size = 4

base_lr = 1e-5
decoder_lr = 1e-5

# decoder_type = "transformer"
# decoder_type = "cnn"
# decoder_type = "mlp"
decoder_type = "embeddings"
###########################################################
import os
folder_path = f'runs/{timestamp}_test_{decoder_type}_{n_samples}samples_{prop_test}prop_test.png'
os.makedirs(folder_path, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')

if decoder_type == "mlp":   
    model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "transformer":
    model = ESMWithTransformer(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "embedding":
    model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280, output_dim=1).to(device)
elif decoder_type == "cnn_embedding":
    model = EmbeddingESMWithCNN(device=device, decoder_input_dim=1280, output_dim=1).to(device)
elif decoder_type == "cnn":
    model = ESMWithCNN(device=device, input_dim=1280).to(device)
elif decoder_type == "embeddings":
    model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280, output_dim=1).to(device)
else:
    print("Unknown decoder type")


print("decoder", decoder_type, "n_layers", n_layers_trained, "batch_size", batch_size, "n_epochs", n_epochs, "n_samples", n_samples, "prop_test", prop_test)


df_val = pd.read_csv('df_desai_new.csv')
df_ref = pd.read_csv('df_Desai_15loci_complete.csv')

# Select n_samples randomly in the df
# df_val = df_val.sample(n=200, random_state=42)
df_val.drop(columns='Unnamed: 0')
print("length of Desai new", len(df_val))
geno_array = np.array(df_val['geno'].astype(str))
# Pad each string in the array to make sure it is 11 characters long
geno_array_padded = np.vectorize(lambda x: x.zfill(11))(geno_array)

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df_val["mutant_sequence"].tolist()
# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
targets_original = df_val['new_log10Kd_REGN10987'].values
targets_ref = df_ref['log10Kd_REGN10987'].values
# Normalize function
def normalize_targets(targets, ref_targets):
    min_val = np.min(ref_targets)
    print("min val in Desai old", min_val)
    max_val = np.max(ref_targets)
    print("max val in Desai old", max_val)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
targets = normalize_targets(targets_original, targets_ref)

# Split the dataset into training and validation sets
sequences_train, sequences_val, targets_train, targets_val = train_test_split(sequences, targets, test_size=prop_test, random_state=42)

esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
    
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
tokenizer = AutoTokenizer.from_pretrained('models')

embeddings_train = batch_process_embeddings(sequences_train)
embeddings_val = batch_process_embeddings(sequences_val)

# Save the embeddings
torch.save(embeddings_train, f'new_embeddings_train_{n_samples}.pt')
torch.save(embeddings_val, f'new_embeddings_val_{n_samples}.pt')

# Load the embeddings
# embeddings_train = torch.load(f'new_embeddings_train_{n_samples}.pt')
# embeddings_val = torch.load(f'new_embeddings_val_{n_samples}.pt')

print('Create datasets and dataloaders')
# Training and validation datasets

train_dataset = SequenceDataset(embeddings_train, targets_train, device)
test_dataset = SequenceDataset(embeddings_val, targets_val, device)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.MSELoss()

# Or load a model
# model_name = "20240508_1548_cnn_pearson_1layers_trained_1600train_samples_6epochs_8batch_size_0.0001decoder_lr_0.0001base_lr_log10Kd_REGN10987_model_save"
# model_name = "20240509_1627_cnn_pearson_4layers_trained_3200train_samples_3epochs_8batch_size_1e-05decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"
model_name = "20240517_0919_mlp_pearson_1layers_trained_7200train_samples_10epochs_4batch_size_0.001decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"

print("Load the model", model_name)
model.load_state_dict(torch.load(f'models/{model_name}.pt'))

# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np

# Test the model with Pearson correlation
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            labels = labels.to(device)
            sequences = sequences.to(device)


            outputs = model(sequences).squeeze()
            
            # Check if the output tensor is 0-dimensional
            if outputs.dim() == 0:
                predictions.append(outputs.item())  # Convert scalar tensor to Python float
                print("output zero dimensional")
            else:
                predictions.extend(outputs.cpu().numpy())
                
            actuals.extend(labels.cpu().numpy())
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predictions, actuals)
    # Spearman correlation
    spearman_corr, _ = spearmanr(predictions, actuals)
    # Compute R-squared
    r_squared = r2_score(actuals, predictions)
    return predictions, actuals, pearson_corr, spearman_corr, r_squared


# Usage example with the above function
predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(model, test_loader, device)
print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
print("Spearman Correlation Coefficient on Test Set:", spearman_correlation)
print("R-squared on Test Set:", r_squared)


# plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+f'/{model_name}_predictions_on_Desai_new.png')
print("Correlation figure saved")

