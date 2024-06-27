


import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
from classifier_head import ESMWithClassifier
from custom_model3 import ESMWithCNN
from cnn_embedding import EmbeddingESMWithCNN
from sequence_dataset import SequenceDataset
from tokenized_dataset import TokenizedDataset
from sklearn.metrics import r2_score
import datetime

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
decoder_type = "cnn"
# decoder_type = "mlp"

###########################################################
import os
folder_path = f'runs/{timestamp}_test_{decoder_type}_{n_samples}samples_{prop_test}prop_test.png'
os.makedirs(folder_path, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')

# if decoder_type == "mlp":   
#     model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
# elif decoder_type == "transformer":
#     model = ESMWithTransformer(device=device, decoder_input_dim=1280).to(device)
# elif decoder_type == "embedding":
#     model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280, output_dim=1).to(device)
# elif decoder_type == "cnn_embedding":
#     model = EmbeddingESMWithCNN(device=device, decoder_input_dim=1280, output_dim=1).to(device)
# elif decoder_type == "cnn":
#     model = ESMWithCNN(device=device, input_dim=1280).to(device)
# else:
#     print("Unknown decoder type")

classif_model = ESMWithClassifier(device=device, input_dim=1280).to(device)
regress_model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)

print("decoder", decoder_type, "n_layers", n_layers_trained, "batch_size", batch_size, "n_epochs", n_epochs, "n_samples", n_samples, "prop_test", prop_test)


df_val = pd.read_csv('df_desai_new.csv')
df_val.drop(columns='Unnamed: 0')
print("Length of df_val", len(df_val))

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df_val["mutant_sequence"].tolist()
# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
targets = df_val['new_log10Kd_REGN10987'].values
# targets = df['log10Kd_ACE2'].values

# Split the dataset into training and validation sets
sequences_train, sequences_val, targets_train, targets_val = train_test_split(sequences, targets, test_size=prop_test, random_state=42)


# Training and validation datasets
def preprocess_sequence(seq):
    # This is a placeholder function to convert sequence strings to tensor.
    # Implement your specific sequence preprocessing here.
    # Example: Convert each character to an integer or one-hot encoded tensor.
    return torch.tensor([ord(c) - ord('A') for c in seq], dtype=torch.float32)  # Simplistic example
def postprocess_sequence(tensor):
    # This function reverses the preprocessing done in preprocess_sequence
    # Assuming tensor is a 1D tensor of float32 dtype, where each element represents an encoded character
    sequence = ''.join(chr(int(c) + ord('A')) for c in tensor)
    return sequence

preprocess_train = [preprocess_sequence(seq).to(device) for seq in sequences_train]  
preprocess_val = [preprocess_sequence(seq).to(device) for seq in sequences_val]  
train_dataset = SequenceDataset(preprocess_train, targets_train, device)
test_dataset = SequenceDataset(preprocess_val, targets_val, device)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Or load a model
# model_name = "esm_with_mlp_2layers_trained_2700train_samples_5epochs_4batch_size"
# model_name = "esm_with_mlp_4layers_trained_4500train_samples_4epochs_8batch_size"
# model_name = "esm_with_cnn_4layers_trained_1800train_samples_4epochs_4batch_size"
# model_name = "esm_with_mlp_4layers_trained_3600train_samples_3epochs_8batch_size"
# model_name = "20240504_1723_cnn_pearson_4layers_trained_9000train_samples_7epochs_8batch_size_0.0001decoder_lr_1e-06base_lr_log10Kd_REGN10987.png_model_save"
# model_name = "20240506_1422_mlp_pearson_1layers_trained_1000train_samples_3epochs_8batch_size_0.01decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"


classif_model_name = "20240510_1010_mlp+classifier_pearson_4layers_trained_2500train_samples_6epochs_8batch_size_0.0001decoder_lr_1e-05base_lr_log10Kd_REGN10987_classif_model_save"

print("Load the classif model", classif_model_name)
classif_model.load_state_dict(torch.load(f'models/{classif_model_name}.pt'))

regress_model_name = "20240510_1010_mlp+classifier_pearson_4layers_trained_2500train_samples_6epochs_8batch_size_0.0001decoder_lr_1e-05base_lr_log10Kd_REGN10987_regress_model_save"

print("Load the classif model", regress_model_name)
regress_model.load_state_dict(torch.load(f'models/{regress_model_name}.pt'))



# Loss function and optimizer
classif_criterion = nn.BCEWithLogitsLoss()
# classif_criterion = nn.MSELoss()
regress_criterion = nn.MSELoss()

classif_optimizer = torch.optim.Adam([
    {'params':[p for p in classif_model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': classif_model.decoder.parameters(), 'lr': decoder_lr}
])

regress_optimizer = torch.optim.Adam([
    {'params':[p for p in regress_model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': regress_model.decoder.parameters(), 'lr': decoder_lr}
])


# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np




def predict(classif_model, regress_model, sequences):
    classif_model.eval()
    regress_model.eval()
    with torch.no_grad():
        sequences = sequences.to(device)
        classif_outputs = torch.sigmoid(classif_model(sequences).squeeze())
        is_outlier = classif_outputs > 0.5
        regression_predictions = torch.full_like(classif_outputs, 5.0, dtype=torch.float)  # Initialize all predictions to 5 (outlier value)
        non_outlier_indices = ~is_outlier
        if non_outlier_indices.any():
            regression_predictions[non_outlier_indices] = regress_model(sequences[non_outlier_indices]).squeeze()
        return regression_predictions

def validate_model(model, test_loader, criterion):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(test_loader)
    return avg_val_loss

def plot_losses(training_losses, validation_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses[1:], label='Training Loss')
    plt.plot(validation_losses[1:], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path+'/training_validation_losses.png')
    plt.close()

# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import torch

def evaluate_model(classif_model, regress_model, test_loader, device):
    classif_model.eval()
    regress_model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            labels = labels.to(device)
            sequences = sequences.to(device)

            regression_predictions = predict(classif_model, regress_model, sequences)
            predictions.extend(regression_predictions.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    pearson_corr, _ = pearsonr(predictions, actuals) if actuals else (None, None)
    spearman_corr, _ = spearmanr(predictions, actuals) if actuals else (None, None)
    r_squared = r2_score(actuals, predictions) if actuals else None

    return predictions, actuals, pearson_corr, spearman_corr, r_squared


# Usage example with the above function
print("Calculating the correlations on the test set")
predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(classif_model, regress_model, test_loader, device)
print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
print("Spearman Correlation Coefficient on Test Set:", spearman_correlation)
print("R-squared on Test Set:", r_squared)


def plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, file_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actuals, y=predictions, alpha=0.6)
    plt.title('Predictions vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')

    # Adding Pearson and Spearman correlation, and R-squared on the plot
    plt.text(0.05, 0.95, f'Pearson Correlation: {pearson_corr:.2f}\nSpearman Correlation: {spearman_corr:.2f}\nR-squared: {r_squared:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Adding x=y line
    max_val = max(max(actuals), max(predictions))
    min_val = min(min(actuals), min(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # k-- is a black dashed line

    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
print("Correlation figure saved")

