import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from custom_model import ESMWithMLP
from custom_model_2 import ESMWithTransformer
from custom_model3 import ESMWithCNN
from custom_classification_model import ESMWithDualHead
from mlp_classifier_model import ESMWithDualHead
from plot_predictions_vs_actuals import plot_predictions_vs_actuals

from sequence_dataset import SequenceDataset
from sklearn.metrics import r2_score
import os
import datetime



def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a model on sequence data.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--samples', type=int, default=350, help='Number of samples to use from the dataset.')
    parser.add_argument('--test_prop', type=float, default=6/7, help='Proportion of the dataset to use for testing.')
    parser.add_argument('--layers_trained', type=int, default=2, help='Number of ESM layers to train.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and evaluation.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate for the optimizer.')
    parser.add_argument('--decoder_lr', type=float, default=1e-3, help='Learning rate for the MLP decoder.')
    parser.add_argument('--decoder_type', type=str, default='transformer', choices=['mlp', 'transformer', 'cnn', 'cnn+classifier', 'mlp+classifier'], help='Type of decoder to use.')
    parser.add_argument('--plot_loss', action='store_true', help='Enable plotting of training and validation loss')
    parser.add_argument('--save_model', action='store_true', help='Enable saving of trained_model parameters')
    parser.add_argument('--antibody', type=str, default='log10Kd_REGN10987', help='Which antibody binding affinity to predict')
    parser.add_argument('--escape_weight', type=float, default=3, help='Weight of the classification part of the loss vs. the regression part')
    args = parser.parse_args()
    return args

args = parse_args()


n_epochs = args.epochs
n_samples = args.samples
prop_test = args.test_prop
n_layers_trained = args.layers_trained
batch_size = args.batch_size
base_lr = args.base_lr
decoder_lr = args.decoder_lr
decoder_type = args.decoder_type
plot_loss = args.plot_loss
save_model = args.save_model
antibody = args.antibody
escape_weight = args.escape_weight

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M")
file_names = f'{timestamp}_{decoder_type}_pearson_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size_{decoder_lr}decoder_lr_{base_lr}base_lr_{antibody}'

folder_path = f'runs/{file_names}'
# Check if the directory exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Set random seed for reproducibility
torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')

if decoder_type == "mlp":   
    model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "transformer":
    model = ESMWithTransformer(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "cnn":
    model = ESMWithCNN(device=device, input_dim=1280).to(device)
elif decoder_type == "cnn+classifier":
    model = ESMWithDualHead(device=device, input_dim=1280).to(device)
elif decoder_type == "mlp+classifier":
    model = ESMWithDualHead(device=device, input_dim=1280).to(device)
elif decoder_type == "custom_loss":
    model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
else:
    print("Unknown decoder type")


print(folder_path)

print('Load the Desai dataset')
# Load the dataset
df = pd.read_csv('df_Desai_15loci_complete.csv')
# df = pd.read_csv('df_desai_new.csv')
# TEMPORARY crop the dataset for faster runtime
df = df.sample(n=n_samples, random_state=42)
#save the cropped dataset
df.to_csv(f'df_cropped_{n_samples}.csv', index=False)
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()

# targets = df['log10Kd_ACE2'].values
targets_original = df['log10Kd_REGN10987'].values

import numpy as np

# Normalize function
def normalize_targets(targets):
    min_val = np.min(targets)
    max_val = np.max(targets)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
targets = normalize_targets(targets_original)

# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309'].values
# print("target for main", targets)

# Split the dataset into training and validation sets
sequences_train, sequences_val, targets_train, targets_val = train_test_split(sequences, targets, test_size=prop_test, random_state=42)


# Quick plot of the target values

# Plot the first 50 affinity values from the training set as a 1D scatter plot on the x-axis
plt.figure(figsize=(10, 2))  # Narrow figure height as we only need one horizontal line
y_zeros = [0] * 50  # Create a list of zeros for the y-axis values
plt.scatter(targets_train[:50], y_zeros, marker='o', color='b')  # Use scatter to plot points on the x-axis
plt.title('Affinity Values Distribution of the First 50 Sequences in Training Set')
plt.xlabel('log10Kd_ACE2')
plt.yticks([])  # Hide y-axis ticks as they are not needed
plt.grid(True, axis='x')  # Grid only along x-axis
plt.savefig(folder_path+'/training_set_affinities_1D_horizontal.png')  # Save the plot to a file


# # Tokenize the training and validation sequences
# tokenizer = AutoTokenizer.from_pretrained('models')
# sequences_train_tokens = tokenizer(sequences_train, return_tensors='pt', padding='longest', truncation=True, max_length=512)
# sequences_val_tokens = tokenizer(sequences_val, return_tensors='pt', padding='longest', truncation=True, max_length=512)


def freeze_esm_layers(model, n_layers_to_freeze):
    for name, param in model.named_parameters():
        # Freeze parameters only in the transformer part of the ESM
        if 'esm.encoder.layer' in name:
            # print("Freezing", name)
            layer_num = int(name.split('.')[3])
            if layer_num >= n_layers_to_freeze:
                param.requires_grad = False
        # else:
            # print("Not freezing because not in a transformer layer", name)
# Choose which parameters to freeze in ESM
freeze_esm_layers(model.esm, n_layers_trained)  # Freeze all but the last `n_layers_trained` layers

print('Create datasets and dataloaders')
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

threshold_escape = -0.5
escape_set_value = -1
import torch.nn.functional as F
# Loss function and optimizer
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # Mask for selecting elements where target > 7
        mask_high = targets > threshold_escape
        # Mask for selecting elements where target < 7
        mask_low = targets < threshold_escape
        # MSE Loss for targets > 7
        mse_loss = F.mse_loss(outputs[mask_high], targets[mask_high], reduction='none')  # 'none' keeps dimensionality

        # ReLU(x-7) Loss for targets < 7
        relu_loss = F.relu(outputs[mask_low] - escape_set_value)  # ReLU(outputs - 5)

        # Combine losses, consider handling empty tensors if no elements meet criteria
        total_loss = torch.zeros_like(outputs)
        total_loss[mask_high] = mse_loss
        # total_loss[mask_low] = relu_loss**2 # Squaring to align with MSE scale if desired
        total_loss[mask_low] = escape_weight*relu_loss**2

        return total_loss.mean()  # Mean to average the losses
    
criterion = CustomLoss()
# Applying different learning rates

if decoder_type=="cnn+classifier":
    optimizer = torch.optim.Adam([
    {'params': model.regression_head.parameters(), 'lr': decoder_lr},
    {'params': model.classification_head.parameters(), 'lr': 1E-2}
])
else:    
    optimizer = torch.optim.Adam([
    {'params': [p for p in model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': model.decoder.parameters(), 'lr': decoder_lr}
])


print('Begin training')
# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=5, plot_loss=False):
    model.train()
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        for sequences, labels in tqdm(train_loader):
            labels = labels.to(device)
            sequences = sequences.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)

            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validation loss
        avg_val_loss = validate_model(model, test_loader, criterion)
        validation_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}')

    if plot_loss:
        plot_losses(training_losses, validation_losses)

def validate_model(model, test_loader, criterion):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
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

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=n_epochs, plot_loss=plot_loss)
# Or load a model
# model.load_state_dict(torch.load('esm_with_mlp.pt'))


# Save the model
if save_model:
    print("Saving the model as ",  folder_path + f'/{file_names}_model_save.pt')
    torch.save(model.state_dict(), folder_path + f'/{file_names}_model_save.pt')


# Test the mlp model with Pearson correlation
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            labels = labels.to(device)
            sequences = sequences.to(device)
            outputs = model(sequences).squeeze()

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


plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
print("Correlation figure saved")

