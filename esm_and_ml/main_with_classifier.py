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
from classifier_head import ESMWithClassifier
from sequence_dataset import SequenceDataset
from sklearn.metrics import r2_score
import os
import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a model on sequence data.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--samples', type=int, default=350, help='Number of samples to use from the dataset.')
    parser.add_argument('--test_prop', type=float, default=6/7, help='Proportion of the dataset to use for testing.')
    parser.add_argument('--layers_trained', type=int, default=2, help='Number of ESM layers to train.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and evaluation.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate for the optimizer.')
    parser.add_argument('--decoder_lr', type=float, default=1e-3, help='Learning rate for the MLP decoder.')
    parser.add_argument('--decoder_type', type=str, default='transformer', choices=['mlp', 'transformer', 'cnn', 'cnn+classifier', 'mlp+classifier', 'hybrid'], help='Type of decoder to use.')
    parser.add_argument('--plot_loss', action='store_true', help='Enable plotting of training and validation loss')
    parser.add_argument('--save_model', action='store_true', help='Enable saving of trained_model parameters')
    parser.add_argument('--antibody', type=str, default='log10Kd_REGN10987', help='Which antibody binding affinity to predict')
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

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M")
file_names = f'{timestamp}_{decoder_type}_pearson_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size_{decoder_lr}decoder_lr_{base_lr}base_lr_{antibody}'

folder_path = f'runs/{file_names}'
os.makedirs(folder_path, exist_ok=True)

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
elif decoder_type == "mlp+classifier":
    classif_model = ESMWithClassifier(device=device, input_dim=1280).to(device)
    regress_model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "cnn+classifier":
    classif_model = ESMWithClassifier(device=device, input_dim=1280).to(device)
    regress_model = ESMWithCNN(device=device, input_dim=1280).to(device)
elif decoder_type == "hybrid":
    model = ESMWithDualHead(device=device, input_dim=1280).to(device)
else:
    print("Unknown decoder type")


print(folder_path)

print('Load the Desai dataset')
# Load the dataset
df = pd.read_csv('df_Desai_15loci_complete.csv')
# TEMPORARY crop the dataset for faster runtime

#save the cropped dataset
# df.to_csv(f'df_cropped_{n_samples}.csv', index=False)

# df = pd.read_csv('df_cropped_1000.csv')
df.drop(columns=['Unnamed: 0'])
df = df.sample(n=n_samples, random_state=42)


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
            layer_num = int(name.split('.')[3])
            if layer_num >= n_layers_to_freeze:
                param.requires_grad = False

# Choose which parameters to freeze in ESM
print("freeze esm classif layers")
freeze_esm_layers(classif_model.esm, n_layers_trained)  # Freeze all but the last `n_layers_trained` layers
print("freeze esm regress layers")
freeze_esm_layers(regress_model.esm, n_layers_trained)  # Freeze all but the last `n_layers_trained` layers


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

# Loss function and optimizer
classif_criterion = nn.BCEWithLogitsLoss()
# classif_criterion = nn.MSELoss()
regress_criterion = nn.MSELoss()


# Applying different learning rates

if decoder_type in ["cnn+classifier", "mlp+classifier"]:
    classif_optimizer = torch.optim.Adam([
    {'params':[p for p in classif_model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': classif_model.decoder.parameters(), 'lr': decoder_lr}
])

    regress_optimizer = torch.optim.Adam([
    {'params':[p for p in regress_model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': regress_model.decoder.parameters(), 'lr': decoder_lr}
])

else:    
    optimizer = torch.optim.Adam([
    {'params': [p for p in model.esm.parameters() if p.requires_grad], 'lr': base_lr},
    {'params': model.decoder.parameters(), 'lr': decoder_lr}
])

print('Begin training')
# Training loop
import torch

def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            sequences = sequences.to(device)
            outputs = model(sequences)
            targets = (labels == 5.0).float().unsqueeze(1)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    model.train()
    return accuracy

def train_classif_model(model, train_loader, test_loader, optimizer, classif_criterion, num_epochs=5):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            labels = labels.to(device)
            sequences = sequences.to(device)
            # Classifier training
            optimizer.zero_grad()
            outputs = model(sequences)
            targets = (labels == 5.0).float().unsqueeze(1)
            loss = classif_criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        
        train_accuracy = calculate_accuracy(train_loader, model, device)
        test_accuracy = calculate_accuracy(test_loader, model, device)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')


def calculate_test_loss(model, classif_model, loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            
            classif_outputs = torch.sigmoid(classif_model(sequences).squeeze())
            non_outliers = (classif_outputs <= 0.5).to(device)
            
            outputs = model(sequences).squeeze().to(device).float()
            filtered_outputs = outputs[non_outliers]
            filtered_labels = labels[non_outliers].to(device)

            if len(non_outliers) > 0:
                loss = criterion(filtered_outputs, filtered_labels)
                total_loss += loss.item() * filtered_outputs.size(0)
                count += filtered_outputs.size(0)
    
    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss

def train_regress_model(model, classif_model, train_loader, test_loader, optimizer, regress_criterion, num_epochs=5, device='cpu'):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        count = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            
            with torch.no_grad():
                classif_outputs = torch.sigmoid(classif_model(sequences).squeeze())
                non_outliers = (classif_outputs <= 0.5).to(device)
            
            outputs = model(sequences).squeeze().to(device).float()
            filtered_outputs = outputs[non_outliers]
            filtered_labels = labels[non_outliers].to(device)

            if len(non_outliers) > 0:
                optimizer.zero_grad()
                loss = regress_criterion(filtered_outputs, filtered_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * filtered_outputs.size(0)
                count += filtered_outputs.size(0)

        avg_loss = total_loss / count if count > 0 else 0
        
        test_loss = calculate_test_loss(model, classif_model, test_loader, regress_criterion, device)
        
        print(f'Regression Model Training : Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}')


# Example usage
print("training classifier")
train_classif_model(classif_model, train_loader, test_loader, classif_optimizer, classif_criterion, num_epochs=n_epochs)
print("training regressor")
train_regress_model(regress_model, classif_model, train_loader, test_loader, regress_optimizer, regress_criterion, num_epochs=n_epochs)


# Save the model
if save_model:
    print("Saving the model as ",  folder_path + f'/{file_names}_classif_model_save.pt')
    torch.save(classif_model.state_dict(), folder_path + f'/{file_names}_classif_model_save.pt')
    print("Saving the model as ",  folder_path + f'/{file_names}_regress_model_save.pt')
    torch.save(regress_model.state_dict(), folder_path + f'/{file_names}_regress_model_save.pt')


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

