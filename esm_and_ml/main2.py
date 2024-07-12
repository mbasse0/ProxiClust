


import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from custom_model import ESMWithMLP, MLP
from custom_model_2 import ESMWithTransformer
from embedding_model import EmbeddingESMWithMLP
from custom_model3 import ESMWithCNN
from sequence_dataset import SequenceDataset
from tokenized_dataset import TokenizedDataset
from sklearn.metrics import r2_score
from simple_model import SimpleMLP
from simple_cnn import SimpleCNN

# Set random seed for reproducibility
torch.manual_seed(42)


#################################################################
# Hyperparams

n_epochs = 20
n_samples = 550
prop_test = 0.9

n_layers_trained = 3
batch_size = 4

base_lr = 1e-2
decoder_lr = 1e-2

# decoder_type = "transformer"
decoder_type = "simple_cnn"

###########################################################
import os
folder_path = f'runs/{decoder_type}_pearson_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size_{decoder_lr}decoder_lr_{base_lr}base_lr_.png'
os.makedirs(folder_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')

if decoder_type == "mlp":   
    model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "transformer":
    model = ESMWithTransformer(device=device, decoder_input_dim=1280).to(device)
elif decoder_type == "embedding":
    model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280, output_dim=5).to(device)
elif decoder_type == "cnn":
    model = ESMWithCNN(device=device, input_dim=1280).to(device)
elif decoder_type == "simple_mlp":
    model = SimpleMLP(input_dim = 33, output_dim=1).to(device)
elif decoder_type == "simple_cnn":
    model = SimpleCNN(input_dim = 33, num_filters=256, output_dim=1).to(device)
else:
    print("Unknown decoder type")


print("decoder", decoder_type, "n_layers", n_layers_trained, "batch_size", batch_size, "n_epochs", n_epochs, "n_samples", n_samples, "prop_test", prop_test)


df_val = pd.read_csv('df_desai_new.csv')
df_val.drop(columns='Unnamed: 0')


# get the columns log10Kd_ACE2', 'new_log10Kd_REGN10987', 'log10Kd_AZD1061', 'log10Kd_AZD8895' of df_val and put them into a new dataframe
kd_new = df_val[['log10Kd_ACE2', 'new_log10Kd_REGN10987', 'log10Kd_AZD1061', 'log10Kd_AZD8895']]

# print('Load the Desai dataset')
# Load the dataset
df = pd.read_csv('df_Desai_15loci_complete.csv')
# TEMPORARY crop the dataset for faster runtime
df = df.sample(n=n_samples, random_state=42)
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()

# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
targets = df['log10Kd_REGN10987'].values
# targets = df['log10Kd_ACE2'].values

# Split the dataset into training and validation sets
sequences_train, sequences_val, targets_train, targets_val = train_test_split(sequences, targets, test_size=prop_test, random_state=42)

# Quick plot of the target values

# # Plot the first 50 affinity values from the training set as a 1D scatter plot on the x-axis
# plt.figure(figsize=(10, 1))  # Narrow figure height as we only need one horizontal line
# y_zeros = [0] * 50  # Create a list of zeros for the y-axis values
# plt.scatter(targets_train[:50], y_zeros, marker='o', color='b')  # Use scatter to plot points on the x-axis
# plt.title('Affinity Values Distribution of the First 50 Sequences in Training Set')
# plt.xlabel('log10Kd_ACE2')
# plt.yticks([])  # Hide y-axis ticks as they are not needed
# plt.grid(True, axis='x')  # Grid only along x-axis
# plt.savefig('training_set_affinities_1D_horizontal.png')  # Save the plot to a file
# plt.show()


# Tokenize the training and validation sequences
tokenizer = AutoTokenizer.from_pretrained('models')
sequences_train_tokens = tokenizer(sequences_train, return_tensors='pt', padding='longest', truncation=True, max_length=512)
sequences_val_tokens = tokenizer(sequences_val, return_tensors='pt', padding='longest', truncation=True, max_length=512)

vocab_size = tokenizer.vocab_size
print("vocab size is ", vocab_size)



def one_hot_encode_tokens(token_ids, vocab_size):
    # token_ids is a batch of token ID tensors from the tokenizer
    # vocab_size is the size of the tokenizer's vocabulary

    # Create a tensor of shape [batch_size, sequence_length, vocab_size]
    # where each token ID is one-hot encoded.
    one_hot_encoded = torch.nn.functional.one_hot(token_ids, num_classes=vocab_size)
    return one_hot_encoded

# Assuming sequences_train_tokens and sequences_val_tokens are already obtained as shown
sequences_train_one_hot = one_hot_encode_tokens(sequences_train_tokens['input_ids'], vocab_size)
sequences_val_one_hot = one_hot_encode_tokens(sequences_val_tokens['input_ids'], vocab_size)

print('Create datasets and dataloaders')
# Training and validation datasets
train_dataset = SequenceDataset(sequences_train_one_hot, targets_train, device)
test_dataset = SequenceDataset(sequences_val_one_hot, targets_val, device)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.MSELoss()
# Applying different learning rates

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': decoder_lr}
])

print('Begin training')
# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for  sequences, labels in tqdm(train_loader):
            # print("shape items in batch", items.shape)
            labels = labels.to(device)
            sequences = sequences.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}')
        # Uncomment to check for overfitting
        validate_model(model, test_loader, criterion)
 
def validate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            sequences, labels = data
            labels = labels.to(device)
            sequences = sequences.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(test_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=n_epochs)
# Or load a model
# model.load_state_dict(torch.load('esm_with_mlp.pt'))


# Save the model
# print("Saving the model as ", f'esm_with_mlp_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.pt')
# torch.save(model.state_dict(), f'esm_with_mlp_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.pt')



# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np


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
    correlation, _ = pearsonr(predictions, actuals)
    # Compute R-squared
    r_squared = r2_score(actuals, predictions)
    return predictions, actuals, correlation, r_squared

# Usage example with the above function
predictions, actuals, pearson_correlation, r_squared = evaluate_model(model, test_loader, device)
print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
print("R-squared on Test Set:", r_squared)


def plot_predictions_vs_actuals(predictions, actuals, correlation, r_squared, file_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actuals, y=predictions, alpha=0.6)
    plt.title('Predictions vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')

    # Adding Pearson correlation and R-squared on the plot
    plt.text(0.05, 0.95, f'Pearson Correlation: {correlation:.2f}\nR-squared: {r_squared:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Adding x=y line
    max_val = max(max(actuals), max(predictions))
    min_val = min(min(actuals), min(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # k-- is a black dashed line

    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
print("Correlation figure saved")

