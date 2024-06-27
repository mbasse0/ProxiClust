


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
from custom_model import ESMWithMLP
from custom_model_2 import ESMWithTransformer
from embedding_model import EmbeddingESMWithMLP
from custom_model3 import ESMWithCNN
from cnn_embedding import EmbeddingESMWithCNN
from sequence_dataset import SequenceDataset
from tokenized_dataset import TokenizedDataset
from sklearn.metrics import r2_score


# Set random seed for reproducibility
torch.manual_seed(42)


#################################################################
# Hyperparams

n_epochs = 1
n_samples = 32000
prop_test = 0.5

n_layers_trained = 0
batch_size = 4

base_lr = 1e-2
decoder_lr = 1e-5

# decoder_type = "transformer"
decoder_type = "cnn_embedding"

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
    model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280, output_dim=1).to(device)
elif decoder_type == "cnn_embedding":
    model = EmbeddingESMWithCNN(device=device, decoder_input_dim=1280, output_dim=1).to(device)
elif decoder_type == "cnn":
    model = ESMWithCNN(device=device, input_dim=1280).to(device)
else:
    print("Unknown decoder type")


print("decoder", decoder_type, "n_layers", n_layers_trained, "batch_size", batch_size, "n_epochs", n_epochs, "n_samples", n_samples, "prop_test", prop_test)


df_val = pd.read_csv('df_desai_new.csv')
df_val.drop(columns='Unnamed: 0')
print("Length of df_val", len(df_val))

# get the columns log10Kd_ACE2', 'new_log10Kd_REGN10987', 'log10Kd_AZD1061', 'log10Kd_AZD8895' of df_val and put them into a new dataframe
kd_new = df_val[['log10Kd_ACE2', 'new_log10Kd_REGN10987', 'log10Kd_AZD1061', 'log10Kd_AZD8895']]

# print('Load the Desai dataset')
# Load the dataset
df = pd.read_csv('df_Desai_15loci_complete.csv')
# TEMPORARY crop the dataset for faster runtime
df = df.sample(n=n_samples, random_state=42)
df.drop(columns=['Unnamed: 0'])

# Remove all lines in df that also appear in "desai_cropped.csv"
df_cropped_train = pd.read_csv('df_cropped_3000.csv')
# df = df[~df['mutant_sequence'].isin(df_cropped_train['mutant_sequence'])]

# Merge with an indicator and keep only those rows that are unique to df
merged_df = pd.merge(df, df_cropped_train, how='left', indicator=True, on=df.columns.tolist())
df_unique = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')

print("Size of df_desai_old", df.shape)


# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df_val["mutant_sequence"].tolist()

# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309']
targets = df_val['new_log10Kd_REGN10987'].values
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
from torch.utils.data import DataLoader, TensorDataset

print("Pre compute the embeddings")
def get_embeddings(model, sequences_tokens):
    # Create a TensorDataset to handle batching
    dataset = TensorDataset(sequences_tokens['input_ids'], sequences_tokens['attention_mask'])
    loader = DataLoader(dataset, batch_size=4)  # Adjust batch size to fit your GPU memory
    
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids, attention_mask = [b.to(model.device) for b in batch]
            outputs = model.esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            batch_embeddings = outputs.hidden_states[-1].detach().cpu()  # Move embeddings to CPU
            embeddings.append(batch_embeddings)
    
    # Concatenate embeddings on the CPU
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

# Example of how to call this function
embeddings_train = get_embeddings(model, sequences_train_tokens)
embeddings_val = get_embeddings(model, sequences_val_tokens)


print('Create datasets and dataloaders')
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

# Create datasets using embeddings
train_dataset = EmbeddingDataset(embeddings_train, targets_train)
test_dataset = EmbeddingDataset(embeddings_val, targets_val)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.MSELoss()
# Applying different learning rates

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': decoder_lr}
])

print('Begin training')
# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in tqdm(train_loader):
            labels = labels.to(device)
            embeddings = embeddings.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
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
        for embeddings, labels in test_loader:
            labels = labels.to(device)
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(test_loader)}")

# Train the model
# train_model(model, train_loader, criterion, optimizer, num_epochs=n_epochs)
# Or load a model
# model.load_state_dict(torch.load('models/esm_with_cnn_4layers_trained_1800train_samples_4epochs_4batch_size.pt'))
# model.load_state_dict(torch.load('models/esm_with_mlp_2layers_trained_2700train_samples_5epochs_4batch_size.pt'))
model.load_state_dict(torch.load('models/esm_with_mlp_4layers_trained_4500train_samples_4epochs_8batch_size.pt'))


# Save the model
# print("Saving the model as ", f'esm_finetuned_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.pt')
# torch.save(model.state_dict(), f'esm_finetuned_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.pt')



# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np

# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for embeddings, labels in test_loader:
            labels = labels.to(device)
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            # Squeeze the outputs if necessary to match the shape of labels
            outputs = outputs.squeeze().cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(outputs)
            actuals.extend(labels)

    # Ensure predictions and actuals are numpy arrays and correctly shaped
    predictions = np.array(predictions).squeeze()  # Squeeze to remove any extra dimensions
    actuals = np.array(actuals).squeeze()

    # Calculate Pearson correlation
    correlation, _ = pearsonr(predictions, actuals)
    # Calculate R-squared or other metrics as needed
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



'''

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            labels = labels.to(device)
            # Ensure outputs are not squeezed inappropriately; maintain the full dimensionality
            outputs = model(input_ids, attention_mask)

            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Compute Pearson correlation and R-squared for each output
    correlations = []
    r_squareds = []
    for i in range(predictions.shape[1]):  # Assuming second dimension is the number of targets
        correlation, _ = pearsonr(predictions[:, i], actuals[:, i])
        r_squared = r2_score(actuals[:, i], predictions[:, i])
        correlations.append(correlation)
        r_squareds.append(r_squared)

    return predictions, actuals, correlations, r_squareds


# Usage example with the above function
predictions, actuals, pearson_correlation, r_squared = evaluate_model(model, test_loader, device)
print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
print("R-squared on Test Set:", r_squared)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_vs_actuals(predictions, actuals, correlations, r_squareds, file_path):
    num_targets = predictions.shape[1]
    fig, axs = plt.subplots(1, num_targets, figsize=(5 * num_targets, 6))

    for i in range(num_targets):
        ax = axs[i] if num_targets > 1 else axs
        sns.scatterplot(x=actuals[:, i], y=predictions[:, i], alpha=0.6, ax=ax)
        ax.set_title(f'Target {i+1}')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predictions')
        ax.text(0.05, 0.95, f'Pearson Correlation: {correlations[i]:.2f}\nR-squared: {r_squareds[i]:.2f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')
        max_val = max(max(actuals[:, i]), max(predictions[:, i]))
        min_val = min(min(actuals[:, i]), min(predictions[:, i]))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')  # Black dashed line

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


# Example usage
file_path = f'{decoder_type}_pearson_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.png'
predictions, actuals, pearson_correlations, r_squareds = evaluate_model(model, test_loader, device)
print("Pearson Correlation Coefficients on Test Set:", pearson_correlations)
print("R-squareds on Test Set:", r_squareds)

file_path = 'path_to_save_plots.png'
plot_predictions_vs_actuals(predictions, actuals, pearson_correlations, r_squareds, file_path)
print("Correlation figures saved")

# Now call the plotting function
# plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, f'{decoder_type}_pearson_{n_layers_trained}layers_trained_{round((1-prop_test)*n_samples)}train_samples_{n_epochs}epochs_{batch_size}batch_size.png')

print("Correlation figure saved")



'''
