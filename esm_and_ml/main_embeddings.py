import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
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
import distance
from sequence_dataset import SequenceDataset
from sklearn.metrics import r2_score
import os
import datetime
from embedding_model import EmbeddingESMWithMLP

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


print(file_names)
folder_path = f'./runs/{file_names}'
# Check if the directory exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Set random seed for reproducibility
torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
print('Create the model')

# if decoder_type == "mlp":   
#     model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device)
# elif decoder_type == "transformer":
#     model = ESMWithTransformer(device=device, decoder_input_dim=1280).to(device)
# elif decoder_type == "cnn":
#     model = ESMWithCNN(device=device, input_dim=1280).to(device)
# elif decoder_type == "cnn+classifier":
#     model = ESMWithDualHead(device=device, input_dim=1280).to(device)
# elif decoder_type == "mlp+classifier":
#     model = ESMWithDualHead(device=device, input_dim=1280).to(device)
# elif decoder_type == "hybrid":
#     model = ESMWithDualHead(device=device, input_dim=1280).to(device)
# else:
#     print("Unknown decoder type")


# model = ESMWithMLP(device=device, decoder_input_dim=1280).to(device) 
model = ESMWithCNN(device=device, input_dim=1280).to(device)


# Load the model trained on desai old (no finetuning)
# model_path = "./models/20240722_1005_mlp_pearson_0layers_trained_5000train_samples_20epochs_8batch_size_0.001decoder_lr_0.0001base_lr_log10Kd_REGN10987_model_save"
# model.load_state_dict(torch.load(model_path + '.pt'))
# print("with the loaded model")
print(folder_path)

print('Load the Desai dataset')
# Load the dataset
df = pd.read_csv('./datasets/df_Desai_15loci_complete.csv')
# df = pd.read_csv('./datasets/df_desai_new.csv')
# TEMPORARY crop the dataset for faster runtime
df = df.sample(n=n_samples, random_state=42)
#save the cropped dataset
# df.to_csv(f'df_cropped_{n_samples}.csv', index=False)
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()


targets = df['log10Kd_ACE2'].values
# targets_original = df['new_log10Kd_REGN10987'].values

import numpy as np

# Normalize function
def normalize_targets(targets):
    min_val = np.min(targets)
    max_val = np.max(targets)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
# targets = normalize_targets(targets_original)

# targets = df['log10Kd_ACE2', 'log10Kd_CB6', 'log10Kd_CoV555', 'log10Kd_REGN10987', 'log10Kd_S309'].values

esm = AutoModelForMaskedLM.from_pretrained('models').to(device)
tokenizer = AutoTokenizer.from_pretrained('models')

# Compute the wildtype embedding
WILDTYPE = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
# OMICRON = 'NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST'


input_ids, attention_mask = tokenizer(WILDTYPE, return_tensors='pt', padding='longest', truncation=True, max_length=512).values()
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
outputs = esm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]
mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
sum_hidden_states = torch.sum(hidden_states * mask_expanded, dim=1)
sum_mask = mask_expanded.sum(dim=1)
wildtype_embedding = sum_hidden_states / sum_mask
wildtype_embedding = wildtype_embedding.squeeze(0).detach().cpu().numpy()

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

# embeddings = batch_process_embeddings(sequences)



def freeze_esm_layers(model, n_layers_to_freeze):
    for name, param in model.named_parameters():
        if 'encoder.layer' in name:
            layer_num = int(name.split('.')[3])
            if layer_num < 32 +1 - n_layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = True

            
# Choose which parameters to freeze in ESM
freeze_esm_layers(model.esm, n_layers_trained) 



for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Training: {name}")

# Split the dataset into training and validation sets
# embeddings_train, embeddings_val, targets_train, targets_val = train_test_split(embeddings, targets, test_size=prop_test, random_state=42)
sequences_train, sequences_val, targets_train, targets_val = train_test_split(sequences, targets, test_size=prop_test, random_state=42)



print('Create datasets and dataloaders')
# Training and validation datasets
def preprocess_sequence(seq):
    # This is a placeholder function to convert sequence strings to tensor.
    # Implement your specific sequence preprocessing here.
    # Example: Convert each character to an integer or one-hot encoded tensor.
    return torch.tensor([ord(c) - ord('A') for c in seq], dtype=torch.float32)  # Simplistic example

preprocess_train = [preprocess_sequence(seq).to(device) for seq in sequences_train]  
preprocess_val = [preprocess_sequence(seq).to(device) for seq in sequences_val]  

train_dataset = SequenceDataset(preprocess_train, targets_train, device)
test_dataset = SequenceDataset(preprocess_val, targets_val, device)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.MSELoss()
# Applying different learning rates
 


optimizer = torch.optim.Adam([
{'params': model.decoder.parameters(), 'lr': decoder_lr},
{'params': [p for p in model.esm.parameters() if p.requires_grad], 'lr': base_lr}
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
        plot_losses(training_losses, validation_losses, exclude_first=False)

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

def plot_losses(training_losses, validation_losses, exclude_first=True):
    plt.figure(figsize=(10, 6))
    if exclude_first:
        plt.plot(training_losses[1:], label='Training Loss')
        plt.plot(validation_losses[1:], label='Validation Loss')
    else:
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
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

