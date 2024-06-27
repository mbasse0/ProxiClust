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
timestamp = now.strftime("%Y%m%d_%H%M%S")
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



# dataset_path = 'df_bloom_complete.csv'
dataset_path = 'df_desai_new.csv'
# dataset_path = 'df_Desai_15loci_complete.csv'

print(dataset_path)
df = pd.read_csv(dataset_path)

# print("length of the whole Desai new dataset", len(df))
# TEMPORARY crop the dataset for faster runtime
# df = df.sample(n=n_samples, random_state=42)
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()
# targets_original = df['new_log10Kd_REGN10987'].values
# antibody = 'log10Kd_ACE2'
antibody = 'log10Kd_ACE2'
print("antibody", antibody)
targets_original = df[antibody].values

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

embeddings = batch_process_embeddings(sequences)



def train_test_split_smart(sequences, targets, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    # Calculate distances from each sequence embedding to the wildtype embedding
    distances = np.array([np.linalg.norm(seq - wildtype_embedding) for seq in sequences])
    
    # Get indices sorted by distance (ascending)
    sorted_indices = np.argsort(distances)
    
    # Determine the number of test samples
    n_test = int(test_size * len(sequences))
    n_train = len(sequences)-n_test

    train_indices = sorted_indices[n_test:]  # Train on sequences less similar to wildtype
    test_indices = sorted_indices[:n_test]   # Test on sequences more similar to wildtype
    

    sequences_train = [sequences[i] for i in train_indices]
    sequences_test = [sequences[i] for i in test_indices]
    targets_train = targets[train_indices]
    targets_test = targets[test_indices]
    return sequences_train, sequences_test, targets_train, targets_test




def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))



def get_diverse_idx(sequences, targets, cycles, samples_per_cycle, init_size, selection_criterion):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

    for j in range(cycles):
        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences[i] for i in labelled_idx]
        
        if selection_criterion == 'largest_mean':
            distances = [np.mean([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'min_linkage':
            # Get the highest minimum distance for min linkage
            distances = [np.min([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'max_linkage':
            # Get the highest maximum distance for max linkage
            distances = [np.max([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))

        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        # new_seq = np.array([sequences[i] for i in new_idx])
        # new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        # remaining_seq = [sequences[i] for i in unlabelled_idx]
        # remaining_targets = [targets[i] for i in unlabelled_idx]
    return labelled_idx, unlabelled_idx



def get_diverse2_idx(sequences, embeddings, targets, cycles, samples_per_cycle, init_size, selection_criterion):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

    for j in range(cycles):
        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences[i] for i in labelled_idx]
        labelled_embeddings = [embeddings[i] for i in labelled_idx]

        if selection_criterion == 'largest_mean':
            hamming_distances = [np.mean([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            euclidian_distances = [np.mean([np.linalg.norm(embeddings[idx] - embedding) for embedding in labelled_embeddings]) for idx in unlabelled_idx] 
            sorted_indices = np.lexsort((-np.array(euclidian_distances), -np.array(hamming_distances)))
        elif selection_criterion == 'min_linkage':
            # Get the highest minimum distance for min linkage
            hamming_distances = [np.min([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            # euclidian_distances = [np.min([np.linalg.norm(sequences[idx] - embedding) for embedding in labelled_embeddings]) for idx in unlabelled_idx] 
            euclidian_distances = [np.mean([np.linalg.norm(embeddings[idx] - embedding) for embedding in labelled_embeddings]) for idx in unlabelled_idx] 

            sorted_indices = np.lexsort((-np.array(euclidian_distances), -np.array(hamming_distances)))
        elif selection_criterion == 'max_linkage':
            # Get the highest maximum distance for max linkage
            hamming_distances = [np.max([hamming_distance(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            euclidian_distances = [np.max([np.linalg.norm(embeddings[idx] - embedding) for embedding in labelled_embeddings]) for idx in unlabelled_idx] 
            sorted_indices = np.lexsort((-np.array(euclidian_distances), -np.array(hamming_distances)))

        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        # new_seq = np.array([sequences[i] for i in new_idx])
        # new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        # remaining_seq = [sequences[i] for i in unlabelled_idx]
        # remaining_targets = [targets[i] for i in unlabelled_idx]
    return labelled_idx, unlabelled_idx



def get_diverse3_idx(sequences, embeddings, targets, cycles, samples_per_cycle, init_size, selection_criterion):
    epistasis_seq_ind = list(range(16384,16400))
    labelled_idx = epistasis_seq_ind + [16384] + [24576, 20480, 18432, 17408, 16896, 16640, 16512, 16448, 0, 16416, 16400]
    print(labelled_idx)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    return labelled_idx, unlabelled_idx



# Loss function and optimizer
criterion = nn.MSELoss()


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
    # plt.savefig(folder_path+'/training_validation_losses.png')
    plt.close()


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



def calculate_disagreement(predictions):
    # Here you might calculate the standard deviation or another statistic
    # that reflects disagreement among the models.
    return torch.std(torch.stack(predictions), dim=0)

cycles = 40
samples_per_cycle = 1
init_size = 10
selection_criterion = 'min_linkage'
runs=2


rsquared_list = []
for run in range(runs):
    '''
    model = EmbeddingESMWithMLP(device=device, decoder_input_dim=1280).to(device) 
    # Applying different learning rates

    if decoder_type=="cnn+classifier":
        optimizer = torch.optim.Adam([
        {'params': model.regression_head.parameters(), 'lr': decoder_lr},
        {'params': model.classification_head.parameters(), 'lr': 1E-2}
    ])
    elif decoder_type == "hybrid":
        optimizer = torch.optim.Adam([
        {'params': model.regression_head.parameters(), 'lr': decoder_lr},
        {'params': model.classification_head.parameters(), 'lr': 1E-2},
        {'params': [p for p in model.esm.parameters() if p.requires_grad], 'lr': base_lr}
    ])
    else:    
        optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': decoder_lr}
    ])

    # Load the model trained on desai old (no finetuning)
    model_path = "models/20240517_0919_mlp_pearson_1layers_trained_7200train_samples_10epochs_4batch_size_0.001decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"
    model.load_state_dict(torch.load(model_path + '.pt'))


    print(f'Run {run+1}')
    # labelled_idx, unlabelled_idx = get_diverse_idx(sequences, targets, cycles, samples_per_cycle, init_size, selection_criterion)
    # labelled_idx, unlabelled_idx = get_diverse2_idx(sequences, embeddings, targets, cycles, samples_per_cycle, init_size, selection_criterion)
    # labelled_idx, unlabelled_idx = get_diverse3_idx(sequences, embeddings, targets, cycles, samples_per_cycle, init_size, selection_criterion)
    print("default")
    # embeddings_train, embeddings_val, targets_train, targets_val = [embeddings[i] for i in labelled_idx], [embeddings[i] for i in unlabelled_idx], [targets[i] for i in labelled_idx], [targets[i] for i in unlabelled_idx]

    embeddings_train, embeddings_val, targets_train, targets_val = train_test_split(embeddings, targets, test_size=prop_test, random_state=42)

    print('Create datasets and dataloaders')
    # Training and validation datasets

    train_dataset = SequenceDataset(embeddings_train, targets_train, device)
    test_dataset = SequenceDataset(embeddings_val, targets_val, device)

    # Training and validation dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=n_epochs, plot_loss=plot_loss)

    # Usage example with the above function
    predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(model, test_loader, device)
    rsquared_list.append(r_squared)
    print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
    print("Spearman Correlation Coefficient on Test Set:", spearman_correlation)
    print("R-squared on Test Set:", r_squared)


    plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
    print("Correlation figure saved")
    '''


     # Initial random selection of data
    embeddings_train, embeddings_val, targets_train, targets_val = train_test_split(embeddings, targets, test_size=0.1, random_state=42)

    for cycle in range(cycles):
        # For simplicity, assuming all models are initialized similarly and share the same architecture
        committee_models = [EmbeddingESMWithMLP(device=device, decoder_input_dim=1280).to(device) for _ in range(committee_size)]
        
        model_path = "models/20240517_0919_mlp_pearson_1layers_trained_7200train_samples_10epochs_4batch_size_0.001decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"
        # Load model states if necessary or reinitialize weights
        for model in committee_models:
            model.load_state_dict(torch.load(model_path + '.pt'))

        # Create DataLoader for unlabeled data
        unlabeled_loader = DataLoader(SequenceDataset(embeddings_val, targets_val, device), batch_size=batch_size, shuffle=False)

        # Collect predictions from all models
        predictions = [model(unlabeled_loader.dataset.tensors[0]) for model in committee_models]
        
        # Calculate disagreement
        disagreements = calculate_disagreement(predictions)
        
        # Select data points with maximum disagreement
        _, top_indices = torch.topk(disagreements, samples_per_cycle)
        selected_embeddings = [embeddings_val[idx] for idx in top_indices]
        selected_targets = [targets_val[idx] for idx in top_indices]

        # Update training and validation sets
        embeddings_train.extend(selected_embeddings)
        targets_train.extend(selected_targets)
        embeddings_val = [emb for idx, emb in enumerate(embeddings_val) if idx not in top_indices]
        targets_val = [targ for idx, targ in enumerate(targets_val) if idx not in top_indices]

        # Update DataLoader for training
        train_loader = DataLoader(SequenceDataset(embeddings_train, targets_train, device), batch_id=batch_size, shuffle=True)
        test_loader = DataLoader(SequenceDataset(embeddings_val, targets_val, device), batch_id=batch_size, shuffle=False)

        # Retrain models or continue training
        for model in committee_models:
            train_model(model, train_loader, optimizer, num_epochs)

        print(f"Completed cycle {cycle + 1}")

print("Done with all ", runs, " runs")
print("Mean R-squared:", np.mean(rsquared_list))