


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
decoder_type = "mlp"
# decoder_type = "embeddings"
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
    print("max al in Desai old", max_val)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
targets = normalize_targets(targets_original, targets_ref)

# targets = df['log10Kd_ACE2'].values

# Split the dataset into training and validation sets
geno_train, geno_test, sequences_train, sequences_val, targets_train, targets_val = train_test_split(geno_array_padded, sequences, targets, test_size=prop_test, random_state=42)


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

class GenosDataset(Dataset):
    def __init__(self, genos, sequences, targets, device):
        self.genos = genos
        self.sequences = sequences
        self.targets = targets
        self.device = device

    def __len__(self):
        return len(self.genos)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].to(self.device)
        geno = self.genos[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32).to(self.device)
        return geno, sequence, target


preprocess_train = [preprocess_sequence(seq).to(device) for seq in sequences_train]  
preprocess_val = [preprocess_sequence(seq).to(device) for seq in sequences_val]  
train_dataset = GenosDataset(geno_train, preprocess_train, targets_train, device)
test_dataset = GenosDataset(geno_test, preprocess_val, targets_val, device)

# Training and validation dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Or load a model
# model_name = "20240508_1548_cnn_pearson_1layers_trained_1600train_samples_6epochs_8batch_size_0.0001decoder_lr_0.0001base_lr_log10Kd_REGN10987_model_save"
# model_name = "20240514_0825_cnn_pearson_4layers_trained_2400train_samples_3epochs_8batch_size_0.0001decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"
model_name = "20240514_0854_mlp_pearson_4layers_trained_2400train_samples_10epochs_8batch_size_0.001decoder_lr_1e-05base_lr_log10Kd_REGN10987_model_save"
print("Load the model", model_name)
model.load_state_dict(torch.load(f'models/{model_name}.pt'))

# Test the mlp model with Pearson correlation
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np

# Test the model with Pearson correlation
def evaluate_model(model, test_loader, device):
    model.eval()
    genos = []
    predictions = []
    actuals = []

    with torch.no_grad():
        for geno, sequences, labels in test_loader:
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
            genos.extend(geno)
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predictions, actuals)
    # Spearman correlation
    spearman_corr, _ = spearmanr(predictions, actuals)
    # Compute R-squared
    r_squared = r2_score(actuals, predictions)
    return genos, predictions, actuals, pearson_corr, spearman_corr, r_squared


# Usage example with the above function
genos, predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(model, test_loader, device)
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

plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+f'/{model_name}_predictions_on_Desai_new.png')
print("Correlation figure saved")


# Separate based on predictions
# pred_array = np.array(predictions)
genos_above_7 = [genos[i] for i in range(len(predictions)) if predictions[i] > 7]
genos_below_7 = [genos[i] for i in range(len(predictions)) if predictions[i] < 7]

print("Number of sequences with predicted affinity below 7",  len(genos_below_7))
print("Number of sequences with predicted affinity above 7",  len(genos_above_7))
# Convert lists of binary strings to 2D NumPy arrays of integers
genos_above_7_array = np.array([list(map(int, list(geno))) for geno in genos_above_7])
genos_below_7_array = np.array([list(map(int, list(geno))) for geno in genos_below_7])

# Calculate the frequency of 1's in each position
freq_above_7 = np.mean(genos_above_7_array, axis=0)
freq_below_7 = np.mean(genos_below_7_array, axis=0)

# Comparing frequencies
difference = freq_above_7 - freq_below_7

# Print the results
print("Frequency of 1's in genos with predictions > 7:", freq_above_7)
print("Frequency of 1's in genos with predictions < 7:", freq_below_7)
print("Difference in frequencies:", difference)


mutation_names = ['S371L', 'S371F', 'T376A', 'D405N', 'R408S', 'G446S', 'L452Q', 'L452R', 'F486V', 'R493Q', 'G496S']

# Assuming your frequency arrays are of the same length
positions = np.arange(len(freq_above_7))  # positions for the x-axis

plt.figure(figsize=(10, 5))  # Set the figure size
plt.bar(positions - 0.15, freq_above_7, width=0.3, label='Sequences with affinity prediction > 7')
plt.bar(positions + 0.15, freq_below_7, width=0.3, label='Sequences with affinity prediction < 7')
plt.xlabel('Mutation')
plt.xticks(positions, mutation_names, rotation=45)  # Set x-ticks to mutation names, rotate for readability
plt.ylabel('Frequency of 1s')
plt.title('Comparison of Frequency of mutation at Each Position')
plt.legend()
plt.savefig(folder_path+f'/{model_name}_genotype_frequencies.png')