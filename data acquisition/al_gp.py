print("Begin library imports")

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from plot_predictions_vs_actuals import plot_predictions_vs_actuals
import numpy as np
from sklearn.metrics import r2_score
import os
import datetime
from gaussian_process import gp_predictor_sklearn

print("End library imports")

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a model on sequence data.')
    parser.add_argument('--samples', type=int, default=350, help='Number of samples to use from the dataset.')
    parser.add_argument('--save_model', action='store_true', help='Enable saving of trained_model parameters')
    args = parser.parse_args()
    return args

args = parse_args()


n_samples = args.samples
save_model = args.save_model

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}_al_gp'

folder_path = f'runs/{file_names}_{n_samples}samples'
# Check if the directory exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Set random seed for reproducibility
torch.manual_seed(42)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('device', device)

print('Load the Desai dataset')
# Load the dataset
# df = pd.read_csv('df_Desai_15loci_complete.csv')
df = pd.read_csv('df_desai_new.csv')
print("length of the whole Desai new dataset", len(df))
# TEMPORARY crop the dataset for faster runtime
df = df.sample(n=n_samples, random_state=42)
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()
targets_original = df['new_log10Kd_REGN10987'].values

print("Done loading the dataset")
# Normalize function
def normalize_targets(targets):
    min_val = np.min(targets)
    max_val = np.max(targets)
    targets_normalized = 2 * ((targets - min_val) / (max_val - min_val)) - 1
    return targets_normalized

# Apply normalization
targets = normalize_targets(targets_original)

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
print("begin batch embeddings")
embeddings = batch_process_embeddings(sequences)
print("batch embeddings done")
model = None


def plot_al(pearson_corrs, spearman_corrs, rsquared_vals, save_path, cycles):
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(cycles), pearson_corrs, label='Pearson Correlation')
    plt.plot(range(cycles), spearman_corrs, label='Spearman Correlation')
    plt.plot(range(cycles), rsquared_vals, label='R-squared')
    plt.xlabel('Number of Sampling Cycles')
    plt.ylabel('Correlation Coefficients')
    plt.title('Correlation Coefficients over Sampling Cycles')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the figure as a PNG file



# Test the mlp model with Pearson correlation
def evaluate_model(model, unlabelled_seq, labels):
    outputs, variances = model.predict_pred(unlabelled_seq)

    predictions_np = np.array(outputs)
    actuals_np = np.array(labels)
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predictions_np, actuals_np)
    # Spearman correlation
    spearman_corr, _ = spearmanr(predictions_np, actuals_np)
    # Compute R-squared
    r_squared = r2_score(actuals_np, predictions_np)
    return predictions_np, actuals_np, pearson_corr, spearman_corr, r_squared

def al_uncertain(model, unlabelled_seq, unlabelled_idx, n_new, normalize_mean):
    outputs, variances = model.predict_pred(unlabelled_seq)
    outputs_np = np.array(outputs)
    variances_np = np.array(variances)
    if normalize_mean:
        ratios = outputs_np/variances_np
        # get the n_new 
        print("sampling smallest mean/var ratios")
        samples_idx = np.argsort(ratios)[:n_new]
    else:
        print("sampling largest variances")
        samples_idx = np.argsort(-variances_np)[:n_new]
    original_idx = [unlabelled_idx[i] for i in samples_idx]
    print("samples_idx", samples_idx)
    print("returning", original_idx)
    print("With unlabelled idx", unlabelled_idx)
    return original_idx

def get_uncertain(model, sequences, targets, cycles, samples_per_cycle, normalize_mean = False):
    first_ind = np.random.randint(0, len(sequences))
    labelled_idx = [first_ind]
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    unlabelled_seq = [sequences[i] for i in unlabelled_idx]
    train_set = [sequences[i] for i in labelled_idx]
    targets_train = [targets[i] for i in labelled_idx]
    
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    for _ in range(cycles):
        model = gp_predictor_sklearn(train_set, targets_train)
        model.train_pred()
        new_idx = al_uncertain(model, unlabelled_seq, unlabelled_idx, samples_per_cycle, normalize_mean)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        unlabelled_seq = [sequences[i] for i in unlabelled_idx]
        train_set = [sequences[i] for i in labelled_idx]
        print("labelled_idx", labelled_idx)
        targets_train = [targets[i] for i in labelled_idx]

        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, sequences, targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    
    # plot_al(pearson_corrs, spearman_corrs, rsquared_vals, folder_path+f'/al_uncertainty_sampling_{n_samples}samples_{cycles}cycles_{samples_per_cycle}samples_per_cycle_{normalize_mean}normalize_mean.png', cycles)

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx


def get_random(model, sequences, targets, cycles, samples_per_cycle):
    first_ind = np.random.randint(0, len(sequences))
    labelled_idx = [first_ind]
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    train_set = [sequences[i] for i in labelled_idx]
    targets_train = [targets[i] for i in labelled_idx]
    

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    for _ in range(cycles):
        model = gp_predictor_sklearn(train_set, targets_train)
        model.train_pred()
        new_idx = np.random.choice(unlabelled_idx, samples_per_cycle, replace=False)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        train_set = [sequences[i] for i in labelled_idx]
        print("labelled_idx", labelled_idx)
        targets_train = [targets[i] for i in labelled_idx]

        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, sequences, targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    
    # plot_al(pearson_corrs, spearman_corrs, rsquared_vals, folder_path+f'/al_random_sampling_{n_samples}samples_{cycles}cycles_{samples_per_cycle}samples_per_cycle.png', cycles)
    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx

print('Running al')

all_pearson1 = []
all_spearman1 = []
all_r_squared1 = []

all_pearson2 = []
all_spearman2 = []
all_r_squared2 = []

all_pearson3 = []
all_spearman3 = []
all_r_squared3 = []

runs = 10
cycles = 20
samples_per_cycle = 5

def plot_results(data1, data2, data3, title):
    # Calculate means and standard deviations for all three datasets
    mean_values1 = np.mean(data1, axis=0)
    std_dev1 = np.std(data1, axis=0)
    mean_values2 = np.mean(data2, axis=0)
    std_dev2 = np.std(data2, axis=0)
    mean_values3 = np.mean(data3, axis=0)
    std_dev3 = np.std(data3, axis=0)
    cycles = np.arange(len(mean_values1))  # Assuming all data arrays are the same length

    plt.figure(figsize=(10, 6))
    plt.errorbar(cycles, mean_values1, yerr=std_dev1, fmt='-o', label="Uncertainty Sampling with Normalized Mean")
    plt.errorbar(cycles, mean_values2, yerr=std_dev2, fmt='-o', label="Uncertainty Sampling without Normalized Mean")
    plt.errorbar(cycles, mean_values3, yerr=std_dev3, fmt='-o', label="Random Sampling")

    plt.xlabel('Number of Sampling Cycles')
    plt.ylabel(title)
    plt.title(f'{title} over Sampling Cycles')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path+f'/{title.replace(" ", "_").lower()}.png')
    
for _ in range(runs):
    model1 = None
    model2 = None
    model3 = None

    print("Running exp1")
    results1 = get_uncertain(model1, embeddings, targets, cycles, samples_per_cycle, True)
    print("Running exp2")
    results2 = get_uncertain(model2, embeddings, targets, cycles, samples_per_cycle, False)
    print("Running exp3")
    results3 = get_random(model3, embeddings, targets, cycles, samples_per_cycle)

    print("Calculating correlations")
    pearson_corrs1, spearman_corrs1, rsquared_vals1 = results1[:3]
    all_pearson1.append(pearson_corrs1)
    all_spearman1.append(spearman_corrs1)
    all_r_squared1.append(rsquared_vals1)

    pearson_corrs2, spearman_corrs2, rsquared_vals2 = results2[:3]
    all_pearson2.append(pearson_corrs2)
    all_spearman2.append(spearman_corrs2)
    all_r_squared2.append(rsquared_vals2)

    pearson_corrs3, spearman_corrs3, rsquared_vals3 = results3[:3]
    all_pearson3.append(pearson_corrs3)
    all_spearman3.append(spearman_corrs3)
    all_r_squared3.append(rsquared_vals3)

np.save('all_pearson1.npy', np.array(all_pearson1))
np.save('all_spearman1.npy', np.array(all_spearman1))
np.save('all_r_squared1.npy', np.array(all_r_squared1))

np.save('all_pearson2.npy', np.array(all_pearson2))
np.save('all_spearman2.npy', np.array(all_spearman2))
np.save('all_r_squared2.npy', np.array(all_r_squared2))

np.save('all_pearson3.npy', np.array(all_pearson3))
np.save('all_spearman3.npy', np.array(all_spearman3))
np.save('all_r_squared3.npy', np.array(all_r_squared3))

plot_results(all_pearson1, all_pearson2, all_pearson3, 'Pearson Correlation')
plot_results(all_spearman1, all_spearman2, all_spearman3, 'Spearman Correlation')
plot_results(all_r_squared1, all_r_squared2, all_r_squared3, 'R-squared Values')

print('Successfully ran al')
# unlabelled_seq = [embeddings[i] for i in unlabelled_idx]

# final_model = gp_predictor_sklearn([embeddings[i] for i in labelled_idx], [targets[i] for i in labelled_idx])
# final_model.train_pred()

# Save the model
# if save_model:
#     print("Saving the model as ",  folder_path + f'/{file_names}_model_save.pt')
#     torch.save(model.state_dict(), folder_path + f'/{file_names}_model_save.pt')




# Usage example with the above function
# predictions, actuals, pearson_correlation, spearman_correlation, r_squared = evaluate_model(final_model, embeddings, targets)
# print("Pearson Correlation Coefficient on Test Set:", pearson_correlation)
# print("Spearman Correlation Coefficient on Test Set:", spearman_correlation)
# print("R-squared on Test Set:", r_squared)


# plot_predictions_vs_actuals(predictions, actuals, pearson_correlation, spearman_correlation, r_squared, folder_path+'/predictions_vs_actuals.png')
# print("Correlation figure saved")

