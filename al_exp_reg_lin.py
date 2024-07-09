
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import os
import datetime
from gaussian_process import gp_predictor_sklearn
from tqdm import tqdm
from sklearn.gaussian_process.kernels import RationalQuadratic
import random
import json
from goppy import OnlineGP, SquaredExponentialKernel
from sklearn.decomposition import PCA
from plot_predictions_vs_actuals import plot_predictions_vs_actuals
from scipy.special import erf
from sklearn.linear_model import LinearRegression



n_samples = 1151

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}_al_gp'

# Set random seed for reproducibility
torch.manual_seed(42)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('device', device)


print('Load the dataset')
# total_rows = 32768  # Example total rows in the CSV
# total_rows = 1151
# Generate random row indices
# skip = sorted(random.sample(range(1, total_rows+1), total_rows - n_samples))
# print("skipped", len(skip))
# Read the random rows
# df = pd.read_csv('df_Desai_15loci_complete.csv', skiprows=skip)

# df = pd.read_csv('df_Desai_15loci_complete.csv')
# df = pd.read_csv('df_desai_new.csv')


# dataset_path = 'df_bloom_complete.csv'
# dataset_path = 'df_desai_new.csv'
# dataset_path = 'df_Desai_15loci_complete.csv'
dataset_path = 'df_desai_old_full.csv'

print(dataset_path)
df = pd.read_csv(dataset_path)

# df["mean_representation"] = df["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])
df["onehot"] = df["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])

# print("length of the whole Desai new dataset", len(df))
# TEMPORARY crop the dataset for faster runtime
# df = df.sample(n=3000, random_state=42)
print("runnning on full desai new")
df.drop(columns=['Unnamed: 0'])

# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()
# targets_original = df['new_log10Kd_REGN10987'].values
antibody = 'log10Kd_ACE2'
# antibody = 'log10Kd_S309'
# antibody = "new_log10Kd_REGN10987"

print("antibody", antibody)
targets_original = df[antibody].values


print("Done loading the dataset")

# full_embeddings = torch.tensor(df['mean_representation'].apply(json.loads))
df["mean_representation"] = df["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])
full_embeddings = torch.tensor(df["mean_representation"].tolist())


pca = PCA(n_components=20)
embeddings = pca.fit_transform(full_embeddings)

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


def batch_process_embeddings(sequences, batch_size=16):
    # Create DataLoader
    dataset = sequences
    loader = DataLoader(dataset, batch_size=batch_size)

    # Collect all embeddings
    all_embeddings = []
    for batch in tqdm(loader):
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



# Test the mlp model with Pearson correlation
def evaluate_model(model, unlabelled_idx, labels):
    X_unlabelled = [df["onehot"][i] for i in unlabelled_idx]
    predictions_np = model.predict(X_unlabelled)
    # predictions_np = np.squeeze(predictions)
    actuals_np = np.array(labels)
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(predictions_np, actuals_np)
    # Spearman correlation
    spearman_corr, _ = spearmanr(predictions_np, actuals_np)
    # Compute R-squared
    r_squared = r2_score(actuals_np, predictions_np)
    return predictions_np, actuals_np, pearson_corr, spearman_corr, r_squared


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(sequences, initial_idx, labelled_idx, unlabelled_idx, save_path=None):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(sequences)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color='gray', alpha=0.5, label='All sequences')
    
    
    # Highlight unlabelled points
    unlabelled_points = tsne_results[unlabelled_idx]
    plt.scatter(unlabelled_points[:, 0], unlabelled_points[:, 1], color='yellow', label='Unlabelled sequences', edgecolors='black')
    
    # Highlight labelled points
    labelled_points = tsne_results[labelled_idx]
    plt.scatter(labelled_points[:, 0], labelled_points[:, 1], color='blue', label='Labelled sequences', edgecolors='black')

    
    # Highlight the initial point
    initial_points = tsne_results[initial_idx]
    plt.scatter(initial_points[:, 0], initial_points[:, 1], color='red', label='Initial points', edgecolors='black', s=100)
        
    # Add legend
    plt.legend()
    
    # Add title
    plt.title('t-SNE of Sequences Dataset')
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    # else:
    #     plt.show()



def get_random(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    model = LinearRegression()
    
    for j in range(cycles):
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        model.fit(X_train, y_train)

        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)


        # Choose a new sample
        new_idx = np.random.choice(unlabelled_idx, samples_per_cycle, replace=False)
        labelled_idx.extend(new_idx)
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        
        
    
    print("Train data after all cycles", len(X_train))
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_random_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/random_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def get_diverse(sequences, sequences_str, targets, cycles, samples_per_cycle, folder_path, selection_criterion='largest_mean', init_size=10, plot_corr=False):
    # Initialize with a random sequence
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    model = LinearRegression()
    
    for j in range(cycles):
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        model.fit(X_train, y_train)

        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)


        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences_str[i] for i in labelled_idx]
        

        if selection_criterion == 'largest_mean':
            distances = [np.mean([hamming_distance(sequences_str[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'min_linkage':
            # Get the highest minimum distance for min linkage
            distances = [np.min([hamming_distance(sequences_str[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'max_linkage':
            # Get the highest maximum distance for max linkage
            distances = [np.max([hamming_distance(sequences_str[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))



        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        labelled_idx.extend(new_idx)
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    
    print("Train data after all cycles", len(X_train))

    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_diverse_{selection_criterion}_cycle_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/diverse_tsne_{selection_criterion}_cycle_{j}.png')

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx



def get_diverse_embeddings(sequences, targets, cycles, samples_per_cycle, folder_path, selection_criterion='largest_mean', init_size=10,plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    
    model = LinearRegression()

    for j in range(cycles):
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        model.fit(X_train, y_train)        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        mean_distances = []
        labelled_sequences = [sequences[i] for i in labelled_idx]
        
        # Find the indices of the unlabelled sequences with the largest mean distances
        if selection_criterion == 'largest_mean':
            distances = [np.mean([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'min_linkage':
            # Get the highest minimum distance for min linkage
            distances = [np.min([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'max_linkage':
            # Get the highest maximum distance for max linkage
            distances = [np.max([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))

        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        new_seq = np.array([sequences[i] for i in new_idx])
        new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_seq = [sequences[i] for i in unlabelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, remaining_seq, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_diverse_embed_{selection_criterion}_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/diverse_emb_tsne_{selection_criterion}_{j}.png')

    # plot_al(pearson_corrs, spearman_corrs, rsquared_vals, folder_path+f'/al_random_sampling_{n_samples}samples_{cycles}cycles_{samples_per_cycle}samples_per_cycle.png', cycles)
    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx


def get_delta_alpha(y_prime, k, alpha, var_f, K_inv):
    prefactor = ( y_prime-np.dot(k, alpha) )/var_f
    array = np.append(np.dot(K_inv, k), -prefactor)
    return array


def approx_prob(gp, y_prime, x_prime, sig_f):
    results = gp.predict(x_prime, what=("mean"))
    f = results["mean"]
    z = erf(-f/(np.sqrt(2)*sig_f))
    if y_prime == 1:
        return 1/2 - z/2
    else:
        return 1/2 + z/2

def get_pde_score(gp, x_prime, sig_f, all_X):
    Y = gp.y_train
    X = gp.x_train
    kernel = gp.kernel
    n = len(X)

    N = len(all_X)

    preds = gp.predict(x_prime, what=("mean", "mse"))
    y_pred = preds["mean"]
    y_var = preds["mse"]
    # prob_1 = approx_prob(gp, 1, x_prime, sig_f)
    k = kernel._calc_distance(x_prime, X).squeeze(0)
    K_inv = gp.inv_cov_matrix
    alpha = np.dot(K_inv, Y)
    delta_alpha = get_delta_alpha(y_pred, k, alpha, y_var, K_inv)
    # new_X = np.vstack((X, x_prime))
    pde = np.sum(kernel._calc_distance(x_prime, all_X).squeeze(0))
    score =  np.linalg.norm(delta_alpha, ord=1) * pde

    return score



def al_pde_values(gp, unlabelled_seq, sequences, unlabelled_idx, n_new):
    
    op_emoc_scores = [-get_pde_score(gp, np.array([el]), 0.1, sequences) for el in unlabelled_seq]

    # print("the emoc scores", len(emoc_scores), emoc_scores)
    samples_idx = np.argsort(op_emoc_scores)[:n_new]
    original_idx = [unlabelled_idx[i] for i in samples_idx]
    return original_idx


def get_pde(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    unlabelled_seq = [sequences[i] for i in unlabelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
    
    for j in range(cycles):
        gp.add(new_seq, new_targets)
        new_idx = al_pde_values(gp, unlabelled_seq, sequences, unlabelled_idx, samples_per_cycle)
        new_seq = np.array([sequences[i] for i in new_idx])
        new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_seq = [sequences[i] for i in unlabelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(gp, remaining_seq, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_random_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/pde_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx



def get_emoc_score(gp, x_prime, sig_f, all_X):
    Y = gp.y_train
    X = gp.x_train
    kernel = gp.kernel
    n = len(X)

    N = len(all_X)

    preds = gp.predict(x_prime, what=("mean", "mse"))
    y_pred = preds["mean"]
    y_var = preds["mse"]
    # prob_1 = approx_prob(gp, 1, x_prime, sig_f)
    k = kernel._calc_distance(x_prime, X).squeeze(0)
    K_inv = gp.inv_cov_matrix
    alpha = np.dot(K_inv, Y)
    delta_alpha = get_delta_alpha(y_pred, k, alpha, y_var, K_inv)
    # new_X = np.vstack((X, x_prime))
    res = 0
    for i in range(len(delta_alpha)):
        if i == len(delta_alpha)-1:
            dist = kernel._calc_distance(x_prime, all_X).squeeze(0)
        else:
            dist = kernel._calc_distance(np.array([X[i]]), all_X).squeeze(0)
        res += abs(delta_alpha[i]) * np.sum(dist)
    pde = np.sum(kernel._calc_distance(x_prime, all_X).squeeze(0))
    score =  np.linalg.norm(delta_alpha, ord=1) * pde

    return score

def al_emoc_values(gp, unlabelled_seq, sequences, unlabelled_idx, n_new):
    
    op_emoc_scores = [-get_emoc_score(gp, np.array([el]), 0.1, sequences) for el in unlabelled_seq]

    # print("the emoc scores", len(emoc_scores), emoc_scores)
    samples_idx = np.argsort(op_emoc_scores)[:n_new]
    original_idx = [unlabelled_idx[i] for i in samples_idx]
    return original_idx

def get_fast_emoc(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    unlabelled_seq = [sequences[i] for i in unlabelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
    
    for j in range(cycles):
        gp.add(new_seq, new_targets)
        new_idx = al_emoc_values(gp, unlabelled_seq, sequences, unlabelled_idx, samples_per_cycle)
        new_seq = np.array([sequences[i] for i in new_idx])
        new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_seq = [sequences[i] for i in unlabelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(gp, remaining_seq, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_emoc_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/emoc_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx

def get_best_emoc(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    unlabelled_seq = [sequences[i] for i in unlabelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
    
    for j in range(cycles):
        gp.add(new_seq, new_targets)
        best_score = 0
        new_idx = 0
        for idx in unlabelled_idx:
            new_seq = np.array([sequences[idx]])
            new_targets = np.expand_dims([targets[idx]], axis=1)
            remaining_seq = [sequences[i] for i in unlabelled_idx if i != idx]
            remaining_targets = [targets[i] for i in unlabelled_idx if i != idx]

            gp_copy = gp.copy()
            gp_copy.add(new_seq, new_targets)
            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(gp_copy, new_seq, new_targets)
            if r_squared > best_score:
                best_score = r_squared
                new_idx = idx

        new_seq = np.array([sequences[i] for i in new_idx])
        new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_seq = [sequences[i] for i in unlabelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(gp, remaining_seq, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_best_emoc_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/best_emoc_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx

def plot_results(datasets, labels, title, folder_path):
    plt.figure(figsize=(10, 6))

    # Ensure that datasets and labels are of the same length
    if len(datasets) != len(labels):
        raise ValueError("Number of datasets and labels must match")

    for data, label in zip(datasets, labels):
        mean_values = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        cycles = np.arange(len(mean_values))
        plt.errorbar(cycles, mean_values, yerr=std_dev, fmt='-o', label=label, alpha=0.7)

    plt.xlabel('Number of Sampling Cycles')
    plt.ylabel(title)
    plt.title(f'{title} over Sampling Cycles for {antibody} ({dataset_path} dataset)')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + f'/{title.replace(" ", "_").lower()}.png')
    plt.show()


# def plot_results(datasets, labels, title, folder_path):
#     plt.figure(figsize=(10, 6))

#     # Ensure that datasets and labels are of the same length
#     if len(datasets) != len(labels):
#         raise ValueError("Number of datasets and labels must match")

#     # Calculate mean and std for the benchmark dataset
#     benchmark_data = datasets[0]
#     benchmark_mean = np.mean(benchmark_data, axis=0)
#     benchmark_std = np.std(benchmark_data, axis=0)
#     cycles = np.arange(len(benchmark_mean))

#     # Plot the benchmark dataset
#     plt.errorbar(cycles, benchmark_mean, yerr=benchmark_std, fmt='-o', label=labels[0], alpha=0.7, color='black')

#     # Plot other datasets relative to the benchmark
#     for data, label in zip(datasets[1:], labels[1:]):
#         mean_values = np.mean(data, axis=0)
#         std_dev = np.std(data, axis=0)
#         diff_mean = mean_values - benchmark_mean
#         diff_std = np.sqrt(std_dev**2 + benchmark_std**2)  # Combining standard deviations

#         plt.errorbar(cycles, diff_mean, yerr=diff_std, fmt='-o', label=label, alpha=0.7)

#     plt.xlabel('Number of Sampling Cycles')
#     plt.ylabel(f'{title} Difference Relative to Benchmark')
#     plt.title(f'{title} Differences over Sampling Cycles for {antibody} ({dataset_path} dataset)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(folder_path + f'/{title.replace(" ", "_").lower()}_relative_to_benchmark.png')
#     plt.show()

from sklearn.neighbors import KernelDensity


def sample_sparse(sequences, unlabelled_idx, labelled_seq, samples_per_cycle):
    # Get a sequence in unlabelled sequence in the most sparse region. Estimate density with a kernel density estimator
    unlabelled_sequences = np.array([sequences[i] for i in unlabelled_idx])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(unlabelled_sequences)
    log_dens = kde.score_samples(unlabelled_sequences)
    sorted_indices = np.argsort(log_dens)
    return [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]

from scipy.spatial.distance import cdist

def sample_sparse_away(sequences, unlabelled_idx, labelled_sequences, samples_per_cycle):
    # Convert sequences to NumPy arrays for distance calculation
    unlabelled_sequences = np.array([sequences[i] for i in unlabelled_idx])
    labelled_sequences = np.array(labelled_sequences)
    
    # Estimate density with a kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(unlabelled_sequences)
    log_dens = kde.score_samples(unlabelled_sequences)
    
    # Calculate the minimum distance of each unlabelled sequence to any labelled sequence
    distances = []
    for unlabelled_seq in unlabelled_sequences:
        min_dist = np.min([np.linalg.norm(unlabelled_seq - labelled_seq) for labelled_seq in labelled_sequences])
        distances.append(min_dist)
    distances = np.array(distances)
    
    
    norm_dens = log_dens/np.mean(log_dens)
    norm_dist = distances/np.mean(distances)
    # Combine log density and distance into a combined score
    combined_score = norm_dens - norm_dist  # Negate distances to favor sparse and far points
    sorted_indices = np.argsort(combined_score)
    
    return [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]

def get_sparse(sequences, sequences_str, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    # Initialize with a random sequence
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
    
    for j in range(cycles):
        gp.add(new_seq, new_targets)
        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences[i] for i in labelled_idx]
        # new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        new_idx = sample_sparse_away(sequences, unlabelled_idx, labelled_sequences, samples_per_cycle)
        new_seq = np.array([sequences[i] for i in new_idx])
        new_targets = np.expand_dims([targets[i] for i in new_idx], axis=1)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        remaining_seq = [sequences[i] for i in unlabelled_idx]
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(gp, remaining_seq, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)
    
    print("Train data after all cycles", len(gp.x_train))

    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_sparse_cycle_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/diverse_sparse_cycle_{j}.png')

    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx


def plot_indiv_results(datasets, labels, title, folder_path):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))  # Using the Viridis colormap

    # Ensure that datasets and labels are of the same length
    if len(datasets) != len(labels):
        raise ValueError("Number of datasets and labels must match")

    for idx, (data, label) in enumerate(zip(datasets, labels)):
        for curve in data:
            cycles = np.arange(len(curve))
            plt.plot(cycles, curve, label=label, alpha=0.7, color = colors[idx])

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[i], lw=4, label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles)
    
    plt.xlabel('Number of Sampling Cycles')
    plt.ylabel(title)
    plt.title(f'{title} over Sampling Cycles')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + f'/{title.replace(" ", "_").lower()}.png')
    plt.show()



import numpy as np
def run_experiments(run_1=True, run_2=True, run_3=True, run_4=True, run_5=True, run_6=True, run_7=True, run_8= True, run_9=True, run_10=True, run_11=True, runs=5, cycles=10, samples_per_cycle=1, init_size=10, folder_path='runs/folder_unspecified'):
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Set random seed for reproducibility
    torch.manual_seed(42)
    results = {"pearson": [], "spearman": [], "r_squared": []}
    labels = []

    def update_results(results, new_results, label):
        pearson_corrs, spearman_corrs, rsquared_vals = new_results[:3]
        if label not in labels:
            labels.append(label)
            results['pearson'].append([])
            results['spearman'].append([])
            results['r_squared'].append([])

        index = labels.index(label)
        results['pearson'][index].append(pearson_corrs)
        results['spearman'][index].append(spearman_corrs)
        results['r_squared'][index].append(rsquared_vals)

    # Running the experiments
    for j in range(runs):
        if j==0:
            plot_corr = True
        if run_1:
            print("Running exp1")
            results1 = get_uncertain(embeddings, targets, cycles, samples_per_cycle, True, folder_path, init_size, plot_corr)
            update_results(results, results1, "Uncertain Acquisition (Normalized)")

        if run_2:
            print("Running exp2")
            results2 = get_uncertain(embeddings, targets, cycles, samples_per_cycle, False, folder_path, init_size, plot_corr)
            update_results(results, results2, "Uncertain Acquisition")

        if run_3:
            print("Running exp3")
            results3 = get_random(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
            update_results(results, results3, "Random Acquisition")

        if run_4:
            print("Running exp4")
            results4 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'largest_mean', init_size, plot_corr)
            update_results(results, results4, "Diverse acquisition (largest mean)")
        
        if run_5:
            print("Running exp5")
            results5 = get_min_values(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
            update_results(results, results5, "Min Values Acquisition")

        if run_6:
            print("Running exp6")
            results6 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'largest_mean', init_size, plot_corr)
            update_results(results, results6, "Diverse embeddings acquisition")
        if run_7:
            # results7 = get_pde(embeddings, targets, cycles, samples_per_cycle, folder_path, plot_corr)
            results7 = get_fast_emoc(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
            # results7 = get_best_emoc(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
            update_results(results, results7, "EMOC approx acquisition")

        print("Calculating correlations")

        if run_8:
            print("Running exp8")
            results8 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
            update_results(results, results8, "Diverse acquisition (min linkage)")

        if run_9:
            print("Running exp9")
            # results9 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'max_linkage', 10, plot_corr)
            # update_results(results, results9, "Diverse acquisition (max linkage)")
            # results9 = get_diverse2(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
            results9 = get_sparse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
            update_results(results, results9, "Sparse acquisition")
        
        if run_10:
            print("Running exp10")
            results10 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
            update_results(results, results10, "Diverse embeddings acquisition (min linkage)")
        
        if run_11:
            print("Running exp11")
            results11 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'max_linkage', init_size, plot_corr)
            update_results(results, results11, "Diverse embeddings acquisition (max linkage)")

    print("Done Runnning the experiments")   
    # Save and plot results
    print("Saving results")
    for key in results:
        for i, data in enumerate(results[key]):
            np.save(f'{folder_path}/all_{key}_{i+1}.npy', np.array(data))

    plot_indiv_results(results['pearson'], labels, 'All Pearson Correlation', folder_path)
    plot_indiv_results(results['spearman'], labels, 'All Spearman Correlation', folder_path)
    plot_indiv_results(results['r_squared'], labels, ' All R-squared Values', folder_path)

    print("Plotting results")
    plot_results(results['pearson'], labels, 'Pearson Correlation', folder_path)
    plot_results(results['spearman'], labels, 'Spearman Correlation', folder_path)
    plot_results(results['r_squared'], labels, 'R-squared Values', folder_path)
    print("Done plotting")


cycles = 60
samples_per_cycle = 1
runs = 4
init_size=30

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}_al_gp'

folder_path = f'runs/{file_names}_reg_lin_{dataset_path}_{antibody}_{cycles}cycles_{samples_per_cycle}samples_per_cycle_{runs}runs_{init_size}init_size'
print("Folder path", folder_path)


# 2,3,4,5,8,9,10
# run_experiments(run_1=False, run_2=True, run_3=True, run_4=True, run_5=True, run_6=False, run_7=False, run_8=True, run_9=True, run_10=True, run_11=False


run_experiments(run_1=False, run_2=False, run_3=True, run_4=True, run_5=False, run_6=False, run_7=False, run_8=True, run_9=False, run_10=True, run_11=False, runs=runs, cycles=cycles, samples_per_cycle=samples_per_cycle, init_size=init_size, folder_path=folder_path)