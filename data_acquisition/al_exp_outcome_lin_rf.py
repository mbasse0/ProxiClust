
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
from tqdm import tqdm
from sklearn.gaussian_process.kernels import RationalQuadratic
import random
import json
from goppy import OnlineGP, SquaredExponentialKernel
from sklearn.decomposition import PCA
from plot_predictions_vs_actuals import plot_predictions_vs_actuals
from scipy.special import erf
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import multiprocessing
import time
from sklearn.ensemble import RandomForestRegressor


start_time = time.time()




# Current date and time
now = datetime.datetime.now()

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
# dataset_path = 'df_desai_new_full.csv'
dataset_path = "./datasets/df_desai_old_full.csv"

dataset_name = dataset_path.split(".")[0]
# dataset_path = 'df_Desai_15loci_complete.csv'

print(dataset_path)
df = pd.read_csv(dataset_path)

# print("length of the whole Desai new dataset", len(df))
# TEMPORARY crop the dataset for faster runtime
# df = df.sample(n=3000, random_state=42)
df.drop(columns=['Unnamed: 0'])


# Assuming the first column is sequences and 'log10Kd_ACE2' is the target
sequences = df["mutant_sequence"].tolist()
# targets_original = df['new_log10Kd_REGN10987'].values
antibody = 'log10Kd_ACE2'
# antibody = 'log10Kd_AZD1061'
# antibody = 'log10Kd_S309'
# antibody = "new_log10Kd_REGN10987"

print("antibody", antibody)
targets_original = df[antibody].values

elapsed_time = time.time() - start_time

# Display the elapsed time
print(f"Time taken to load the dataset: {elapsed_time:.2f} seconds")

# full_embeddings = torch.tensor(df['mean_representation'].apply(json.loads))
df["onehot"] = df["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
df["mean_representation"] = df["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])
full_embeddings = torch.tensor(df["mean_representation"].tolist())

pca = PCA(n_components=20)
embeddings = pca.fit_transform(full_embeddings)

print(f"Computed embeddings dim 20, {time.time() - start_time:.2f}")

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
from sklearn.cluster import DBSCAN
import seaborn as sns


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



def plot_tsne_clusters(sequences, initial_idx, labelled_idx, unlabelled_idx, save_path=None):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(sequences)
    
    # Perform DBSCAN clustering on the t-SNE results
    # epsilon = 1.5
    # min_samples = 8
    min_samples= 5
    epsilon=1
    print("Min samples in DBSCAN", min_samples)
    print("epsilon in DBSCAN", epsilon)
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(tsne_results)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    
    # Plot all points with cluster coloring
    palette = sns.color_palette("hsv", len(set(clusters)))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=clusters, palette=palette, legend='full', alpha=0.5)
    
    # Highlight unlabelled points
    # unlabelled_points = tsne_results[unlabelled_idx]
    # plt.scatter(unlabelled_points[:, 0], unlabelled_points[:, 1], color='yellow', label='Unlabelled sequences', edgecolors='black')
    
    # Highlight labelled points
    labelled_points = tsne_results[labelled_idx]
    plt.scatter(labelled_points[:, 0], labelled_points[:, 1], color='blue', label='Labelled sequences', edgecolors='black')

    # Highlight the initial points
    initial_points = tsne_results[initial_idx]
    plt.scatter(initial_points[:, 0], initial_points[:, 1], color='red', label='Initial points', edgecolors='black', s=100)
        
    # Add legend
    plt.legend()
    
    # Add title
    plt.title('t-SNE of Sequences Dataset with DBSCAN Clustering')
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return clusters


def get_from_tsne(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    tsne_clusters = plot_tsne_clusters(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/init_tsne.png')
    print("Tsne clusters", tsne_clusters)
    print("num clusters", len(set(tsne_clusters)))
    print("num of elements", len(tsne_clusters))

    # Randomly arrange range(num_clusters) 
    cluster_order = np.random.permutation(len(set(tsne_clusters)))

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
    
    for j in range(cycles):
        gp.add(new_seq, new_targets)
        # Choose a random point in cluster j
        new_idx = np.random.choice([i for i in range(len(sequences)) if tsne_clusters[i] == cluster_order[j%len(cluster_order)]], samples_per_cycle, replace=False)
        print("Cluster for this cycle", cluster_order[j])
        print("number of elements in this cluster", len([i for i in range(len(sequences)) if tsne_clusters[i] == cluster_order[j%len(cluster_order)]]))
        print("New idx", new_idx)

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
    
    print("Train data after all cycles", len(X_train))
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_random_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/from_tsne_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx


def get_random(sequences, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    step_size = 10
    while len(labelled_idx) < cycles + init_size:
        # Select the next batch of samples
        if len(unlabelled_idx) > step_size:
            new_idx = np.random.choice(unlabelled_idx, step_size, replace=False)
        else:
            new_idx = unlabelled_idx  # if fewer than 10 samples remain, take them all

        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

        # Train the model
        X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
        y_train = np.array([targets[i] for i in labelled_idx])
        model.fit(X_train, y_train)

        # Evaluate the model
        remaining_targets = [targets[i] for i in unlabelled_idx]
        predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        rsquared_vals.append(r_squared)

        print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")
        
        if plot_corr:
            plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_random_{len(X_train)}.png')

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

    step_size = 5
    for j in range(cycles):
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
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

        if (j + 1) % (step_size) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx

def get_diverse_embeddings(sequences, targets, cycles, samples_per_cycle, folder_path, selection_criterion='min_linkage', init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    for j in range(cycles):
        # Calculate distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences[i] for i in labelled_idx]

        if selection_criterion == 'largest_mean':
            distances = [np.mean([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'min_linkage':
            distances = [np.min([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'max_linkage':
            distances = [np.max([np.linalg.norm(sequences[idx] - seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))

        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

        if (j + 1) % (10) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

            if plot_corr:
                plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_diverse_embedding_{selection_criterion}_cycle_{j}.png')

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx


def get_diverse_embeddings_cosine(sequences, targets, cycles, samples_per_cycle, folder_path, selection_criterion='largest_mean', init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    
    for j in range(cycles):
        # Calculate mean cosine distances from each unlabelled sequence to the labelled sequences
        labelled_sequences = [sequences[i] for i in labelled_idx]
        
        # Find the indices of the unlabelled sequences with the largest mean distances
        if selection_criterion == 'largest_mean':
            distances = [np.mean([cosine(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'min_linkage':
            # Get the highest minimum distance for min linkage
            distances = [np.min([cosine(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))
        elif selection_criterion == 'max_linkage':
            # Get the highest maximum distance for max linkage
            distances = [np.max([cosine(sequences[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
            sorted_indices = np.argsort(-np.array(distances))

        new_idx = [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]
        labelled_idx.extend(new_idx)
    
        if (j + 1) % (10) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx



def get_subset_combinatorial(df, indices, from_variant='wt'):
    if dataset_name=='df_desai_new_full':
        num_loci=11
    else:
        num_loci=15
    if from_variant=='wt':
        return df[df["onehot"].apply(lambda x : not any(x[ind] for ind in range(num_loci) if ind not in indices))]
    elif from_variant=='omicron':
       return df[df["onehot"].apply(lambda x : not any( x[ind] for ind in range(num_loci) if ind not in indices))] 
    else:
        print("Wrong origin variant")

def get_from_epistasis_clusters(sequences, targets, clusters_generators, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    clusters = {}
    cluster_weights = {}
    for i, cluster in enumerate(clusters_generators):
        print("Cluster for this iter", cluster)
        comb_idx = get_subset_combinatorial(df, cluster).index.tolist()
        clusters[i] = comb_idx
        cluster_weights[i] = len(comb_idx)
    
    print("clusters", clusters)
    print("cluster_weights", cluster_weights)
    total_weight = sum(cluster_weights.values())
    cluster_weights = {cluster: weight/total_weight for cluster, weight in cluster_weights.items()}
    print("cluster_weights norm", cluster_weights)
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
        print("Cycle", j)
        gp.add(new_seq, new_targets)
        # Choose a cluster based on its weight and new_idx is chosen randomly in that cluster
        cluster = np.random.choice(list(cluster_weights.keys()), p=list(cluster_weights.values()))
        print("Chosen cluster", cluster)
        new_idx = np.random.choice(clusters[cluster], samples_per_cycle, replace=False)
        while new_idx in labelled_idx:
            new_idx = np.random.choice(clusters[cluster], samples_per_cycle, replace=False)
                
        # new_idx = np.random.choice(clusters[cluster], samples_per_cycle, replace=False)
        print("Chosen index", new_idx)
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
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_epistasis_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/epistasis_tsne_{j}.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, gp, labelled_idx, unlabelled_idx



def get_ucb(sequences, sequences_str, targets, clusters, cluster_weights, alpha, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    
   
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    for k in range(cycles):        
        labelled_sequences = [sequences_str[i] for i in labelled_idx]
        unlabeled_sequences = [sequences_str[i] for i in unlabelled_idx]

        labelled_emb = np.array([sequences[i] for i in labelled_idx])
        unlabelled_emb = np.array([sequences[i] for i in unlabelled_idx])
        scores = {}

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(labelled_emb)
        log_dens = kde.score_samples(unlabelled_emb)
        for i in range(len(log_dens)):
            scores[unlabelled_idx[i]] = -log_dens[i]

        # Get the highest minimum distance for min linkage
        distances = [np.min([hamming_distance(sequences_str[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
        for j, idx in enumerate(unlabelled_idx):
            add_score = alpha*distances[j]
            if idx in scores:   
                scores[idx] += add_score

            else:
                scores[idx] = add_score


        sorted_indices = sorted(unlabelled_idx, key=lambda x: scores[x], reverse=True)
       
        new_idx = [sorted_indices[i] for i in range(samples_per_cycle)]
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

        if (j + 1) % (10) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx



def get_ucb_auto_weight(sequences, sequences_str, targets, clusters, cluster_weights, alpha, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):

    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    new_seq = np.array([sequences[i] for i in init_indices])
    new_targets = np.expand_dims([targets[i] for i in init_indices], axis=1)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    

    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    for k in range(cycles):
        
        labelled_sequences = [sequences_str[i] for i in labelled_idx]
        scores = {}
        for cluster in clusters:
            for el in clusters[cluster]:
                scores[el] = cluster_weights[cluster]            
        
        # Get the highest minimum distance for min linkage
        distances = [np.min([hamming_distance(sequences_str[idx], seq) for seq in labelled_sequences]) for idx in unlabelled_idx]
        # Normalize the distances between 0 and 1
        distances = (distances - np.min(distances))/(np.max(distances) - np.min(distances))

        for j, idx in enumerate(unlabelled_idx):
            add_score = distances[j]
            if idx in scores:   
                scores[idx] += add_score

            else:
                scores[idx] = add_score


        sorted_indices = sorted(unlabelled_idx, key=lambda x: scores[x], reverse=True)
       
        new_idx = [sorted_indices[i] for i in range(samples_per_cycle)]
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    
    X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
    y_train = np.array([targets[i] for i in labelled_idx])
    model.fit(X_train, y_train)


    remaining_seq = [sequences[i] for i in unlabelled_idx]
    remaining_targets = [targets[i] for i in unlabelled_idx]

    predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
    pearson_corrs.append(pearson_corr)
    spearman_corrs.append(spearman_corr)
    rsquared_vals.append(r_squared)
    
    print("Train data after all cycles", len(X_train))
    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_usb.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/ucb_auto_weight_tsne.png')
    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx



def plot_results(datasets, title, folder_path):
    plt.figure(figsize=(10, 6))
    for data in datasets:
        mean_values = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        cycles = np.arange(len(mean_values))
        plt.errorbar(cycles, mean_values, yerr=std_dev, fmt='-o', alpha=0.7)

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
#     cycles = benchmark_mean

#     # Plot the benchmark dataset
#     plt.errorbar(cycles, benchmark_mean, yerr=benchmark_std, fmt='-o', label=labels[0], alpha=0.7, color='black')

#     # Plot other datasets relative to the benchmark
#     for data, label in zip(datasets[1:], labels[1:]):
#         mean_values = np.mean(data, axis=0)
#         std_dev = np.std(data, axis=0)
#         diff_mean = mean_values
#         diff_std = np.sqrt(std_dev**2 + benchmark_std**2)  # Combining standard deviations

#         plt.errorbar(cycles, diff_mean, yerr=diff_std, fmt='-o', label=label, alpha=0.7)

#     plt.xlabel('R² of random acquisition strategy (benchmark)')
#     plt.ylabel(f'{title} R² of strategy')
#     plt.title(f'{title} R² for strategy compared to R² for random benchmark for {antibody} ({dataset_path} dataset)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(folder_path + f'/{title.replace(" ", "_").lower()}_relative_to_benchmark.png')
#     plt.show()

from sklearn.neighbors import KernelDensity


def sample_sparse(sequences, unlabelled_idx, labelled_idx, samples_per_cycle):
    # Get a sequence in unlabelled sequence in the most sparse region. Estimate density with a kernel density estimator
    unlabelled_sequences = np.array([sequences[i] for i in unlabelled_idx])
    labelled_sequences = np.array([sequences[i] for i in labelled_idx])

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(labelled_sequences)
    log_dens = kde.score_samples(unlabelled_sequences)
    sorted_indices = np.argsort(log_dens)
    return [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]

def sample_dense(sequences, unlabelled_idx, labelled_idx, samples_per_cycle):
    # Get a sequence in unlabelled sequence in the most sparse region. Estimate density with a kernel density estimator
    unlabelled_sequences = np.array([sequences[i] for i in unlabelled_idx])
    labelled_sequences = np.array([sequences[i] for i in labelled_idx])

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(labelled_sequences)
    log_dens = kde.score_samples(unlabelled_sequences)
    sorted_indices = np.argsort(-log_dens)
    return [unlabelled_idx[i] for i in sorted_indices[:samples_per_cycle]]

def sample_sparse_away(sequences, unlabelled_idx, labelled_seq, samples_per_cycle):
    # Convert sequences to NumPy arrays for distance calculation
    unlabelled_sequences = np.array([sequences[i] for i in unlabelled_idx])
    labelled_sequences = np.array([sequences[i] for i in labelled_idx])
    
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
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    
    for j in range(cycles):
        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        # labelled_sequences = [sequences_str[i] for i in labelled_idx]
        new_idx = sample_sparse(sequences, unlabelled_idx, labelled_idx, samples_per_cycle)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]

        if (j + 1) % (10) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

    print("Train data after all cycles", len(X_train))

    # if plot_corr:
    #     plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_sparse_cycle_{j}.png')
    #     plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/sparse_tsne_cycle_{j}.png')

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx

    

def get_dense(sequences, sequences_str, targets, cycles, samples_per_cycle, folder_path, init_size=10, plot_corr=False):
    init_indices = np.random.choice(len(sequences), size=init_size, replace=False)
    labelled_idx = list(init_indices)
    unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
    pearson_corrs = []
    spearman_corrs = []
    rsquared_vals = []
    
    for j in range(cycles):
        # Calculate mean Hamming distances from each unlabelled sequence to the labelled sequences
        # labelled_sequences = [sequences_str[i] for i in labelled_idx]
        new_idx = sample_dense(sequences, unlabelled_idx, labelled_idx, samples_per_cycle)
        labelled_idx.extend(new_idx)
        unlabelled_idx = [i for i in range(len(sequences)) if i not in labelled_idx]
        if (j + 1) % (10) == 0 or j == cycles - 1:
            X_train = np.array([df.iloc[i]['onehot'] for i in labelled_idx])
            y_train = np.array([targets[i] for i in labelled_idx])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            remaining_targets = [targets[i] for i in unlabelled_idx]

            predictions, actuals, pearson_corr, spearman_corr, r_squared = evaluate_model(model, unlabelled_idx, remaining_targets)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
            rsquared_vals.append(r_squared)
            print(f"Train data size: {len(X_train)}; Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}, R²: {r_squared:.2f}")

    print("Train data after all cycles", len(X_train))

    if plot_corr:
        plot_predictions_vs_actuals(predictions, actuals, pearson_corr, spearman_corr, r_squared, f'{folder_path}/acquisition_dense_cycle_{j}.png')
        plot_tsne(sequences, init_indices, labelled_idx, unlabelled_idx, save_path=f'{folder_path}/dense_tsne_cycle_{j}.png')

    return pearson_corrs, spearman_corrs, rsquared_vals, model, labelled_idx, unlabelled_idx

    


def plot_indiv_results(datasets, title, folder_path):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))  # Using the Viridis colormap

    for idx, data in enumerate(datasets):
        for curve in data:
            cycles = np.arange(len(curve))
            plt.plot(cycles, curve, alpha=0.7, color = colors[idx])

    from matplotlib.lines import Line2D
    # handles = [Line2D([0], [0], color=colors[i], lw=4, label=labels[i]) for i in range(len(labels))]
    # plt.legend(handles=handles)
    
    plt.xlabel('Number of Sampling Cycles')
    plt.ylabel(title)
    plt.title(f'{title} over Sampling Cycles')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + f'/{title.replace(" ", "_").lower()}.png')
    plt.show()



import numpy as np
def run_experiments(args):
    run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9, run_10, run_11, cycles, samples_per_cycle, init_size, folder_path, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Check if the directory exists, and if not, create it
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        pass
    # Set random seed for reproducibility
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




    # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
    cluster_generators = [[3, 4, 5, 2], [9, 7, 8, 6], [0, 1, 2, 3]]
    clusters = {}
    cluster_weights = {}
    for i, cluster in enumerate(cluster_generators):
        # clusters[i] = get_subset_combinatorial(df, cluster).index.tolist()
        comb_idx = get_subset_combinatorial(df, cluster).index.tolist()
        clusters[i] = comb_idx
        print("Cluster i assigned to", clusters[i])
        cluster_weights[i] = len(comb_idx)
    
    # Assign weights to each cluster based on the number of sequences in each cluster
    total_weight = sum(cluster_weights.values())

    cluster_weights = {cluster: weight/total_weight for cluster, weight in cluster_weights.items()}
    plot_corr = True
    # Running the experiments
    if run_1:
        print("Running exp1")
        results1 = get_uncertain(embeddings, targets, cycles, samples_per_cycle, True, folder_path, init_size, plot_corr)
        update_results(results, results1, "Uncertain Acquisition (Normalized)")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_2:
        print("Running exp2")
        # results2 = get_uncertain(embeddings, targets, cycles, samples_per_cycle, False, folder_path, init_size, plot_corr)
        # update_results(results, results2, "Uncertain Acquisition")
        # print(f"Time is , {time.time() - start_time:.2f}")

        results2 = get_dense(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results2, "Dense acquisition")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_3:
        print("Running exp3")
        results3 = get_random(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results3, "Random Acquisition")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_4:
        print("Running exp4")
        # results4 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'largest_mean', init_size, plot_corr)
        # update_results(results, results4, "Diverse acquisition (largest mean)")
        results4 = get_from_tsne(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results4, "t-SNE Acquisition")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_5:
        print("Running exp5")
        # results5 = get_min_values(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # update_results(results, results5, "Min Values Acquisition")
        results5 = get_diverse_embeddings_cosine(embeddings, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        update_results(results, results5, "Diverse embeddings cosine dissimilarity acquisition (max linkage dissimilarity) ")
        # alpha = 0.05
        # # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
        # print(cluster_generators)
        # results5 = get_ucb(embeddings, sequences, targets, clusters, cluster_weights, alpha, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # update_results(results, results5, f"UCB acquisition (alpha = {alpha}) ")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_6:
        print("Running exp6")
        # results6 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'largest_mean', init_size, plot_corr)
        # update_results(results, results6, "Diverse embeddings acquisition")
        # results6 = get_diverse_embeddings_cosine(embeddings, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        # update_results(results, results6, "Diverse embeddings cosine dissimilarity acquisition (min linkage dissimilarity) ")
        alpha = 1.0
        # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
        print(cluster_generators)
        results6 = get_ucb(embeddings, sequences, targets, clusters, cluster_weights, alpha, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results6, f"UCB acquisition (alpha = {alpha}) ")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_7:
        print("Running exp7")

        # results7 = get_pde(embeddings, targets, cycles, samples_per_cycle, folder_path, plot_corr)
        # results7 = get_fast_emoc(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # results7 = get_best_emoc(embeddings, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # cluster_generators = [[3, 4, 7, 9, 5], [6, 2, 0, 5, 1], [0, 1, 6, 2, 5]]
        # cluster_generators = [[2, 3, 1, 4, 5, 0, 6, 7, 8], [11, 10, 8, 9, 7, 12, 13, 6, 14]]
        # cluster_generators = [list(range(15))]
        # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
        cluster_generators = [[5, 4, 6, 3, 7], [12, 11, 13, 10, 14], [2, 3, 1, 4, 0], [10, 9, 11, 7, 6], [0, 1, 2, 3, 4], [7, 6, 5, 8, 10], [9, 8, 10, 7, 6], [13, 14, 12, 11, 10]]
        print("Cluster generators", cluster_generators)
        results7 = get_from_epistasis_clusters(embeddings, targets, cluster_generators, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results7, "Epistasis cluster exploitation")
        print(f"Time is , {time.time() - start_time:.2f}")

    print("Calculating correlations")

    if run_8:
        print("Running exp8")
        # results8 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        # update_results(results, results8, "Diverse acquisition (min linkage)")
        alpha = 0.1
        # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
        print(cluster_generators)
        results8 = get_ucb(embeddings, sequences, targets, clusters, cluster_weights,  alpha, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results8, f"UCB acquisition (alpha = {alpha}) ")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_9:
        print("Running exp9")
        # results9 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'max_linkage', 10, plot_corr)
        # update_results(results, results9, "Diverse acquisition (max linkage)")
        # results9 = get_diverse2(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        results9 = get_sparse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        update_results(results, results9, "Sparse acquisition")
        # alpha = 0.01
        # # cluster_generators=[[4, 5, 3, 6, 2, 7, 8, 1], [12, 11, 13, 10, 14, 7, 9, 8], [1, 2, 0, 3, 4, 5, 6, 7], [9, 7, 8, 6, 10, 11, 5, 4], [14, 13, 12, 11, 10, 8, 9, 7]]
        # print(cluster_generators)
        # results9 = get_ucb(embeddings, sequences, targets, clusters, cluster_weights,  alpha, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # update_results(results, results9, f"UCB acquisition (alpha = {alpha}) ")
        print(f"Time is , {time.time() - start_time:.2f}")

    if run_10:
        print("Running exp10")
        results10 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        update_results(results, results10, "Diverse embeddings acquisition (min linkage)")
        # alpha = 10
        # results10 = get_ucb(embeddings, sequences, targets, clusters, cluster_weights, alpha, cycles, samples_per_cycle, folder_path, init_size, plot_corr)
        # update_results(results, results10, f"UCB auto weight acquisition (alpha = {alpha}) ")
        print(f"Time is , {time.time() - start_time:.2f}")
    if run_11:
        print("Running exp11")
        # results11 = get_diverse_embeddings(embeddings, targets, cycles, samples_per_cycle, folder_path, 'max_linkage', init_size, plot_corr)
        # results11 = get_diverse_embeddings_cosine(embeddings, targets, cycles, samples_per_cycle, folder_path, 'largest_mean', init_size, plot_corr)
        # update_results(results, results11, "Diverse embeddings cosine dissimilarity acquisition (max mean dissimilarity) ")

        results11 = get_diverse(embeddings, sequences, targets, cycles, samples_per_cycle, folder_path, 'min_linkage', init_size, plot_corr)
        update_results(results, results11, "Diverse acquisition (min linkage)")
        print(f"Time is , {time.time() - start_time:.2f}")

    print("Done Runnning one set of experiments")   
    return results, labels
    


cycles = 190
samples_per_cycle = 1
runs = 112
init_size=10


# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}_al_lin_outcome_rf'

folder_path = f'./runs/{file_names}_{dataset_name}_{antibody}_{cycles}cycles_{samples_per_cycle}samples_per_cycle_{runs}runs_{init_size}init_size'
print("Folder path", folder_path)

print("GOod ACQ COMP FOR 190 ")

def run_parallel_experiments(run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9, run_10, run_11, runs=5, cycles=10, samples_per_cycle=1, init_size=10, folder_path='runs/folder_unspecified'):
    np.random.seed()  # Ensure the main process has a unique seed
    seeds = np.random.randint(0, 2**32 - 1, size=runs)  # Generate unique seeds for each run
    tasks = [(run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9, run_10, run_11, cycles, samples_per_cycle, init_size, folder_path, seeds[i]) for i in range(runs)]
    num_process = 112
    with Pool(processes=num_process) as pool:  # Adjust the number of processes as needed
            all_results = pool.map(run_experiments, tasks)
        
    # all_results  = []
    # for k in range(runs):
    #     all_results.append(run_single_experiment(k))
    return all_results




results = run_parallel_experiments(run_1=False, run_2=False, run_3=True, run_4=False, run_5=True, run_6=False, run_7=False, run_8=False, run_9=True, run_10=True, run_11=True, runs=runs, cycles=cycles, samples_per_cycle=samples_per_cycle, init_size=init_size, folder_path=folder_path)
print("All results", results)

# Obtain the strategy names from the results
strategies = results[0][1]

# Initialize dictionaries to hold mean and std values
r2_means = {strategy: [] for strategy in strategies}
r2_stds = {strategy: [] for strategy in strategies}

# Extract R² scores and compute mean and std
for strategy_idx, strategy in enumerate(strategies):
    all_r2_scores = []
    for result in results:
        strategy_r2_scores = result[0]['r_squared'][strategy_idx]
        all_r2_scores.append(strategy_r2_scores[0])
    # Convert to numpy array for easier calculation
    all_r2_scores = np.array(all_r2_scores)
    r2_means[strategy] = np.mean(all_r2_scores, axis=0)
    r2_stds[strategy] = np.std(all_r2_scores, axis=0)

np.save(f'{folder_path}/{file_names}_{dataset_name}_{antibody}_{cycles}cycles_{samples_per_cycle}samples_per_cycle_{runs}runs_{init_size}init_size_r2_means.npy', r2_means)
np.save(f'{folder_path}/{file_names}_{dataset_name}_{antibody}_{cycles}cycles_{samples_per_cycle}samples_per_cycle_{runs}runs_{init_size}init_size_r2_stds.npy', r2_stds)

print("Done saving the files")

# plot_results(results['pearson'], labels, 'Pearson Correlation', folder_path)
# plot_results(results['spearman'], labels, 'Spearman Correlation', folder_path)
# plot_results(results['r_squared'], labels, 'R-squared Values', folder_path)