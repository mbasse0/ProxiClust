import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from plot_regression import plot_regression
import random 
import seaborn as sns
from itertools import combinations
import os
import datetime

# Current date and time
now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")

print("Timestamp", timestamp)
file_names = f'{timestamp}'

folder_path = f'runs/{file_names}'
# Check if the directory exists, and if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Load the dataframe

dataset_path = 'df_desai_old_full.csv'
print(dataset_path)
df_sorted = pd.read_csv(dataset_path)

df_sorted["onehot"] = df_sorted["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


# antibody = "log10Kd_REGN10987"
# antibody = 'log10Kd_S309'
antibody = "log10Kd_ACE2"

print(antibody)
targets_sorted = df_sorted[antibody].to_list()
## Load the pdb file
# file_path = "6m0j.pdb"
# file_path = "7xsw.pdb"
# file_path = "8vyg.pdb"
file_path = "8j26.pdb"


ppdb = PandasPdb().read_pdb(file_path)
ppdb


"""
atom_df = ppdb.df["ATOM"]
CA_df = atom_df[atom_df["atom_name"] == "CA"]

residues_chainA = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
residues_chainE = CA_df[CA_df["chain_id"]=='E'].sort_values("residue_number")
residues_chainE = residues_chainE[residues_chainE["alt_loc"] != 'A']

aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

residues_chainA["single_letter_residue"] = residues_chainA["residue_name"].map(aa_dict)
residues_chainE["single_letter_residue"] = residues_chainE["residue_name"].map(aa_dict)

amino_acid_sequenceA = ''.join(residues_chainA["single_letter_residue"].values)
amino_acid_sequenceE = ''.join(residues_chainE["single_letter_residue"].values)
"""


# For S309


atom_df = ppdb.df["ATOM"]
CA_df = atom_df[atom_df["atom_name"] == "CA"]

residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])

residues_chain_RBD = CA_df[CA_df["chain_id"]=='C'].sort_values("residue_number")
# residues_chainE = residues_chainE[residues_chainE["alt_loc"] != 'A']

aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


# S309 : H,A // L,B
residues_chain_ab["single_letter_residue"] = residues_chain_ab["residue_name"].map(aa_dict)
residues_chain_RBD["single_letter_residue"] = residues_chain_RBD["residue_name"].map(aa_dict)



amino_acid_sequence_ab = ''.join(residues_chain_ab["single_letter_residue"].values)
amino_acid_sequence_RBD = ''.join(residues_chain_RBD["single_letter_residue"].values)



# Find residues in E that are closest to the interface
def find_interface_residues(pos1, pos2, num_residues):
    distances = cdist(pos1, pos2)
    return np.argsort(np.min(distances, axis=1))[:num_residues]

### Kmeans with 3D positions
# Initialize Cluster Centers with KMeans++

def proximity_clustering(positions, n_clusters, k=5):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_

    # Use Nearest Neighbors to Define Groups
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(positions)

    # For each center, find the k nearest neighbors
    groups = {}
    group_indices = {}
    for i, center in enumerate(centers):
        distances, indices = neighbors.kneighbors([center]) 
        group_indices[i] = indices[0]
        groups[i] = positions[indices[0]]  

    return group_indices, groups
 

def kmeans_clustering(positions, n_clusters=2):

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    indices_list = []
    for i in range(n_clusters):
        indices = [index for index, label in enumerate(labels) if label == i]
        indices_list.append(indices)

    return indices_list, centers

def even_kmeans_clustering(positions, n_clusters=2, samples_per_exp=5):

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_

    labels = kmeans.labels_
    indices_list = []
    for i in range(n_clusters):
        indices = [index for index, label in enumerate(labels) if label == i]

        # Choose samples_per_exp elements of indices if it is larger than samples_per_exp
        if len(indices) > samples_per_exp:
            indices = random.sample(indices, samples_per_exp)

        indices_list.append(indices)


    return indices_list, centers


## Testing the strategies


def get_subset_combinatorial(df, indices, from_variant='wt'):
    num_loci=15
    if from_variant=='wt':
        return df[df["onehot"].apply(lambda x : not any(x[ind] for ind in range(num_loci) if ind not in indices))]
    elif from_variant=='omicron':
       return df[df["onehot"].apply(lambda x : not any( x[ind] for ind in range(num_loci) if ind not in indices))] 
    else:
        print("Wrong origin variant")


def random_benchmark(df, n_exp, samples_per_exp, iterations, model_type='linear', use_onehot=True):
    r2s = []
    overlaps = []
    test_set = df

    for iter in range(iterations):
        indices_list = [np.random.choice(range(15), size=samples_per_exp, replace=False) for _ in range(n_exp)]

        # # Calculate overlaps
        # index_counts = {}
        # for indices in indices_list:
        #     for index in indices:
        #         if index in index_counts:
        #             index_counts[index] += 1
        #         else:
        #             index_counts[index] = 1
        
        # # Count the overlaps (indices appearing in more than one group)
        # overlap_count = sum(count > 1 for count in index_counts.values())
        # overlaps.append(overlap_count)
        # print(f"Overlap count: {overlap_count}")

        training_set = None
        for indices in indices_list:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        r2 = plot_regression(training_set, test_set, "custom split", antibody, "onehot", model_type, show=False, use_onehot=use_onehot)
        r2s.append(r2)
    return r2s


def exhaustive_benchmark(df, test_set, n_exp, samples_per_exp, mut_pos_desai, model_type='linear', use_onehot=True):
    r2s = []
    overlaps = []
    all_combinations = list(combinations(mut_pos_desai, samples_per_exp))
    num_comb = len(all_combinations)
    all_choices = list(combinations(range(num_comb), n_exp))
    indices_lists = [ [all_combinations[el] for el in choice ]for choice in all_choices]
    
    

    print(f"Number of indices list to test: {len(indices_lists)}")
    for i, indices_list in enumerate(indices_lists):
        if i%100 == 0:
            print("Iteration", i , " out of ", len(indices_lists))

        # # Calculate overlaps
        # index_counts = {}
        # for indices in indices_list:
        #     for index in indices:
        #         if index in index_counts:
        #             index_counts[index] += 1
        #         else:
        #             index_counts[index] = 1
        
        # # Count the overlaps (indices appearing in more than one group)
        # overlap_count = sum(count > 1 for count in index_counts.values())
        # overlaps.append(overlap_count)
        # print(f"Overlap count: {overlap_count}")

        training_set = None
        for indices in indices_list:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        r2 = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", model_type, show=False, use_onehot=use_onehot)
        r2s.append(r2)
    return r2s

def compute_quantile(val, all_values):
    return sum(val > all_values)/len(all_values)

def run_multiple_benchmarks(df, n_loci=8, n_iter=2, n_exp=2, samples_per_exp=5):
    chosen_ind = []
    r2_lists = []
    r2_strat_list = []
    quantile_list = []
    test_ind = random.sample(range(len(df)), 1000)
    for k in range(n_iter):
        key_few_mut = random.sample(range(len(desai_mut_pos)), n_loci)
        chosen_ind.append(key_few_mut)
        print("Chosen indices", key_few_mut)
        df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in key_few_mut])
        subset_mut_pos = [desai_mut_pos[i] for i in key_few_mut] 
        desai_residues = residues_chain_RBD.iloc[subset_mut_pos]
        desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

        # Compute the R² for the proximity strategy
        ind, dist = proximity_clustering(desai_positions, n_exp, k=samples_per_exp)
        strategy_lists = [[key_few_mut[el] for el in ind] for ind in ind.values()]
        print("Proximal Strategy gives ", strategy_lists)
        training_set = None
        for indices in strategy_lists:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])

        test_set = df.iloc[test_ind]
        r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", "linear", show=False, use_onehot=True)
        print("R² for strategy", r2_strat)
        r2_strat_list.append(r2_strat)

        r2_exhaustive = exhaustive_benchmark(df, test_set, n_exp, samples_per_exp, key_few_mut, model_type='linear', use_onehot=True)
        r2_lists.append(r2_exhaustive)
        median_exhaustive = np.median(r2_exhaustive)
        print("Median of exhaustive", median_exhaustive)
        quantile_list.append(compute_quantile(r2_strat, r2_exhaustive))

        plt.clf()
        plt.boxplot(r2_exhaustive)
        plt.ylim(median_exhaustive - 1 , 1)
        plt.axhline(y=r2_strat, color='r', linestyle='--')
        plt.title(f"Benchmark for indices: {key_few_mut}")
        plt.savefig(folder_path+f"/benchmark_{key_few_mut}.png")

    np.save(folder_path+"/r2_lists.npy", r2_lists)
    np.save(folder_path+"/chosen_indices.npy", chosen_ind)
    np.save(folder_path+"/r2_strat_list.npy", r2_strat_list)
    print("Average increase in R² comapared to median value", np.mean([r2_strat_list[i] - np.median(r2_lists[i]) for i in range(len(r2_strat_list))]))
    print("Average increase in R² comapared to median value, for max values > 0", np.mean([r2_strat_list[i] - np.median(r2_lists[i]) for i in range(len(r2_strat_list)) if np.max(r2_lists[i]) > 0]))
    print("Average quantile", np.mean(quantile_list))
    print("Average quantile for max values > 0", np.mean([quantile_list[i] for i in range(len(r2_strat_list)) if np.max(r2_lists[i]) > 0]))
    print("All quantiles", quantile_list)
    print("All quantiles when positive strategy exists", [quantile_list[i] for i in range(len(quantile_list)) if np.max(r2_lists[i])>0])
    print("Number of positive strategies", len([i for i in range(len(quantile_list)) if np.max(r2_lists[i])>0]))



def run_10choose8_benchmarks(df, n_loci=8, n_exp=2, samples_per_exp=5):
    chosen_ind = []
    r2_lists = []
    r2_strat_list = []
    quantile_list = []
    test_ind = random.sample(range(len(df)), 1000)

    # indices_pool = [13, 14, 11,  7,  6, 10, 12,  8,  4,  9]
    indices_pool = [ 0,  5,  6,  2, 12,  1, 13,  11,  3, 14]
    few_mut_list = list(combinations(indices_pool, n_loci))
    for k in range(len(few_mut_list)):
        key_few_mut = few_mut_list[k]
        chosen_ind.append(key_few_mut)
        print("Chosen indices", key_few_mut)
        df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in key_few_mut])
        subset_mut_pos = [desai_mut_pos[i] for i in key_few_mut] 
        desai_residues = residues_chain_RBD.iloc[subset_mut_pos]
        desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

        # Compute the R² for the proximity strategy
        # ind, dist = proximity_clustering(desai_positions, n_exp, k=samples_per_exp)
        # strategy_lists = [[key_few_mut[el] for el in ind] for ind in ind.values()]

        # strategy_lists, _ = even_kmeans_clustering(desai_positions, n_exp, samples_per_exp)
        strategy_lists, _ = kmeans_clustering(desai_positions, n_exp)
        print("kmeans Strategy gives ", strategy_lists)
        training_set = None
        for indices in strategy_lists:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        test_set = df.iloc[test_ind]
        r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", "linear", show=False, use_onehot=True)
        print("R² for strategy", r2_strat)
        r2_strat_list.append(r2_strat)

        r2_exhaustive = exhaustive_benchmark(df, test_set, n_exp, samples_per_exp, key_few_mut, model_type='linear', use_onehot=True)
        r2_lists.append(r2_exhaustive)
        median_exhaustive = np.median(r2_exhaustive)
        print("Median of exhaustive", median_exhaustive)
        quantile_list.append(compute_quantile(r2_strat, r2_exhaustive))

        plt.clf()
        plt.boxplot(r2_exhaustive)
        plt.ylim(median_exhaustive - 1 , 1)
        plt.axhline(y=r2_strat, color='r', linestyle='--')
        plt.title(f"Benchmark for indices: {key_few_mut}")
        plt.savefig(folder_path+f"/benchmark_{key_few_mut}.png")

    threshold_keep = 0.7
    np.save(folder_path+"/r2_lists.npy", r2_lists)
    np.save(folder_path+"/chosen_indices.npy", chosen_ind)
    np.save(folder_path+"/r2_strat_list.npy", r2_strat_list)
    print("Average increase in R² comapared to median value", np.mean([r2_strat_list[i] - np.median(r2_lists[i]) for i in range(len(r2_strat_list))]))
    print("Average increase in R² comapared to median value, for max values > ", threshold_keep, np.mean([r2_strat_list[i] - np.median(r2_lists[i]) for i in range(len(r2_strat_list)) if np.max(r2_lists[i]) > threshold_keep]))
    print("Average quantile", np.mean(quantile_list))
    print("Average quantile for max values > ", threshold_keep, np.mean([quantile_list[i] for i in range(len(r2_strat_list)) if np.max(r2_lists[i]) > threshold_keep]))
    print("All quantiles", quantile_list)
    print("All quantiles when positive strategy > ", threshold_keep, [quantile_list[i] for i in range(len(quantile_list)) if np.max(r2_lists[i])>threshold_keep])
    print("Number of positive strategies > ", threshold_keep, len([i for i in range(len(quantile_list)) if np.max(r2_lists[i])>threshold_keep]))


def proximity_strategy(df, mut_indices, n_exp, samples_per_exp):
    print("mutation indices ", mut_indices)
    df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in mut_indices])
    subset_mut_pos = [desai_mut_pos[i] for i in mut_indices] 
    desai_residues = residues_chain_RBD.iloc[subset_mut_pos]
    desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

    # Compute the R² for the proximity strategy
    ind, dist = proximity_clustering(desai_positions, n_exp, k=samples_per_exp)
    strategy_lists = [[mut_indices[el] for el in ind] for ind in ind.values()]
    # strategy_lists, dist = uneven_proximity_clustering(desai_positions, n_exp, max_exp_size=samples_per_exp)
    
    print("Proximal Strategy gives ", strategy_lists)

    training_set = None
    for indices in strategy_lists:
        training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
    training_set = training_set.drop_duplicates(subset=['onehot'])
    test_set = df
    r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", "linear", show=False, use_onehot=True)
    print("R² for strategy", r2_strat)
    return r2_strat, len(training_set)


# These are the positions of mutations in Wildtype
# desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]


# For ab, need to change the desai_mut_pos to adapt to the pdb file
# desai_mut_pos = [5, 37, 39, 41, 83, 106, 112, 140, 145, 152, 154, 157, 161]
# desai_mut_pos = [5, 37, 39, 41, 83, 106, 112, -1, -1, 140, 145, 152, 154, 157, 161]
desai_mut_pos = [6, 38, 40, 42, 84, 107, 113, 144, 145, 151, 160, 163, 165, 168, 172]

mut_possible = 13
n_exp = 3
samples_per_exp = 5

n_exp_list = [2,3,4,5,6]
samples_per_exp_list = [2,3,4,5,6,7,8]


key_few_mut = [ 5,  6, 12, 13, 11,  1,  2, 14, 10,  0]
# key_few_mut = range(15)
res = []


# run_10choose8_benchmarks(df_sorted, n_loci=8, n_exp=2, samples_per_exp=5)
# for n_exp in n_exp_list:
#     for samples_per_exp in samples_per_exp_list:
#         print("Running proximal strategy for n_exp", n_exp, "samples_per_exp", samples_per_exp)
#         # run_multiple_benchmarks(df_sorted, n_loci=8, n_iter=2, n_exp=n_exp, samples_per_exp=samples_per_exp)
#         r2_strat, num_train = proximity_strategy(df_sorted, key_few_mut, n_exp, samples_per_exp)
#         res.append((n_exp, samples_per_exp, num_train, r2_strat))

# np.save(folder_path+"/res_S309.npy", res)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# df_comb = pd.concat([get_subset_combinatorial(df_sorted,[5, 14, 12, 13, 3, 2, 11, 6]), get_subset_combinatorial(df_sorted, [9, 7, 8, 10, 4, 11, 14, 12])])

# df_comb.to_csv("df_comb.csv", index=False)

df_comb = df_sorted.sample(1000)

from embedding_model import EmbeddingESMWithMLP

device = "cuda"

model = EmbeddingESMWithMLP(device, 1280, 1).to(device)

# Train the model using df_comb["mean_representation"]

# df_comb["mean_representation"] = df_comb["mean_representation"].apply(lambda x: torch.tensor(x).float())
targets = df_comb[antibody]

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("len df comb", len(df_comb))

print("mean rep col", len(df_comb["mean_representation"].values))
# Define the DataLoader
# Convert numpy arrays to tensors
mean_representation_tensors = [torch.tensor(arr) for arr in df_comb["mean_representation"].values]

# Stack the tensors
stacked_mean_representation = torch.stack(mean_representation_tensors)

print("stacked tensors", stacked_mean_representation.shape)

# Define the DataLoader
dataset = TensorDataset(stacked_mean_representation, torch.tensor(targets.values).float())
loader = DataLoader(dataset, batch_size=4)


# Train the model
from hadamard_transform import hadamard_transform

def penalization_term(model, loader, coefficient=0.001):
    # Get the regression predictions of the model for all elements in df
    model.eval()
    all_outputs = []
    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        # Extend the all_outputs list
        all_outputs.append(outputs.detach().cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    print("all outputs", all_outputs.shape)
    # Hadamard transform the predictions vector
    had_coefficients = hadamard_transform(torch.tensor(all_outputs.squeeze(1)))
    # Compute the penalization term based on the sparsity of the hadamard coefficients (L1 norm)
    return coefficient*torch.norm(torch.tensor(had_coefficients), p=1)

model.train()
for epoch in range(10):
    for batch in loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        print("loss without penalization", loss)
        loss += penalization_term(model, loader)
        print("loss with penalization", loss)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")


# Test the model on the whole df_sorted
model.eval()

df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: torch.tensor(x).float())

# Convert numpy arrays to tensors
mean_representation_tensors = [torch.tensor(arr) for arr in df_sorted["mean_representation"].values]

# Stack the tensors
stacked_mean_representation = torch.stack(mean_representation_tensors)

# Define the DataLoader
dataset = TensorDataset(stacked_mean_representation, torch.tensor(targets_sorted).float())
loader = DataLoader(dataset, batch_size=4)


# Compute the predictions
predictions = []
for batch in loader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    predictions.append(outputs.detach().cpu().numpy())

predictions = np.concatenate(predictions)

# Compute the R²
print("targets shape", len(targets_sorted))
print("predictions shape", predictions.shape)
from sklearn.metrics import r2_score
r2 = r2_score(targets_sorted, predictions)
print(f"R² for the model: {r2}")