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
dataset_path = "CH65_processed_new.csv"
# dataset_path = "CH65_SI06_processed.csv"
print("Dataset:", dataset_path)
df_sorted =  pd.read_csv(dataset_path)
# df_sorted = pd.read_csv("CH65_SI06_processed.csv")

df_sorted["onehot"] = df_sorted["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
# df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


antibody = "log10Kd_pinned"

print(antibody)
targets_sorted = np.array(df_sorted[antibody].to_list())
## Load the pdb file

file_path = "5ugy.pdb"


ppdb = PandasPdb().read_pdb(file_path)
ppdb

list_mut_heavy = [31, 33, 34, 35, 52, 57, 83, 84, 85, 87]
adjusted_mut_heavy = [el - 1 for el in list_mut_heavy]


list_mut_light = [26, 29, 35, 48, 49, 98]
adjusted_mut_light = [el - 2 for el in list_mut_light]

adjusted_list_mut = adjusted_mut_light + adjusted_mut_heavy


atom_df = ppdb.df["ATOM"]
CA_df = atom_df[atom_df["atom_name"] == "CA"]

residues_RBS_light = CA_df[CA_df["chain_id"]=='L'].sort_values("residue_number")
residues_RBS_heavy = CA_df[CA_df["chain_id"]=='H'].sort_values("residue_number")
residues_RBS = pd.concat([residues_RBS_light, residues_RBS_heavy])


desai_residues_light = residues_RBS_light.iloc[adjusted_mut_light]
desai_residues_heavy = residues_RBS_heavy.iloc[adjusted_mut_heavy]


desai_residues = pd.concat([desai_residues_light, desai_residues_heavy])
desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

desai_mut_pos = adjusted_mut_light + [len(residues_RBS_light) + adjusted_mut_heavy[i] for i in range(len(adjusted_mut_heavy))] 


residues_antigen_1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
residues_antigen_2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
residues_antigen = pd.concat([residues_antigen_1, residues_antigen_2])


aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


residues_RBS_light["single_letter_residue"] = residues_RBS_light["residue_name"].map(aa_dict)
residues_RBS_heavy["single_letter_residue"] = residues_RBS_heavy["residue_name"].map(aa_dict)



amino_acid_sequence_RBS_light = ''.join(residues_RBS_light["single_letter_residue"].values)
amino_acid_sequence_RBS_heavy = ''.join(residues_RBS_heavy["single_letter_residue"].values)


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

    indices_pool = [ 6, 10,  7, 11,  1,  0,  8,  9,  5,  4]

    for k in range(n_iter):
        key_few_mut = random.sample(indices_pool, n_loci)
        chosen_ind.append(key_few_mut)
        print("Chosen indices", key_few_mut)
        df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in key_few_mut])

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



def run_choose_benchmarks(df, n_loci=8, n_exp=2, samples_per_exp=5):
    chosen_ind = []
    r2_lists = []
    r2_strat_list = []
    quantile_list = []
    test_ind = random.sample(range(len(df)), 1000)

    # indices_pool = [13, 14, 11,  7,  6, 10, 12,  8,  4,  9]
    # indices_pool = [ 0,  5,  6,  2, 12,  1, 13,  11,  3, 14]
    # indices_pool = [ 6, 10,  7, 11,  1,  0,  8,  9,  5,  4]
    indices_pool = list(range(16))
    few_mut_list = list(combinations(indices_pool, n_loci))
    for k in range(len(few_mut_list)):
        print("Iteration", k, " out of ", len(few_mut_list))
        key_few_mut = few_mut_list[k]
        chosen_ind.append(key_few_mut)
        print("Chosen indices", key_few_mut)
        df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in key_few_mut])
        subset_mut_pos = [desai_mut_pos[i] for i in key_few_mut] 
        desai_residues = residues_RBS.iloc[subset_mut_pos]
        desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

        # Compute the R² for the proximity strategy
        ind, dist = proximity_clustering(desai_positions, n_exp, k=samples_per_exp)
        strategy_lists = [[key_few_mut[el] for el in ind] for ind in ind.values()]

        # strategy_lists, _ = even_kmeans_clustering(desai_positions, n_exp, samples_per_exp)
        # strategy_lists, _ = kmeans_clustering(desai_positions, n_exp)
        # print("even kmeans Strategy gives ", strategy_lists)
        training_set = None
        for indices in strategy_lists:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        test_set = df.iloc[test_ind]
        r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", "linear", show=False, use_onehot=True)
        print("R² for strategy", r2_strat)
        r2_strat_list.append(r2_strat)

        # r2_exhaustive = exhaustive_benchmark(df, test_set, n_exp, samples_per_exp, key_few_mut, model_type='linear', use_onehot=True)
        # r2_lists.append(r2_exhaustive)
        # median_exhaustive = np.median(r2_exhaustive)
        # print("Median of exhaustive", median_exhaustive)
        # quantile_list.append(compute_quantile(r2_strat, r2_exhaustive))

        # plt.clf()
        # plt.boxplot(r2_exhaustive)
        # plt.ylim(median_exhaustive - 1 , 1)
        # plt.axhline(y=r2_strat, color='r', linestyle='--')
        # plt.title(f"Benchmark for indices: {key_few_mut}")
        # plt.savefig(folder_path+f"/benchmark_{key_few_mut}.png")

    np.save(folder_path+"/r2_lists.npy", r2_lists)
    np.save(folder_path+"/chosen_indices.npy", chosen_ind)
    np.save(folder_path+"/r2_strat_list.npy", r2_strat_list)
    print("Average increase in R² comapared to median value", np.mean([r2_strat_list[i] - np.median(r2_lists[i]) for i in range(len(r2_strat_list))]))
    print("Average quantile", np.mean(quantile_list))
    print("All quantiles", quantile_list)


def proximity_strategy(df, mut_indices, n_exp, samples_per_exp):
    print("mutation indices ", mut_indices)
    df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in mut_indices])

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



mut_possible = 16
n_exp = 3
samples_per_exp = 5

n_exp_list = [2,3,4,5,6]
samples_per_exp_list = [2,3,4,5,6,7,8]


# key_few_mut = [ 6, 10,  7, 11,  1,  0,  8,  9,  5,  4]
key_few_mut = range(16)
print("RUnning for 90, wrong interface")
desai_residues_light = residues_RBS_light.iloc[[el for el in key_few_mut if el < 6]]
desai_residues_heavy = residues_RBS_heavy.iloc[[el for el in key_few_mut if el >= 6]]
desai_residues = pd.concat([desai_residues_light, desai_residues_heavy])
desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

res = []

# run_choose_benchmarks(df_sorted, n_loci=10, n_exp=2, samples_per_exp=8)
r2_all = proximity_strategy(df_sorted, list(range(16)), 2, 8)
print("R2 all", r2_all)
# for n_exp in n_exp_list:
#     for samples_per_exp in samples_per_exp_list:
#         print("Running proximal strategy for n_exp", n_exp, "samples_per_exp", samples_per_exp)
#         # run_multiple_benchmarks(df_sorted, n_loci=8, n_iter=2, n_exp=n_exp, samples_per_exp=samples_per_exp)
#         r2_strat, num_train = proximity_strategy(df_sorted, key_few_mut, n_exp, samples_per_exp)
#         res.append((n_exp, samples_per_exp, num_train, r2_strat))

# np.save(folder_path+"/res_MA90.npy", res)
