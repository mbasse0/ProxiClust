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
from multiprocessing import Pool
import multiprocessing


# # For S309


# atom_df = ppdb.df["ATOM"]
# CA_df = atom_df[atom_df["atom_name"] == "CA"]

# residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
# residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
# residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])

# residues_chain_RBD = CA_df[CA_df["chain_id"]=='C'].sort_values("residue_number")
# # residues_chainE = residues_chainE[residues_chainE["alt_loc"] != 'A']

# aa_dict = {
#     'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
#     'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
#     'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
#     'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
#     'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
# }


# # S309 : H,A // L,B
# residues_chain_ab["single_letter_residue"] = residues_chain_ab["residue_name"].map(aa_dict)
# residues_chain_RBD["single_letter_residue"] = residues_chain_RBD["residue_name"].map(aa_dict)



# amino_acid_sequence_ab = ''.join(residues_chain_ab["single_letter_residue"].values)
# amino_acid_sequence_RBD = ''.join(residues_chain_RBD["single_letter_residue"].values)



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


def random_benchmark(df, antibody, test_set, n_exp, samples_per_exp, mut_pos_desai, num_random, model_type='linear', use_onehot=True):
    r2s = []
    all_combinations = list(combinations(mut_pos_desai, samples_per_exp))
    num_comb = len(all_combinations)
    all_choices = list(combinations(range(num_comb), n_exp))
    random_ind = np.random.choice(len(all_choices), num_random, replace=False)
    random_choices = [all_choices[i] for i in random_ind]
    indices_lists = [ [all_combinations[el] for el in choice ]for choice in random_choices]
    
    

    for i, indices_list in enumerate(indices_lists):
        training_set = None
        for indices in indices_list:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        r2 = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", model_type, show=False, use_onehot=use_onehot)
        r2s.append(r2)
    return r2s



def exhaustive_benchmark(df, antibody, test_ind, n_exp, samples_per_exp, mut_pos_desai, model_type='linear', use_onehot=True):
    r2s = []
    all_combinations = list(combinations(mut_pos_desai, samples_per_exp))
    num_comb = len(all_combinations)
    all_choices = list(combinations(range(num_comb), n_exp))
    indices_lists = [ [all_combinations[el] for el in choice ]for choice in all_choices]
    test_set = df.iloc[test_ind]
    

    print(f"Number of indices list to test: {len(indices_lists)}")
    for i, indices_list in enumerate(indices_lists):
        training_set = None
        for indices in indices_list:
            training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
        training_set = training_set.drop_duplicates(subset=['onehot'])
        r2 = plot_regression(training_set, test_set, "custom split", antibody, "onehot", model_type, show=False, use_onehot=use_onehot)
        r2s.append(r2)
    return r2s

def compute_quantile(val, all_values):
    return sum(val > all_values)/len(all_values)

def sub_choose_exp(args):
    k, desai_mut_pos, residues_chain_RBD, antibody, reduced_few_mut_list, df, test_ind, n_exp, samples_per_exp = args
    print("Iteration", k, " out of ", len(reduced_few_mut_list))
    key_few_mut = reduced_few_mut_list[k]
    chosen_ind = key_few_mut
    print("Chosen indices", key_few_mut)
    df["custom_onehot"] = df["onehot"].apply(lambda x: [x[i] for i in key_few_mut])
    subset_mut_pos = [desai_mut_pos[i] for i in key_few_mut] 
    desai_residues = residues_chain_RBD.iloc[subset_mut_pos]
    desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

    # Compute the R² for the proximity strategy
    ind, dist = proximity_clustering(desai_positions, n_exp, k=samples_per_exp)
    strategy_lists = [[key_few_mut[el] for el in ind] for ind in ind.values()]

    # strategy_lists, _ = even_kmeans_clustering(desai_positions, n_exp, samples_per_exp)
    # strategy_lists, _ = kmeans_clustering(desai_positions, n_exp)
    print("even kmeans Strategy gives ", strategy_lists)
    training_set = None
    for indices in strategy_lists:
        training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
    training_set = training_set.drop_duplicates(subset=['onehot'])
    test_set = df.iloc[test_ind]
    r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "custom_onehot", "linear", show=False, use_onehot=True)
    print("R² for strategy", r2_strat)

    # r2_benchmark = exhaustive_benchmark(df, test_set, n_exp, samples_per_exp, key_few_mut, model_type='linear', use_onehot=True)
    num_random_iter = 50
    print("Number of random iterations", num_random_iter)
    r2_benchmark = random_benchmark(df, antibody, test_set, n_exp, samples_per_exp, key_few_mut, num_random_iter, model_type='linear', use_onehot=True)

    r2_benchmark
    quantile = compute_quantile(r2_strat, r2_benchmark)

    return r2_strat, r2_benchmark, chosen_ind, quantile


def proximity_strategy(df, test_ind, antibody, desai_mut_pos, residues_chain_RBD, mut_indices, n_exp, samples_per_exp):
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
    test_set = df.iloc[test_ind]
    r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "onehot", "linear", show=False, use_onehot=True)
    print("R² for strategy", r2_strat)
    return r2_strat


def get_r2_from_strategy(args):
    df, antibody, residue_lists, test_ind = args
    training_set = None
    for indices in residue_lists:
        training_set = pd.concat([training_set, get_subset_combinatorial(df, indices)])
    training_set = training_set.drop_duplicates(subset=['onehot'])
    test_set = df.iloc[test_ind]
    r2_strat = plot_regression(training_set, test_set, "custom split", antibody, "onehot", "linear", show=False, use_onehot=True)
    return r2_strat

    

def get_interface(positions, residues_chain_ab, n_keep=10):
    positions_ab = residues_chain_ab[["x_coord", "y_coord", "z_coord"]].values

    # Find residues in E that are closest to the interface
    return np.argsort(np.min(cdist(positions, positions_ab), axis=1))[:n_keep]

def main():
    if __name__ == '__main__':
        print("Number of cpus", multiprocessing.cpu_count())
        # Current date and time
        now = datetime.datetime.now()

        # Format datetime in YYYYMMDD_HHMMSS format
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        print("Timestamp", timestamp)
        file_names = f'{timestamp}'

        folder_path = f'../runs/{file_names}'
        os.makedirs(folder_path, exist_ok=True)

        # Load the dataframe

        dataset_path = 'df_desai_old_full.csv'
        print(dataset_path)
        df_sorted = pd.read_csv(dataset_path)

        df_sorted["onehot"] = df_sorted["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
        df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


        # antibody = "log10Kd_REGN10987"
        antibody = 'log10Kd_S309'
        # antibody = "log10Kd_ACE2"
        # antibody = "log10Kd_CoV555"
        # antibody = "log10Kd_CB6"

        print(antibody)
        targets_sorted = df_sorted[antibody].to_list()
        ## Load the pdb file
        
        if antibody == "log10Kd_ACE2":
            file_path = "../PDB_files/6m0j.pdb"
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            num_residues = 15
        elif antibody=="log10Kd_REGN10987":
            file_path = "../PDB_files/8j26.pdb"
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            num_residues = 15
        elif antibody=="log10Kd_S309":
            file_path = "../PDB_files/7xck.pdb"
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            desai_mut_pos = [el+9 for el in desai_mut_pos]
            num_residues = 15
        elif antibody == "log10Kd_CoV555":
            file_path = "../PDB_files/7kmg.pdb"
            num_residues = 15
        elif antibody == "log10Kd_CB6":
            file_path = "../PDB_files/7c01.pdb"
            num_residues = 15

        


        ppdb = PandasPdb().read_pdb(file_path)


        atom_df = ppdb.df["ATOM"]
        CA_df = atom_df[atom_df["atom_name"] == "CA"]


        if antibody == "log10Kd_ACE2":
            residues_chain_ab = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
            residues_chain_RBD = CA_df[CA_df["chain_id"]=='E'].sort_values("residue_number")
            residues_chain_RBD = residues_chain_RBD[residues_chain_RBD["alt_loc"] != 'A']
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            desai_mut_pos = [pos-2 for pos in desai_mut_pos]
        elif antibody=="log10Kd_REGN10987":
            residues_chain_RBD = CA_df[CA_df["chain_id"]=='C'].sort_values("residue_number")
            residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
            residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
            residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            desai_mut_pos = [pos-2 for pos in desai_mut_pos]
        elif antibody=="log10Kd_S309":
            residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
            residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
            residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])
            residues_chain_RBD = CA_df[CA_df["chain_id"]=='M'].sort_values("residue_number")
        elif antibody == "log10Kd_CoV555":
            residues_chain_RBD = CA_df[CA_df["chain_id"]=='C'].sort_values("residue_number")
            residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
            residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='B'].sort_values("residue_number")
            residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            desai_mut_pos = [pos-3 for pos in desai_mut_pos]
        elif antibody == "log10Kd_CB6":
            residues_chain_RBD = CA_df[CA_df["chain_id"]=='A'].sort_values("residue_number")
            residues_chain_ab_part1 = CA_df[CA_df["chain_id"]=='C'].sort_values("residue_number")
            residues_chain_ab_part2 = CA_df[CA_df["chain_id"]=='D'].sort_values("residue_number")
            residues_chain_ab = pd.concat([residues_chain_ab_part1, residues_chain_ab_part2])
            desai_mut_pos = [ 8, 40, 42, 44, 86,109, 115, 146, 147, 153, 162, 165, 167, 170, 174]
            desai_mut_pos = [pos-2 for pos in desai_mut_pos]


        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        residues_chain_ab["single_letter_residue"] = residues_chain_ab["residue_name"].map(aa_dict)
        residues_chain_RBD["single_letter_residue"] = residues_chain_RBD["residue_name"].map(aa_dict)

        amino_acid_sequence_ab = ''.join(residues_chain_ab["single_letter_residue"].values)
        amino_acid_sequence_RBD = ''.join(residues_chain_RBD["single_letter_residue"].values)

        # These are the positions of mutations in Wildtype
        

        
        desai_residues = residues_chain_RBD.iloc[desai_mut_pos]
        desai_positions = desai_residues[["x_coord", "y_coord", "z_coord"]].values

        interface_size = 10
        interface_residues = get_interface(desai_positions, residues_chain_ab, interface_size)
        print(f"Interface of size {interface_size} for ", antibody, "is ", interface_residues)


        num_processes = 48
        # EXP 1 -------------------------------------------------------------------------------------------------------------
        
        n_exp = 2
        samples_per_exp = 6
        print("Samples per exp", samples_per_exp)
        print("N exp", n_exp)
        test_ind = random.sample(range(len(df_sorted)), 15000)

        r2_opti = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, interface_residues, n_exp, samples_per_exp)
        print("R2 opti", r2_opti)

        r2_all = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, list(range(num_residues)), n_exp, samples_per_exp)
        print("R2 all", r2_all)

        # Parallelize the exhaustive scan on interface

        all_combinations = list(combinations(interface_residues, samples_per_exp))
        num_comb = len(all_combinations)
        all_choices = list(combinations(range(num_comb), n_exp))
        indices_lists = [ [all_combinations[el] for el in choice ]for choice in all_choices]

        tasks = [(df_sorted, antibody, strat, test_ind) for strat in indices_lists]
        r2_list_interface = []
        print("Starting processes for interface")
        # Use multiprocessing to run the loop in parallel
        with Pool(processes=num_processes) as pool:  # Match the number of processes to the number of allocated CPU cores
            results = pool.map(get_r2_from_strategy, tasks)
        for r2 in results:
            r2_list_interface.append(r2)


        print("Done exhaustive scan on interface")

        num_random_iter = 100000
        print("Number of random samples:", num_random_iter)
        # Sample from every possible strategy
        all_combinations = list(combinations(list(range(num_residues)), samples_per_exp))
        pairs = list(combinations(range(len(all_combinations)), 2))

        chosen_ind = np.random.choice(len(pairs), num_random_iter, replace=False)
        chosen_pairs = [pairs[i] for i in chosen_ind]
        chosen_strategies = [[list(all_combinations[i]), list(all_combinations[j])] for i,j in chosen_pairs]

        tasks = [(df_sorted, antibody, strat, test_ind) for strat in chosen_strategies]
        r2_lists = []
        print("Starting processes")
        # Use multiprocessing to run the loop in parallel
        with Pool(processes=num_processes) as pool:  # Match the number of processes to the number of allocated CPU cores
            results = pool.map(get_r2_from_strategy, tasks)
        for r2_list in results:
            r2_lists.append(r2_list)

        np.save(folder_path+f"/{antibody}r2_list_interface_exp1_{samples_per_exp}per_exp.npy", r2_list_interface)
        np.save(folder_path+f"/{antibody}r2_lists_exp1_{samples_per_exp}per_exp.npy", r2_lists)
        print("DONE EXP1")
        

        
        # EXP 5 -------------------------------------------------------------------------------------------------------------
        """
        num_random_iter = 10
        test_ind = random.sample(range(len(df_sorted)), 5000)
        n_exp = 2
        samples_per_exp = 7

        r2_list_interface = exhaustive_benchmark(df_sorted, antibody, test_ind, n_exp, samples_per_exp, interface_residues, model_type='linear', use_onehot=True)

        r2_opti = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, interface_residues, n_exp, samples_per_exp)
        print("R2 opti", r2_opti)

        r2_all = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, list(range(num_residues)), n_exp, samples_per_exp)
        print("R2 all", r2_all)

        all_subsets = list(combinations(range(15), 7))

        # Function to find pairs with no overlap
        def find_non_overlapping_pairs(subsets):
            non_overlapping_pairs = []
            n = len(subsets)
            for i in range(n):
                for j in range(i + 1, n):
                    if set(subsets[i]).isdisjoint(subsets[j]):
                        non_overlapping_pairs.append((subsets[i], subsets[j]))
            return non_overlapping_pairs

        # Find and print non-overlapping pairs
        non_overlapping_pairs = find_non_overlapping_pairs(all_subsets)
        test_set = df_sorted.iloc[test_ind]
        
        chosen_strategies = [list(el) for el in non_overlapping_pairs]
        tasks = [(df_sorted, antibody, strat, test_ind) for strat in chosen_strategies]
        r2_lists = []
        print("Starting processes")
        # Use multiprocessing to run the loop in parallel
        with Pool(processes=num_processes) as pool:  # Match the number of processes to the number of allocated CPU cores
            results = pool.map(get_r2_from_strategy, tasks)
        for r2_list in results:
            r2_lists.append(r2_list)

        np.save(folder_path+"/r2_lists.npy", r2_lists)
        np.save(folder_path+"/r2_list_interface.npy", r2_list_interface)
        print("DONE EXP5")
        """
        
        # EXP2 -------------------------------------------------------------------------------------------------------------
        
        
        # test_ind = random.sample(range(len(df_sorted)), 25000)
        # n_exp = 2
        # samples_per_exp = 8
        # print("N exp", n_exp)
        # print("samples per exp", samples_per_exp)
        # r2_list_interface = exhaustive_benchmark(df_sorted, antibody, test_ind, n_exp, samples_per_exp, interface_residues, model_type='linear', use_onehot=True)
        
        # r2_opti = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, interface_residues, n_exp, samples_per_exp)

        # print("R2 opti", r2_opti)
        # np.save(folder_path+f"/{antibody}_r2_list_interface_exp2_{samples_per_exp}per_exp.npy", r2_list_interface)
        # print("Done EXP2")

        
        # EXP3 -------------------------------------------------------------------------------------------------------------
        '''
        test_ind = random.sample(range(len(df_sorted)), 10000)
        n_exp = 2
        samples_per_exp = 8
        

        r2_opti = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, interface_residues, n_exp, samples_per_exp)

        print("R2 opti", r2_opti)

        num_random_iter = 1000000
        # Sample from every possible strategy
        all_combinations = list(combinations(list(range(num_residues)), samples_per_exp))
        pairs = list(combinations(range(len(all_combinations)), 2))

        chosen_ind = np.random.choice(len(pairs), num_random_iter, replace=False)
        chosen_pairs = [pairs[i] for i in chosen_ind]
        chosen_strategies = [[list(all_combinations[i]), list(all_combinations[j])] for i,j in chosen_pairs]

        tasks = [(df_sorted, antibody, strat, test_ind) for strat in chosen_strategies]
        r2_lists = []
        print("Starting processes")
        # Use multiprocessing to run the loop in parallel
        with Pool(processes=num_processes) as pool:  # Match the number of processes to the number of allocated CPU cores
            results = pool.map(get_r2_from_strategy, tasks)
        for r2_list in results:
            r2_lists.append(r2_list)

        np.save(folder_path+"/chosen_strategies_exp3.npy", chosen_strategies)
        np.save(folder_path+"/r2_lists_exp3.npy", r2_lists)

        print("Done EXP3")
        '''

        # EXP6 ----------------------------------------------------------------------------------------------------------
        
        # test_ind = random.sample(range(len(df_sorted)), 10000)
        # n_exp = 2
        # samples_per_exp = 8

        # sorted_residues = [13, 14, 11,  7,  6, 10, 12,  8,  4,  9,  5,  3,  2,  1,  0]
        # r2_opti_list = []

        # for j in range(8,16):
        #     interface_residues = sorted_residues[:j]
        #     r2_opti = proximity_strategy(df_sorted, test_ind, antibody, desai_mut_pos, residues_chain_RBD, interface_residues, n_exp, samples_per_exp)
        #     r2_opti_list.append(r2_opti)

        # np.save(folder_path+"/r2_opti_list_exp6.npy", r2_opti_list)
        # print("DONE EXP6")

        
main()

