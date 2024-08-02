import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import multiprocessing
import os
import datetime
from sklearn.ensemble import RandomForestRegressor



# This script intends to compare the needed number of train samples for different sizes of combinatorial datasets (synthetically created from extractions from desai old dataset)


now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")

print("Timestamp", timestamp)
file_names = f'{timestamp}_perf_by_size'

folder_path = f'runs/{file_names}'
os.makedirs(folder_path, exist_ok=True)




df_sorted = pd.read_csv('datasets/df_desai_old_full.csv')
# df_sorted = pd.read_csv('df_desai_new_full.csv')


df_sorted["onehot"] = df_sorted["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])


def get_subset_combinatorial(df, indices, from_variant='wt'):
    num_loci=15
    if from_variant=='wt':
        return df[df["onehot"].apply(lambda x : not any(x[ind] for ind in range(num_loci) if ind not in indices))]
    elif from_variant=='omicron':
       return df[df["onehot"].apply(lambda x : not any( x[ind] for ind in range(num_loci) if ind not in indices))] 
    else:
        print("Wrong origin variant")



def sub_r2_compute(args):
    X, y, size, antibody, seed = args

    #Set random seed
    np.random.seed(seed)

     
    
    # Ensure we can split the data and handle the case where the subset size is larger than available data
    if len(X) < size:
        print(f"Subset size {size} is larger than the available data. Skipping.")
    
    r2_splits = []
    for s in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=seed)
        # model = LinearRegression()
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(X_train.tolist(), y_train)
        y_pred = model.predict(X_test.tolist())
        r2 = r2_score(y_test, y_pred)
        r2_splits.append(r2)

    # print("std after splits", np.std(r2_splits))
    # return np.mean(r2_splits)
    # return np.median(r2_splits)
    return r2_splits

# num_splits = 50
num_splits = 1
print("Num splits,", num_splits)


num_comb = 5000
# num_comb = 50
print("Num combinations,", num_comb)
antibody = "log10Kd_ACE2"
# antibody = "log10Kd_REGN10987"
# antibody = 'log10Kd_S309'
print(antibody)


# Define the range and number of samples
start = np.log10(500)   # Start at 10 (log10 of 10 is 1)
stop = np.log10(5000)  # Stop at 5000 (log10 of 5000 is approximately 3.7)
num_samples = 10       # Number of samples

# Generate logarithmically spaced values
subset_sizes = np.logspace(start, stop, num=num_samples, dtype=int)

print("subset sizes", subset_sizes)


results_dict = {}


list_j = range(10,15)

print("List j is ", list_j)
for j in list_j:
    print("Tests for comb size", j)
    median_r2_list = []
    for size in subset_sizes:
        print("Train set size", size)
        if size < 2**j:
            # Get all the subsets of length j
            r2_list = []
            all_combinations = list(combinations(range(15), j))
            if num_comb < len(all_combinations):
                chosen_combinations_ind = np.random.choice(len(all_combinations), num_comb)
                chosen_combinations = [list(all_combinations[ind]) for ind in chosen_combinations_ind]
            else:
                chosen_combinations = all_combinations
            
            X_list = []
            y_list = []
            for subset in chosen_combinations:
                df_subset = get_subset_combinatorial(df_sorted, subset, from_variant='wt')

                # Get the subset of the dataframe
                # Train a linear model on this subset, and predict the subset
                X_list.append(np.array(df_subset["onehot"].tolist()))
                y_list.append(df_subset[antibody].values)
            tasks = [(X_list[k], y_list[k], size, antibody, j*size + j) for k in range(len(chosen_combinations))]
            
            with Pool(processes=112) as pool:  # Match the number of processes to the number of allocated CPU cores
                results = pool.map(sub_r2_compute, tasks)

            flat_results = [item for sublist in results for item in sublist]
            # print("Flat results", len(flat_results), flat_results)

            median_res = np.median(flat_results)
            median_r2_list.append(median_res)
            print("R² score for ", j, "loci combinatorial dataset with ", size, " training samples", median_res)
        else:
            median_r2_list.append(np.nan)

    results_dict[j] = median_r2_list
    # np.save(folder_path + f"/r2_scores_{j}_loci_{num_comb}comb_{num_splits}splits{antibody}.npy", mean_r2_list)

np.save(folder_path + f"/all_r2_scores_loci_{num_comb}comb_{num_splits}splits{antibody}.npy", results_dict)  # Saving as a NumPy binary file
