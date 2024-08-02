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


# This script intends to compare the needed number of train samples for different sizes of combinatorial datasets (synthetically created from extractions from desai old dataset)


now = datetime.datetime.now()

# Format datetime in YYYYMMDD_HHMMSS format
timestamp = now.strftime("%Y%m%d_%H%M%S")

print("Timestamp", timestamp)
file_names = f'{timestamp}'

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
    df_sorted, subset, size, antibody= args
     # Get the subset of the dataframe
    df_subset = get_subset_combinatorial(df_sorted, subset, from_variant='wt')
    # Train a linear model on this subset, and predict the subset
    X = np.array(df_subset["onehot"].tolist())
    y = df_subset[antibody].values
    
    # Ensure we can split the data and handle the case where the subset size is larger than available data
    if len(X) < size:
        print(f"Subset size {size} is larger than the available data. Skipping.")
    
    r2_splits = []
    for s in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train.tolist(), y_train)
        y_pred = model.predict(X_test.tolist())
        r2 = r2_score(y_test, y_pred)
        r2_splits.append(r2)
    return np.mean(r2_splits)

num_splits = 1000
print("Num splits,", num_splits)

num_comb = 1000
print("Num combinations,", num_comb)
antibody = "log10Kd_ACE2"
# antibody = "log10Kd_REGN10987"
# antibody = 'log10Kd_S309'



subset_sizes = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500 , 750, 1000]
print("subset sizes", subset_sizes)
list_j = range(7,16)
list_j = [7,14]
for j in list_j:
    mean_r2_list = []
    for size in subset_sizes:
        if size < 2**j:
            # Get all the subsets of length j
            r2_list = []
            all_combinations = list(combinations(range(15), j))
            chosen_combinations_ind = np.random.choice(len(all_combinations), num_comb)
            chosen_combinations = [list(all_combinations[ind]) for ind in chosen_combinations_ind]

            tasks = [(df_sorted, subset, size, antibody) for subset in chosen_combinations]
            
            with Pool(processes=112) as pool:  # Match the number of processes to the number of allocated CPU cores
                results = pool.map(sub_r2_compute, tasks)
            for r2 in results:
                r2_list.append(r2)

            mean_r2_list.append(np.mean(r2_list))
            print("R² score for ", j, "loci combinatorial dataset with ", size, " training samples", np.mean(r2_list))
        else:
            mean_r2_list.append(np.nan)
    np.save(folder_path + f"/r2_scores_{j}_loci_{num_comb}comb_{num_splits}splits{antibody}.npy", mean_r2_list)
