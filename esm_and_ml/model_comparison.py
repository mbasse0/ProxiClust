import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import datetime
import os
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# Current date and time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_names = f'{timestamp}_model_comparison'
dataset_name = 'df_desai_old_full'
antibody = "log10Kd_ACE2"

print('antibody')
# antibody = "log10Kd_CoV555"
# antibody = "log10Kd_S309"
# train_sizes = [20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, 5000, 10000]
# Generate train sizes that span in a log scale between 20 and 20000
train_sizes = np.logspace(np.log10(20), np.log10(10000), num=15).astype(int)
# train_sizes = [20, 50, 100, 200, 300, 750, 1000, 2000, 5000, 10000]
print("Logarithmically spaced train sizes:", train_sizes)

num_random = 112  # Number of random tests for averaging
num_process = 112
folder_path = f'./runs/{file_names}_{dataset_name}_{antibody}_{num_random}runs'
print(folder_path)
os.makedirs(folder_path, exist_ok=True)

df_sorted = pd.read_csv(f'./datasets/{dataset_name}.csv')
df_sorted["onehot"] = df_sorted["onehot"].apply(lambda x: [int(i) for i in x.replace('[','').replace(']','').split(', ')])
df_sorted["mean_representation"] = df_sorted["mean_representation"].apply(lambda x: [float(i) for i in x.replace('[','').replace(']','').split(', ')])
targets_sorted = df_sorted[antibody].to_list()

# Prepare data
y = np.array(targets_sorted)
X_mean_representation = np.array(df_sorted["mean_representation"].to_list())
X_onehot = np.array(df_sorted["onehot"].to_list())


print("Computing the PCA at the beginning")
# Compute PCA on the entire X_mean_representation dataset
# pca = PCA(n_components=20)
# X_mean_pca = pca.fit_transform(X_mean_representation)
# print("Done computing the PCA")

# kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))


# Define models
models = {
    "linear_onehot": (LinearRegression(), X_onehot),
    "mlp_mean": (MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42), X_mean_representation),
    # "pca_mlp_mean": (MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42), X_mean_pca),
    # "pca_linear_mean": (LinearRegression(), X_mean_pca),
    "linear_mean": (LinearRegression(), X_mean_representation),
    "mlp_onehot": (MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42), X_onehot),
    "rf_onehot": (RandomForestRegressor(n_estimators=100, random_state=42), X_onehot),
    "rf_mean": (RandomForestRegressor(n_estimators=100, random_state=42), X_mean_representation),
    # "pca_rf_mean": (RandomForestRegressor(n_estimators=100, random_state=42), X_mean_pca)
    # "gp_onehot": (GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42), X_onehot),
    # "gp_pca_mean": (GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42), X_mean_pca)
}

# models = {
#     "linear_onehot": (LinearRegression(), X_onehot),
#     "rf_onehot": (RandomForestRegressor(n_estimators=100, random_state=42), X_onehot),
#     "mlp_onehot_100": (MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42), X_onehot),
#     "mlp_onehot_100_5000iter": (MLPRegressor(hidden_layer_sizes=(100,), max_iter=5000, random_state=42), X_onehot),
#     "mlp_onehot_256_5000iter": (MLPRegressor(hidden_layer_sizes=(256,), max_iter=5000, random_state=42), X_onehot),
#     "mlp_onehot_512_10000iter": (MLPRegressor(hidden_layer_sizes=(512,), max_iter=10000, random_state=42), X_onehot),

# }
# Add second order polynomial features for linear regression models
def add_polynomial_features(models, X_onehot, X_mean_representation, X_mean_pca):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    models["second_order_linear_onehot"] = (LinearRegression(), poly.fit_transform(X_onehot))
    # models["second_order_linear_mean"] = (LinearRegression(), poly.fit_transform(X_mean_representation))
    models["second_order_pca_linear_mean"] = (LinearRegression(), poly.fit_transform(X_mean_pca))

# add_polynomial_features(models, X_onehot, X_mean_representation, X_mean_pca)

results = {model_name: {"scores": [], "stds": [], "medians": []} for model_name in models}

def train_and_evaluate(args):
    model, X, y, train_size, seed = args
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=len(y)-train_size, random_state=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

for train_size in train_sizes:
    print("Processing for train size of ", train_size)
    for model_name, (model, X) in models.items():
        print(model_name)
        seeds = np.random.randint(0, 2**32 - 1, size=num_random)  # Generate unique seeds for each run
        args = [(model, X, y, train_size, seeds[i]) for i in range(num_random)]
        with Pool(processes=num_process) as pool:
            r2_list = pool.map(train_and_evaluate, args)
        results[model_name]["scores"].append(np.mean(r2_list))
        results[model_name]["stds"].append(np.std(r2_list))
        results[model_name]["medians"].append(np.median(r2_list))

# Save the individual results
# for model_name in models:
#     np.save(f'{folder_path}/{model_name}_r2_scores.npy', results[model_name]["scores"])
#     np.save(f'{folder_path}/{model_name}_r2_stds.npy', results[model_name]["stds"])
#     np.save(f'{folder_path}/{model_name}_r2_medians.npy', results[model_name]["medians"])

# Save the entire results dictionary
np.save(f'{folder_path}/results_{file_names}_{dataset_name}_{antibody}_{num_random}runs.npy', results, allow_pickle=True)
