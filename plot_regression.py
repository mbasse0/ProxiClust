
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge



def plot_regression(training_set, test_set, mut_name, antibody, train_column="onehot",model_type='linear', show=True, use_onehot=True):
    # Extract data
    y_train=training_set[antibody].tolist()
    y_test=test_set[antibody].tolist()

    if use_onehot:
        X_train = training_set[train_column].tolist()
        X_test = test_set[train_column].tolist()
    else:
        X_train=training_set['mean_representation'].tolist()
        X_test=test_set['mean_representation'].tolist()
    # Scale data for models like SVR that are sensitive to the scale of input data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Select and fit the model
    if model_type == 'tree':
        model = DecisionTreeRegressor(max_depth=5)
        X_train_used = X_train  # No need to scale data for tree-based models
        X_test_used = X_test
    elif model_type == 'svr':
        model = SVR(kernel='rbf')
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train_used = X_train
        X_test_used = X_test
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 100)
        X_train_used = X_train
        X_test_used = X_test
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    else:  # Default to linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X_train_used = X_train
        X_test_used = X_test

    model.fit(X_train_used, y_train)
    y_pred = model.predict(X_test_used)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)
    # print(f"R2 score for {model_type}: {r2}")

    # Plot the results
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.xlabel(f'True Values {antibody}')
    plt.ylabel(f'Predictions {antibody}')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')  # y=x line
    plt.title(f'{model_type.title()} Regression, with {mut_name}\nR² = {r2:.2f}')
    if show:
        plt.show()

    return r2