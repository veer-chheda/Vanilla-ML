from vanillaml.utils import load_data, standardize, normalize, one_hot_encoding, train_test_split, plot_linear_regression
import pandas as pd
import numpy as np

data = load_data('datasets/Boston.csv')

X_df = data.drop('medv', axis=1)
y_df = data['medv']
X = X_df.to_numpy().astype(np.float64)
y = y_df.to_numpy().astype(np.float64)

plot_linear_regression(X, y, dim=2)