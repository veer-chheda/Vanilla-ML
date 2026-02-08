from vanillaml.utils import load_data, standardize, normalize, one_hot_encoding, train_test_split
import pandas as pd
from vanillaml.algorithms import LinearRegression
from vanillaml.utils import MSE, MAE, RMSE
import numpy as np

data = load_data('datasets/Boston.csv')

X_df = data.drop('medv', axis=1)
y_df = data['medv']

# print(X_df.head())
# print(y_df.head())

X = X_df.to_numpy().astype(np.float64)
# print(X)
y = y_df.to_numpy().astype(np.float64)
# print(y)
# print(X.shape, y.shape)
X = normalize(X)
y = normalize(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
regression = LinearRegression(lr=0.001, n_iterations=1000)
regression.fit(X_train, y_train)
y_predict = regression.predict(X_test)

mse = MSE(y_test, y_predict)
mae = MAE(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

print(mse, mae, rmse)


