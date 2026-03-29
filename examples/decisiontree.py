from vanillaml.algorithms import DecisionTree
import pandas as pd
import numpy as np
from vanillaml.math import accuracy
from vanillaml.utils import load_data, standardize, normalize, one_hot_encoding, train_test_split, convert_categorical

data = load_data('datasets/iris.csv')
X_df = data.drop('virginica', axis=1)
y_df = data['virginica']
X = X_df.to_numpy()
y = y_df.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

decisiontree = DecisionTree(max_depth=2, min_samples_split=5)
decisiontree.fit(X_train, y_train)
preds = decisiontree.predict(X_test)
pred_accuracy = accuracy(y_test, preds)
print(pred_accuracy)