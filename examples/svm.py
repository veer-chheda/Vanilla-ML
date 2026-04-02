import pandas as pd
import numpy as np
from vanillaml.algorithms import SVM
from vanillaml.math import accuracy
from vanillaml.utils import train_test_split, standardize

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(url, names=columns)

X = df.drop('class', axis=1).to_numpy()
y = df['class'].to_numpy()

X = standardize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm = SVM(lambda_param=0.01, learning_rate=0.001, n_iterations=1000)
svm.fit(X_train, y_train)
preds = svm.predict(X_test)
y_test_converted = np.where(y_test <= 0, -1, 1)

final_accuracy = accuracy(y_test_converted, preds)
print(final_accuracy)