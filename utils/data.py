import numpy as np
import pandas as pd

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xml'):
        return pd.read_xml(file_path)
    elif file_path.endswith('.html'):
        return pd.read_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def shuffle_data(X, y, seed = 42):
    # No need to set the seed, function uses random seed if not specified.
    np.random.seed()
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def train_test_split(X, y, test_size = 0.2, shuffle = True):
    if shuffle:
        X, y = shuffle_data(X, y)
    if test_size >= 1:
        raise ValueError("test_size must be less than 1")
    split_index = len(y) - int((test_size) * (len(y)))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

def standardize(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis = 0)
    X_std = (X - mean)/std_dev
    return X_std

def normalize(X):
    x_max, x_min = X.max(axis=0), X.min(axis=0)
    X_normalized = (X - x_min)/(x_max - x_min + 1e-8)
    return X_normalized

def one_hot_encoding(X):
    unique_labels = len(np.unique(X))
    one_hot_encoded = np.zeros(X.size, unique_labels)
    one_hot_encoded[np.arange(X.size), X] = 1

