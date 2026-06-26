import numpy as np
import math

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def softmax(X):
    e_x = np.exp(X - np.max(X, axis = -1, keepdims=True))
    return e_x / np.sum(e_x, axis = -1, keepdims=True)

def tanh(X):
    return 2 / (1 + np.exp(-2 * X)) - 1

def ReLU(X):
    return np.where(X >= 0, X, 0)

def LeakyReLU(X, alpha=0.3):
    return np.where(X >= 0, X, alpha * X)
    