import numpy as np
import math

def MSE(y_test, y_pred):
    return np.mean(np.power(y_test - y_pred, 2))

def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))

def RMSE(y_test, y_pred):
    return np.sqrt(MSE(y_test, y_pred))

