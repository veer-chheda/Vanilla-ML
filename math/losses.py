import numpy as np
import math

class CrossEntropy:
    def loss(self, y, y_pred):
        return - np.sum(y * np.log(y_pred)) / y.shape[0]
    
    def gradient(self, y, y_pred):
        return - (y - y_pred) / y.shape[0]