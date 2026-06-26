import numpy as np
import math

def CrossEntropy(y, y_pred):
    return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)