import numpy as np

def covariance_matrix(A, B=None):
    if B is None:
        B=A
    covariance = np.array((1/(A.shape[0] - 1) ) * np.dot((A - A.mean(axis=0)).T, B - B.mean(axis=0)))
    return covariance

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

def manhattan_distance(x,y):
    return np.abs(np.sum(x-y))
