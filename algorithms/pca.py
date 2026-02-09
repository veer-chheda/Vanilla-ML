import numpy as np
from vanillaml.math import covariance_matrix

class PCA:
    def __call__(self, X, dims):
        if dims is None:
            raise ValueError('Specify dims')
        covariance = covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        index = eigenvalues.argsort()[::-1]
        dim_eigenvalues = eigenvalues[index][:dims]
        dim_eigenvectors = eigenvectors[:, index[:dims]]
        X_new = np.dot(X, dim_eigenvectors)
        return X_new
