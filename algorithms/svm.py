import numpy as np

class SVM:
    def __init__(self, lambda_param=0.01, learning_rate=0.001, n_iterations=100):
        self.lambda_param = 0.01
        self.learning_rate = 0.001
        self.n_iterations = 100
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, m = X.shape
        y_new = np.where(y<=0, -1, 1)
        self.weights = np.zeros(m)
        self.bias = 0

        for i in range(0, self.n_iterations):
            for j in range(n):
                point_condition = y_new[j] * (np.dot(X[j], self.weights) - self.bias)
                if point_condition >= 1:
                    self.weights -= self.learning_rate * (2 * self.weights * self.lambda_param)
                else:
                    self.weights -= self.learning_rate * (2 * self.weights * self.lambda_param) - np.dot(X[j], y_new[j])
                    self.bias -= self.learning_rate * y_new[j]

    def predict(self, x):
        return np.sign(np.dot(x, self.weights) - self.bias)
