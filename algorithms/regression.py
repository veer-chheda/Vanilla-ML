import numpy as np

class Regression:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = -(1/n_samples) * np.dot(X.T, y - y_pred)
            db = -(1/n_samples) * np.sum(y - y_pred)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_params(self):
        return self.weights, self.bias

    def test(self):
        regression = Regression(lr=0.01, n_iterations=1000)
        X = np.array([[1,2,3],[2,4,6],[3,6,9],[4,8,12]])
        y = np.array([1,2,3,4])
        regression.fit(X,y)
        print(regression.get_params())
        print(regression.predict(np.array([[7,10,14]])))

regressor = Regression()
regressor.test()