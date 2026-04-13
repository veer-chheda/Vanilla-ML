import numpy as np
import cvxopt

class SVM:
    def __init__(self, lambda_param=0.01, learning_rate=0.001, n_iterations=100, form='primal', kernel='linear', C=1.0):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.form = form
        self.kernel_function = kernel
        self.C = C

    def kernel(self, x1, x2, type='linear', gamma=0.1):
        if type == 'linear':
            return np.dot(x1,x2)
        if type == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x1-x2)**2)

    def fit(self, X, y):
        n, m = X.shape
        y_new = np.where(y<=0, -1, 1)
        self.weights = np.zeros(m)
        self.bias = 0

        if self.form == 'primal':
            for i in range(0, self.n_iterations):
                for j in range(n):
                    point_condition = y_new[j] * (np.dot(X[j], self.weights) - self.bias)
                    if point_condition >= 1:
                        self.weights -= self.learning_rate * (2 * self.weights * self.lambda_param)
                    else:
                        self.weights -= self.learning_rate * (2 * self.weights * self.lambda_param) - np.dot(X[j], y_new[j])
                        self.bias -= self.learning_rate * y_new[j]
        else: 
            y_new = y_new.astype(float).reshape(-1,1)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = self.kernel(X[i], X[j], type=self.kernel_function)
            P = cvxopt.matrix(np.outer(y_new,y_new) * K)
            q = cvxopt.matrix(np.ones(n) * -1)
            A = cvxopt.matrix(y_new.T)
            b = cvxopt.matrix(0.0)
            G_high = np.diag(np.ones(n))
            G_low = np.diag(np.ones(n) * -1)
            G = cvxopt.matrix(np.vstack((G_low, G_high)))
            h_low = np.zeros(n)
            h_high = np.ones(n) * self.C
            h = cvxopt.matrix(np.concatenate((h_low, h_high)))

            solve = cvxopt.solvers.qp(P, q, G, h, A, b)
            alpha = np.ravel(solve['x'])
            is_support_vector = alpha > 1e-5
            self.alpha = alpha[is_support_vector]
            self.support_vector_X = X[is_support_vector]
            self.support_vector_y = y_new[is_support_vector]
            

            for i in range(len(self.alpha)):
                self.bias += self.support_vector_y[i] - np.sum(self.alpha * self.support_vector_y.flatten() * K[is_support_vector][:, i])
            self.bias /= len(self.alpha)
                

    def predict(self, x):
        if self.form == 'primal':
            return np.sign(np.dot(x, self.weights) - self.bias)
        y_pred = np.zeros(len(x))
        for i in range(len(x)):
            pred_sum = 0
            for alpha, support_vector_X, support_vector_y in zip(self.alpha, self.support_vector_X, self.support_vector_y.flatten()):
                pred_sum += alpha * support_vector_y * self.kernel(x[i], support_vector_X, type=self.kernel_function)
            y_pred[i] = pred_sum
        return np.sign(y_pred + self.bias)

