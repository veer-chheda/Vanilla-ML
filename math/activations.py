import numpy as np
import math

class sigmoid:
    def call(self, X):
        return 1/(1 + np.exp(-X))
    
    def gradient(self, X):
        return self.call(X) * (1 - self.call(X))

class softmax:
    def call(self, X):
        e_x = np.exp(X - np.max(X, axis = -1, keepdims=True))
        return e_x / np.sum(e_x, axis = -1, keepdims=True)
    
    def gradient(self, X):
        # This took some time, reminder to create an explanation for this
        probs = self.call(X)
        diagonal = probs[..., :, None] * np.eye(probs.shape[-1])
        outer = probs[..., None, :] * probs[..., :, None]
        return diagonal - outer  


class tanh:
    def call(self, X):
        return 2 / (1 + np.exp(-2 * X)) - 1
    
    def gradient(self, X):
        return 1 - self.call(X)**2

class ReLU:
    def call(self, X):
        return np.where(X >= 0, X, 0)
    def gradient(self, X):
        return np.where(X >= 0, 1, 0)

class LeakyReLU:
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, X):
        return np.where(X >= 0, X, self.alpha * X)

    def gradient(self, X):
        return np.where(X >= 0, 1, self.alpha)
    