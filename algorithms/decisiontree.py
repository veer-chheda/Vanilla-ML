import numpy as np

class Node:
    def __init__(self, feature, left, right, threshold, information_gain, type='internal', pred=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.threshold = threshold
        self.type = type
        if type == 'leaf':
            self.pred = pred

class DecisionTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, X):
        counts = np.bincount(X)
        probs = counts / X.shape[0]
        entropy = 0
        for i in probs:
            if probs > 0:
                entropy += i * np.log2(i)
        return -entropy

    def _information_gain(self, parent, left, right):
        n_left = left.shape[0] / parent.shape[0]
        n_right = right.shape[0] / parent.shape[0]
        return self._entropy(parent) - (n_left * self._entropy(left)) - (n_right * self._entropy(right))

    def _best_split(self, X, y):
        best_split = {}
        best_information_gain = -1e6
        for i in range(X.shape[1]):
            X_i = X[:, i]
            for j in np.unique(X_i):
