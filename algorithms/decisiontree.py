import numpy as np

class Node:
    def __init__(self, feature, left, right, threshold, information_gain, type='internal', pred=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.threshold = threshold
        self.type = type
        if type != 'internal':
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
        max_information_gain = -1e6
        for i in range(X.shape[1]):
            X_i = X[:, i]
            for threshold in np.unique(X_i):
                left = [k for k in X if k[i] <= threshold]
                left_y = [k for k in y if k[i] <= threshold]
                right = [k for k in X if k[i] > threshold]
                right_y = [k for k in y if k[i] > threshold]
                if len(left) > 0 and len(right) > 0:
                    information_gain = self._information_gain(y, left_y, right_y)
                    if information_gain > max_information_gain:
                        best_split = {
                            'feature': i,
                            'threshold': threshold,
                            'left': left,
                            'left_y': left_y,
                            'right': right,
                            'right_y': right_y,
                            'information_gain': information_gain
                        }
                        max_information_gain = information_gain
        return best_split

    def _build_tree(self, X, y, depth=0):
        n, m = X.shape
        if n >= self.min_samples_split and m <= self.max_depth:
            best_split = self._best_split(X, y)
            if best_split['information_gain'] > 0:
                left = self._build_tree(best_split['left'], best_split['left_y'], depth+1)
                right = self._build_tree(best_split['right'], best_split['right_y'], depth+1)

                return Node(feature=best_split['feature'], left=left, right=right, threshold=best_split['threshold'], information_gain=best_split['information_gain'], type='internal')

        labels = np.bincount(y)
        return Node(type='leaf', pred=np.argmax(labels))
