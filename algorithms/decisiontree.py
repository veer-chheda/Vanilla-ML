import numpy as np

class Node:
    def __init__(self, feature=None, left=None, right=None, threshold=None, information_gain=None, type='internal', pred=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.threshold = threshold
        self.type = type
        self.pred = pred

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, X):
        counts = np.bincount(X)
        probs = counts / X.shape[0]
        entropy = 0
        for i in probs:
            if i > 0:
                entropy += i * np.log2(i)
        return -entropy

    def _information_gain(self, parent, left, right):
        n_left = left.shape[0] / parent.shape[0]
        n_right = right.shape[0] / parent.shape[0]
        return self._entropy(parent) - (n_left * self._entropy(left)) - (n_right * self._entropy(right))

    def _best_split(self, X, y):
        best_split = {}
        max_information_gain = -1
        for i in range(X.shape[1]):
            X_i = X[:, i]
            for threshold in X_i:
                left = X[:, i] <= threshold
                right = X[:, i] > threshold
                if len(left) > 0 and len(right) > 0:
                    information_gain = self._information_gain(y, y[left], y[right])
                    if information_gain > max_information_gain:
                        best_split = {
                            'feature': i,
                            'threshold': threshold,
                            'left': X[left],
                            'left_y': y[left],
                            'right': X[right],
                            'right_y': y[right],
                            'information_gain': information_gain
                        }
                        max_information_gain = information_gain
        return best_split

    def _build_tree(self, X, y, depth=0):
        n, m = X.shape
        if n >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._best_split(X, y)
            if best_split['information_gain'] > 0:
                left = self._build_tree(best_split['left'], best_split['left_y'], depth+1)
                right = self._build_tree(best_split['right'], best_split['right_y'], depth+1)

                return Node(feature=best_split['feature'], left=left, right=right, threshold=best_split['threshold'], information_gain=best_split['information_gain'], type='internal')

        labels = np.bincount(y)
        return Node(type='leaf', pred=np.argmax(labels))

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict(self, X, tree):
        if tree.pred != None:
            return tree.pred
        feature = X[tree.feature]
        if feature >= tree.threshold:
            return self._predict(X=X, tree = tree.right)
        else:
            return self._predict(X=X, tree=tree.left)

    def predict(self, X):
        preds = []
        for i in X:
            val = self._predict(i, self.root)
            preds.append(val)
        return preds
