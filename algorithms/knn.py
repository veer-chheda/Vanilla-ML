import numpy as np
from vanillaml.math import euclidean_distance, manhattan_distance

class KNN:
    
    def predict(self, X, y, point, k=None, distance='euclidean'):
        distances = []
        if distance == 'euclidean':
            for i in X:
                distance_X = euclidean_distance(i, point)
                distances.append(distance_X)
        elif distance == 'manhattan':
            for i in X:
                distance_X = manhattan_distance(i, point)
                distances.append(distance_X)
        distances = np.array(distances)
        idx = distances.argsort()
        top_k = y[idx[:k]]
        values, counts = np.unique(top_k, return_counts=True)
        idx = counts.argsort()
        return values[idx[-1]]


    def test(self):
        knn = KNN()
        training_data = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
        training_labels = np.array(['A', 'A', 'A', 'B', 'B'])
        test_point = np.array([4, 5])
        print(knn.predict(training_data, training_labels, test_point, k=3))
