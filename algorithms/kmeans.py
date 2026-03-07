import numpy as np
from vanillaml.math import euclidean_distance

class KMeans:
    def fit(self, X, k=None, n=None):
        rng = np.random.default_rng()
        clusters = rng.random((k, X.shape[1]))
        for i in range(n):
            cluster_idx = {i: [] for i in range(clusters.shape[0])}
            for j in range(X.shape[0]):
                min_distance = float('inf')
                min_cluster = None
                for k in range(clusters.shape[0]):
                    distance = euclidean_distance(clusters[k], X[j])
                    if distance < min_distance:
                        min_distance = distance
                        min_cluster = k
                cluster_idx[min_cluster].append(j)
            for k in range(clusters.shape[0]):
                if len(cluster_idx[k]) != 0:
                    clusters[k] = np.sum(X[cluster_idx[k]])/len(cluster_idx[k])
        return clusters
    
    def test(self):
        n = np.random.randint(100,500)
        m = np.random.randint(2,5)
        k = np.random.randint(2,5)
        rng = np.random.default_rng()
        X = rng.random((n, m))
        kmeans = KMeans()
        clusters = kmeans.fit(X, k, 100)
        print(clusters)

        
