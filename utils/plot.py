import matplotlib.pyplot as plt
from vanillaml.algorithms import PCA
from mpl_toolkits import mplot3d

def plot_linear_regression(X, y, dim=2):
    if X.shape[0] == 1:
        plt.plot(X, y, marker='o', line='-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.show()
    elif X.shape[0] == 2:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.show()
    elif X.shape[0] > 2:
        if dim == 2:
            pca = PCA()
            X_pca = pca(X, dims=2)
            plt.figure()
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Linear Regression')
            plt.show()
        elif dim == 3:
            xlabel = 'X'
            ylabel = 'Y'
            zlabel = 'Z'
            if X.shape[0] > 3:
                pca = PCA()
                X = pca(X, dims=3)
                xlabel = 'PC1'
                ylabel = 'PC2'
                zlabel = 'PC3'
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=y,
                alpha=0.7
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.show()

