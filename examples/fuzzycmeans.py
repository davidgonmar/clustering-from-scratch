import numpy as np
import matplotlib.pyplot as plt
from clustering.fuzzycmeans import FuzzyCMeans

np.random.seed(0)
cluster1 = np.random.randn(50, 2) + np.array([5, 5])
cluster2 = np.random.randn(50, 2) + np.array([-5, -5])
cluster3 = np.random.randn(50, 2) + np.array([5, -5])

data = np.vstack([cluster1, cluster2, cluster3])


kmeans = FuzzyCMeans(n_clusters=3, max_iter=30000000)
kmeans.fit(data)
centers = kmeans.centroids
labels = kmeans.get_hard_labels() # get the hard labels from the membership matrix
print(labels)

plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# example 2 with better dataset from sklearn
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

kmeans = FuzzyCMeans(n_clusters=3, max_iter=30000000)

kmeans.fit(X)

centers = kmeans.centroids

labels = kmeans.get_hard_labels() # get the hard labels from the membership matrix

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# example 3 with rings dataset

from sklearn.datasets import make_circles

X, y = make_circles(n_samples=150, noise=0.01, factor=0.5, random_state=42)

kmeans = FuzzyCMeans(n_clusters=2, max_iter=30000000)

kmeans.fit(X)

centers = kmeans.centroids

labels = kmeans.get_hard_labels() # get the hard labels from the membership matrix

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')

plt.show()