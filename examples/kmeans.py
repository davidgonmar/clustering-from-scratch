import numpy as np
import matplotlib.pyplot as plt
from clustering.kmeans import KMeans

np.random.seed(0)
cluster1 = np.random.randn(50, 2) + np.array([5, 5])
cluster2 = np.random.randn(50, 2) + np.array([-5, -5])
cluster3 = np.random.randn(50, 2) + np.array([5, -5])

data = np.vstack([cluster1, cluster2, cluster3])


kmeans = KMeans(n_clusters=3, max_iter=100, distance_metric='manhattan', init='kmeans++')
kmeans.fit(data)
centers = kmeans.centroids
labels = kmeans.labels

plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
