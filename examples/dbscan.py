import numpy as np
import matplotlib.pyplot as plt
from clustering.dbscan import DBSCAN

np.random.seed(0)
cluster1 = np.random.randn(50, 2) + np.array([5, 5])
cluster2 = np.random.randn(50, 2) + np.array([-5, -5])
cluster3 = np.random.randn(50, 2) + np.array([5, -5])

data = np.vstack([cluster1, cluster2, cluster3])


dbscan = DBSCAN(eps=1.5, minpts=5)

dbscan.fit(data)

labels = dbscan.get_labels()

plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
