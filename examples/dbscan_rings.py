import numpy as np
import matplotlib.pyplot as plt
from clustering.dbscan import DBSCAN

np.random.seed(0)

#create random rings dataset


RINGS = 3
MIN_RADIUS = 1
MAX_RADIUS = 5
PTRS_PER_RING = 200
THICKNESS = 0.2

POSITIONS = [[0, 0], [10, 10], [-10, -10]] # dbscan will not work fine with intersecting rings

# Initialize an empty array to store points
ptrs = np.array([]).reshape(0, 2)

for i in range(RINGS):
    radius = np.random.uniform(MIN_RADIUS, MAX_RADIUS)
    inner_radius = radius - THICKNESS / 2
    outer_radius = radius + THICKNESS / 2
    
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, PTRS_PER_RING))
    theta = np.random.uniform(0, 2 * np.pi, PTRS_PER_RING)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # noise and random position
    noise = np.random.randn(PTRS_PER_RING, 2) * 0.1
    ring = np.vstack([x, y]).T + noise + POSITIONS[i]
    
    
    ptrs = np.vstack([ptrs, ring])


ptrs = ptrs.reshape(-1, 2)

dbscan = DBSCAN(eps=1, minpts=5)

dbscan.fit(ptrs)

labels = dbscan.get_labels()

print(labels)

plt.scatter(ptrs[:, 0], ptrs[:, 1], c=labels, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
