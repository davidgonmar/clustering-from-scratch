import numpy as np



NOISE = -1
UNCLASSIFIED = -2

class DBSCAN:
    """
    Follows the exact implementation from the paper
    """
    def _distance(self, x1: np.ndarray, x2: np.ndarray):
        # only sum over the features, not the samples
        return ((x1 - x2) ** 2).sum(-1) ** 0.5
    
    def _region_query(self, pid: int) -> np.ndarray:
        p = self.ptrs[pid]
        mask = self._distance(self.ptrs, p) < self.eps
        return np.where(mask)[0] # return point_ids of points within eps distance of p

    def _expand_cluster(self, pid: int, cluster_id: int) -> bool:
        # first, get the seeds
        seeds = self._region_query(pid)
        # check if it is a core point
        if len(seeds) < self.minpts:
            self.labels[pid] = NOISE
            return False
        # pid is a core point, so assign cluster_id to it and remove it from seeds
        self.labels[pid] = cluster_id
        seeds = seeds[seeds != pid]

        # iterate through seeds, and expand the cluster with density-reachable points
        while len(seeds) > 0:
            current_p = seeds[0]
            result = self._region_query(current_p)
            # we only want to expand with core points!!!
            if len(result) >= self.minpts:
                for i in range(len(result)):
                    result_p = result[i]
                    # if the point is not assigned to a cluster, assign it to cluster_id, and append it to seeds to search it later
                    if self.labels[result_p] == UNCLASSIFIED or self.labels[result_p] == NOISE:
                        if self.labels[result_p] == UNCLASSIFIED:
                            seeds = np.append(seeds, result_p)
                        self.labels[result_p] = cluster_id
            seeds = seeds[1:]
        return True

    def fit(self, ptrs: np.ndarray):
        assert ptrs.ndim == 2, "expected ptrs to be of shape (n_samples, n_features), got {}".format(ptrs.shape)
        assert self.fitted is False, "DBSCAN has already been fitted"

        self.ptrs = ptrs
        self.labels = np.full(ptrs.shape[0], UNCLASSIFIED)

        self.fitted = True

        cluster_id = 0
        for p in range(self.ptrs.shape[0]):
            if self.labels[p] == UNCLASSIFIED:
                if self._expand_cluster(p, cluster_id):
                    cluster_id += 1
        return self
    
    def get_labels(self):
        return self.labels
    

    def __init__(self, eps: float = 1e-05, minpts: int = 5):
        self.eps = eps
        self.fitted = False
        self.minpts = minpts
        self.labels = None

