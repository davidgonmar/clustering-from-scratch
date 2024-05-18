import numpy as np
import logging


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, distance_metric='euclidean', init='kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.fitted = False

        metric_str_to_func = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'cosine': self._cosine_distance
        }

        if distance_metric not in metric_str_to_func:
            raise ValueError("distance_metric must be one of {}".format(metric_str_to_func.keys()))
        
        self.distance_metric = metric_str_to_func[distance_metric]


        init_str_to_func = {
            'random': self._init_random,
            'kmeans++': self._init_kmeans_plusplus
        }

        if init not in init_str_to_func:
            raise ValueError("init must be one of {}".format(init_str_to_func.keys()))
            
        self._init_fn = init_str_to_func[init]

    def _init_random(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids randomly.
        Args:
            X: (n_samples, n_features) ndarray
        
        Returns:
            (n_clusters, n_features) ndarray: the initial centroids
        """
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def _init_kmeans_plusplus(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids using the KMeans++ algorithm.
        Args:
            X: (n_samples, n_features) ndarray
        
        Returns:
            (n_clusters, n_features) ndarray: the initial centroids
        """

        # Basically, the algorithm selects the first centroid randomly, and then selects the next centroid
        # based on the probability of each sample being chosen as the next centroid. The probability is
        # proportional to the square of the distance of the sample to the closest centroid.
        # That is, the farther a sample is from the closest centroid, the more likely it is to be chosen as the next centroid.
        n_samples = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        centroids[0] = X[np.random.choice(n_samples)] # Choose the first centroid randomly
        
        for k in range(1, self.n_clusters):
            dists = np.array([min([np.linalg.norm(x_i - y_k) ** 2 for y_k in centroids[:k]]) for x_i in X]) # shape (n_samples,)
            probs = dists / np.sum(dists)
            centroids[k] = X[np.random.choice(n_samples, p=probs)]
        
        return centroids

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two vectors.
        Args:
            x1: (n_features,) ndarray
            x2: (n_features,) ndarray
        
        Returns:
            float: the Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Manhattan distance between two vectors.
        Args:
            x1: (n_features,) ndarray
            x2: (n_features,) ndarray
        
        Returns:
            float: the Manhattan distance between x1 and x2
        """
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Cosine distance between two vectors (1 - cos_similarity(x1, x2))
        Args:
            x1: (n_features,) ndarray
            x2: (n_features,) ndarray
        
        Returns:
            float: the Cosine distance between x1 and x2
        """
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    def fit(self, X: np.ndarray):
        """
        Fit the model to the data matrix X.
        Args:
            X: (n_samples, n_features) ndarray
        
        Returns:
            self
        """
        assert len(X.shape) == 2, "X must be a 2D array with shape (n_samples, n_features), got shape {}".format(X.shape)
        if self.fitted:
            logging.warning("The model has already been fitted. Re-fitting will overwrite the previous model.")
        
        self.fitted = True
        n_samples, n_features = X.shape

        # Initialize the centroids and labels
        self.labels = np.zeros(n_samples)
        self.centroids = self._init_fn(X)

        for _ in range(self.max_iter):
            # For each sample, assign it to the closest centroid
            new_labels = np.array([np.argmin([self.distance_metric(x_i, y_k) for y_k in self.centroids]) for x_i in X])

            # Might have converged early
            if np.array_equal(self.labels, new_labels):
                logging.info("Converged early at iteration {}. Stopping optimization.".format(_))
                break

            self.labels = new_labels

            # Mean of the samples assigned to each centroid
            for k in range(self.n_clusters):
                self.centroids[k] = np.mean(X[self.labels == k], axis=0)

        return self
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        Args:
            X: (n_samples, n_features) ndarray

        Returns:
            (n_samples,) ndarray: the predicted cluster for each sample
        """
        return np.array([np.argmin([self.distance_metric(x_i, y_k) for y_k in self.centroids]) for x_i in X])
