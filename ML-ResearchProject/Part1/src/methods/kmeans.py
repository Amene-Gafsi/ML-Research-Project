import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.centers = None
        self.K = K
        self.max_iters = max_iters

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        centers = self.centers
        if centers is None:
            centers = self.init_centers(data, self.K)
            #centers = self.kmeans_plus_plus_initialization(data, self.K)
        cluster_assignments = centers
        # Loop over the iterations
        for i in range(max_iter):
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration
            distances = self.compute_distance(data, old_centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments, self.K)

            # End of the algorithm if the centers have not moved
            if np.all(old_centers == centers):
                print(f"K-Means has converged after {i + 1} iterations!")
                break

        # Compute the final cluster assignments
        self.centers = centers
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        return centers, cluster_assignments

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns
            pred_labels (array): labels of shape (N,)
        """
        K = self.K
        self.centers = None
        # normally we  would just let the k_means algo deal with center initialisation, but for better accuracy,
        # we use the knowledge of data labels to init the cluster centers precisely. Yet this weill logically work
        # only if K is of correct value
        if len(np.unique(training_labels)) == K:
            self.centers = self.init_centers_labeled(training_data, training_labels, K)
        else:
            print("using k-means with wrong param K")

        (_, cluster_assignments) = self.k_means(training_data, K)
        centers_labeled = np.zeros((K, 1))
        for i in range(K):
            cluster_labels = training_labels[cluster_assignments == i]
            centers_labeled[i] = np.bincount(cluster_labels).argmax()

        predictions = np.zeros(len(training_labels))
        predictions = np.ndarray.ravel(centers_labeled[cluster_assignments])

        return predictions

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        distances = self.compute_distance(test_data, self.centers)
        pred_labels = self.find_closest_cluster(distances)
        return pred_labels

    def init_centers(self, data, K):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:K]
        # Use these index to select centers from data
        centers = data[random_idx[:K]]

        return centers

    def kmeans_plus_plus_initialization(self, X, K):
        """
        Initializes K-means++ by selecting the initial K centroids from the data points.

        Args:
            X (np.ndarray): The input data, with shape (N, d), where N is the number of data points and d is the number of features.
            K (int): The number of clusters.

        Returns:
            centers (np.ndarray): The initial centers, with shape (K, d).
        """
        # Select the first centroid randomly from the data points.
        centers = [X[np.random.randint(X.shape[0])]]

        # Select the remaining K-1 centroids using the K-means++ algorithm.
        for k in range(1, K):
            distances = np.zeros(X.shape[0])

            # Compute the distance between each data point and the closest centroid that has already been selected.
            for i, x in enumerate(X):
                distances[i] = np.min(np.sum((centers - x) ** 2, axis=1))

            # Select the next center with probability proportional to the square of its distance to the closest center.
            probabilities = distances / np.sum(distances)
            index = np.random.choice(X.shape[0], p=probabilities)
            centers.append(X[index])

        return np.array(centers)

    def init_centers_labeled(self,xtrain, ytrain, K):
        """
        Initializes the centers of clusters for K-means using the labeled data.

        Args:
            xtrain (np.ndarray): The input data, with shape (N, D), where N is the number of data points and D is the number of features.
            ytrain (np.ndarray): The labels for the input data, with shape (N, 1).
            K (int): The number of clusters.

        Returns:
            centers (np.ndarray): The initial centers of clusters, with shape (K, D).
        """
        N, D = xtrain.shape
        centers = np.zeros((K, D))

        # Select K points from each class and use them as the initial centers.
        for k in range(K):
            indices = np.where(ytrain == k)[0]
            center = np.mean(xtrain[indices], axis=0)
            centers[k] = center

        return centers

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]

        distances = np.zeros((N, K))
        for k in range(K):
            # Compute the euclidean distance for each data to each center
            center = centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))

        return distances

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def compute_centers(self, data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(K)])
        return centers
