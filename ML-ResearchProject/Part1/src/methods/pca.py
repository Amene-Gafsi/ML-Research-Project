import numpy as np


## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d

        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        # Center the data with the mean
        X_tilde = training_data - self.mean
        # Create the covariance matrix
        C = np.cov(X_tilde.T)
        # Compute the eigenvectors and eigenvalues. Hint: use np.linalg.eigh
        eigvals, eigvecs = np.linalg.eig(C)
        # Choose the top d eigenvalues and corresponding eigenvectors.
        # Sort the eigenvalues( with corresponding eigenvectors )
        # in decreasing order first.
        indices = np.arange(0, len(eigvals), 1)
        indices = ([x for _, x in sorted(zip(eigvals, indices))])[::-1]
        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        # Create matrix W and the corresponding eigen values
        W = eigvecs[:, 0:self.d]
        self.W = W
        eg = eigvals[0:self.d]

        # project the data using W
        Y = np.dot(X_tilde, W)

        # Compute the explained variance
        exvar = 100 * (np.sum(eg)) / np.sum(eigvals)
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """

        return np.dot(data, self.W)
