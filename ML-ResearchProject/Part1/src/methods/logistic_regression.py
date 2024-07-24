import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]
        C = get_n_classes(training_labels)
        self.w = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            grad = self.gradient_logistic_multi(training_data, label_to_onehot(training_labels), self.w)
            self.w = self.w - self.lr * grad
            pred_labels = self.predict(training_data)
            if np.mean(pred_labels == training_labels) == 1:
                break
            
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        y_pred = self.f_softmax(test_data, self.w)
        pred_labels = onehot_to_label(y_pred)
        return pred_labels

    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        up = np.exp(data@W)
        down = np.sum(up, axis=1, keepdims=True)
        return up/down

    def loss_logistic_multi(self, data, labels, w):
        """
        Loss function for multi class logistic regression, i.e., multi-class entropy.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value
        """
        y = self.f_softmax(data, w)
        return -np.sum(labels*np.log(y))
    
    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        return data.T.dot(self.f_softmax(data, W)-labels)
    
        