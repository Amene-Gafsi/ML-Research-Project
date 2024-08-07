import numpy as np


def splitting_fn(data, labels, indices, fold_size, fold):
    # added by Nikolay
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """
    # Validation/testing data (val_data takes up the remaining 1 - fold_size % of the date)
    val_data = data[indices[int(fold * fold_size): int((fold + 1) * fold_size)], :]
    val_label = labels[indices[int(fold * fold_size): int((fold + 1) * fold_size)]]

    # Training data (train_data takes up fold_size % of all data)
    first_train_data_part = data[indices[0:int(fold * fold_size)], :]
    second_train_data_part = data[indices[int((fold + 1) * fold_size):]]
    train_data = np.concatenate((first_train_data_part, second_train_data_part), axis=0)

    first_train_label = labels[indices[0:int(fold * fold_size)]]
    second_train_label = labels[indices[int((fold + 1) * fold_size):]]
    train_label = np.concatenate((first_train_label, second_train_label), axis=0)
    return train_data, train_label, val_data, val_label


# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels


def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)


def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]), data], axis=1)
    return data


def normalize_fn(data):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    means = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - means) / std


def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.


def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels == val)

        tp = sum(predpos * gtpos)
        fp = sum(predpos * ~gtpos)
        fn = sum(~predpos * gtpos)
        if tp == 0:
            continue
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        macrof1 += 2 * (precision * recall) / (precision + recall)

    return macrof1 / len(class_ids)
