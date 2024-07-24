import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
from torchinfo import summary

from src.methods.deep_network import MLP, CNN, Trainer
from src.methods.pca import PCA

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, splitting_fn


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    # normalizing data with the use of normalize_fn added by our team in utils
    xtrain = normalize_fn(xtrain)
    xtest = normalize_fn(xtest)

    # appending a bias term
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)

    ### WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        var_exp = pca_obj.find_principal_components(xtrain)
        print("current dimension is: ", args.pca_d)
        print("the explained variance is", var_exp)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn" and not args.use_cross_validation:
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = ytrain.max() + 1
        if args.nn_type == "mlp":
            model = MLP(xtrain.shape[1], n_classes)

        elif args.nn_type == "cnn":
            model = CNN(xtrain.shape[1], n_classes)

        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)
        pass

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
        pass

    elif args.method == "svm":
        method_obj = SVM(C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma, degree=args.svm_degree,
                         coef0=args.svm_coef0)
        pass

    # using cross validation for finding the best hyperparameters
    # we also shuffle and separate the training data here, this is for easier une of the fold technique
    if args.use_cross_validation:
        print("Using cross validation")
        k_fold = 4
        N = xtrain.shape[0]
        indices = np.arange(N)
        np.random.shuffle(indices)
        fold_size = N // k_fold

        if args.method == "nn" and args.nn_type == "mlp":
            max_fval = 0
            max_iters_values_mlp = np.array([100, 500, 1000])
            lr_values = np.array([1e-2, 1e-3, 1e-4, 1e-5])
            n_classes = ytrain.max() + 1


            for max_iter in max_iters_values_mlp:
                for lr in lr_values:
                    print("testing for", lr, max_iter)

                    max_fvals = []
                    for fold in range(k_fold):
                        model = MLP(xtrain.shape[1], n_classes)
                        method_obj = Trainer(model, lr, max_iter, batch_size=args.nn_batch_size)

                        (train_data, train_label, val_data, val_label) = splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                        method_obj.fit(train_data, train_label)
                        pred_labels = method_obj.predict(val_data)
                        fval = macrof1_fn(pred_labels, val_label)
                        max_fvals.append(fval)

                    mean_fval = np.mean(max_fvals)

                    if max_fval < mean_fval:
                        best_lr, best_max_iters = lr, max_iter
                        max_fval = mean_fval

            model = MLP(xtrain.shape[1], n_classes)
            summary(model)
            method_obj = Trainer(model, lr=best_lr, epochs=best_max_iters, batch_size=args.nn_batch_size)
            print("best parameters are", "lr =", best_lr, "max_iter =", best_max_iters)

        elif args.method == "nn" and args.nn_type == "cnn":
            max_fval = 0
            max_iters_values_cnn = np.array([100, 500, 1000])
            lr_values = np.array([1e-2, 1e-3, 1e-4, 1e-5])
            n_classes = ytrain.max() + 1


            for max_iter in max_iters_values_cnn:
                for lr in lr_values:
                    print("testing for", lr, max_iter)

                    max_fvals = []
                    for fold in range(k_fold):
                        model = CNN(xtrain.shape[1], n_classes)
                        method_obj = Trainer(model, lr, max_iter, batch_size=args.nn_batch_size)

                        (train_data, train_label, val_data, val_label) = splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                        method_obj.fit(train_data, train_label)
                        pred_labels = method_obj.predict(val_data)
                        fval = macrof1_fn(pred_labels, val_label)
                        max_fvals.append(fval)

                    mean_fval = np.mean(max_fvals)

                    if max_fval < mean_fval:
                        best_lr, best_max_iters = lr, max_iter
                        max_fval = mean_fval

            model = CNN(xtrain.shape[1], n_classes)
            summary(model)
            method_obj = Trainer(model, lr=best_lr, epochs=best_max_iters, batch_size=args.nn_batch_size)
            print("best parameters are", "lr =", best_lr, "max_iter =", best_max_iters)

        if args.method == "logistic_regression":
            max_fval = 0
            max_iters_values_lr = [500, 1000, 2000]
            lr_values = [1e-2, 1e-3, 1e-4, 1e-5]

            for i in range(len(max_iters_values_lr)):
                for j in range(len(lr_values)):
                    max_fvals = []
                    for fold in range(k_fold):
                        method_obj = LogisticRegression(lr_values[j], max_iters_values_lr[i])
                        (train_data, train_label, val_data, val_label) = \
                            splitting_fn(xtrain, ytrain, indices, fold_size, fold)
                        method_obj.fit(train_data, train_label)
                        pred_lables = method_obj.predict(val_data)
                        fval = macrof1_fn(pred_lables, val_label)
                        max_fvals.append(fval)
                    mean_fval = np.mean(max_fvals)
                    print(mean_fval)

                    if max_fval < mean_fval:
                        (best_k, best_max_iters) = (lr_values[j], max_iters_values_lr[i])
                        print(best_k, best_max_iters)
                        max_fval = mean_fval

                    method_obj = LogisticRegression(best_k, best_max_iters)
                    print(best_k, best_max_iters)

        if args.method == "kmeans":
            max_fval = 0
            K_values = [10]
            max_iters_values_km = [100, 200]

            for i in range(len(max_iters_values_km)):
                for j in range(len(K_values)):
                    max_fvals = []
                    for fold in range(k_fold):
                        method_obj = KMeans(K_values[j], max_iters_values_km[i])

                        (train_data, train_label, val_data, val_label) = \
                            splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                        method_obj.fit(train_data, train_label)
                        pred_lables = method_obj.predict(val_data)
                        fval = macrof1_fn(pred_lables, val_label)
                        max_fvals.append(fval)

                    mean_fval = np.mean(max_fvals)

                    if max_fval < mean_fval:
                        (best_k, best_max_iters) = (K_values[j], max_iters_values_km[i])
                        max_fval = mean_fval

            method_obj = KMeans(best_k, best_max_iters)

        if args.method == "svm":
            max_fval = 0
            C_values = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            if args.svm_kernel == "linear":
                for i in range(len(C_values)):
                    max_fvals = []
                    for fold in range(k_fold):
                        method_obj = SVM(C=C_values[i], kernel="linear")

                        (train_data, train_label, val_data, val_label) = \
                            splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                        method_obj.fit(train_data, train_label)
                        pred_lables = method_obj.predict(val_data)
                        fval = macrof1_fn(pred_lables, val_label)
                        max_fvals.append(fval)

                    mean_fval = np.mean(max_fvals)
                    print(mean_fval)

                    if max_fval < mean_fval:
                        best_C = C_values[i]
                        max_fval = mean_fval

                print(best_C)
                method_obj = SVM(C=best_C, kernel="linear")

            if args.svm_kernel == "rbf":
                gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]

                for i in range(len(C_values)):
                    for j in range(len(gamma_values)):
                        max_fvals = []
                        for fold in range(k_fold):
                            method_obj = SVM(C=C_values[i], kernel="rbf", gamma=gamma_values[j])

                            (train_data, train_label, val_data, val_label) = \
                                splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                            method_obj.fit(train_data, train_label)
                            pred_lables = method_obj.predict(val_data)
                            fval = macrof1_fn(pred_lables, val_label)
                            max_fvals.append(fval)

                        mean_fval = np.mean(max_fvals)
                        print(C_values[i], gamma_values[j], mean_fval)

                        if max_fval < mean_fval:
                            (best_C, best_gamma) = (C_values[i], gamma_values[j])
                            max_fval = mean_fval

                print(best_C, best_gamma)
                method_obj = SVM(C=best_C, kernel="rbf", gamma=best_gamma)

            if args.svm_kernel == "poly":
                degree_values = [1, 2, 3, 4, 5]
                coef0_values = [0, 10, 100, 1000, 10000]

                for i in range(len(C_values)):
                    for j in range(len(degree_values)):
                        for k in range(len(coef0_values)):
                            max_fvals = []
                            for fold in range(k_fold):
                                method_obj = SVM(C=C_values[i], kernel="poly", degree=degree_values[j],
                                                 coef0=coef0_values[k])

                                (train_data, train_label, val_data, val_label) = \
                                    splitting_fn(xtrain, ytrain, indices, fold_size, fold)

                                method_obj.fit(train_data, train_label)
                                pred_lables = method_obj.predict(val_data)
                                fval = macrof1_fn(pred_lables, val_label)
                                max_fvals.append(fval)

                            mean_fval = np.mean(max_fvals)
                            print(C_values[i], degree_values[j], coef0_values[k], mean_fval)

                            if max_fval < mean_fval:
                                (best_C, best_degree, best_coef0) = (C_values[i], degree_values[j], coef0_values[k])
                                max_fval = mean_fval

                print(best_C, best_degree, best_coef0)
                method_obj = SVM(C=best_C, kernel="poly", degree=best_degree, coef0=best_coef0)

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str,
                        help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="svm", type=str,
                        help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear",
                        help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    parser.add_argument('--use_cross_validation', action="store_true")
    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
