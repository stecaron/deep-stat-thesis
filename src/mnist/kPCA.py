import os
import numpy
import datetime
import time
from comet_ml import Experiment
import pandas
import argparse
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import KernelPCA, PCA
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error

from src.mnist.data import load_mnist
from src.mnist.utils.pca_functions import my_scorer, anomaly_scorer


def train(normal_digit, anomalies, folder, file, p_train, p_test):

    # Create an experiment
    experiment = Experiment(project_name="deep-stats-thesis",
                            workspace="stecaron",
                            disabled=True)
    experiment.add_tag("mnist_kpca")

    # General parameters
    DOWNLOAD_MNIST = True
    PATH_DATA = os.path.join(os.path.expanduser("~"), 'Downloads/mnist')

    # Define training parameters
    hyper_params = {
        "TRAIN_SIZE": 2000,
        "TRAIN_NOISE": p_train,
        "TEST_SIZE": 800,
        "TEST_NOISE": p_test,
        # on which class we want to learn outliers
        "CLASS_SELECTED": [normal_digit],
        # which class we want to corrupt our dataset with
        "CLASS_CORRUPTED": anomalies,
        "INPUT_DIM": 28 * 28,  # In the case of MNIST
        "ALPHA": p_test,  # level of significance for the test
        # hyperparameters gamma in rbf kPCA
        "GAMMA": [1],
        "N_COMP": [30]
    }

    # Log experiment parameterso0p
    experiment.log_parameters(hyper_params)

    # Load data
    train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

    # Normalize data
    train_data.data = train_data.data / 255.
    test_data.data = test_data.data / 255.

    # Build "train" and "test" datasets
    id_maj_train = numpy.random.choice(numpy.where(
        numpy.isin(train_data.train_labels,
                   hyper_params["CLASS_SELECTED"]))[0],
        int((1 - hyper_params["TRAIN_NOISE"]) *
            hyper_params["TRAIN_SIZE"]),
        replace=False)
    id_min_train = numpy.random.choice(numpy.where(
        numpy.isin(train_data.train_labels,
                   hyper_params["CLASS_CORRUPTED"]))[0],
        int(hyper_params["TRAIN_NOISE"] *
            hyper_params["TRAIN_SIZE"]),
        replace=False)
    id_train = numpy.concatenate((id_maj_train, id_min_train))

    id_maj_test = numpy.random.choice(numpy.where(
        numpy.isin(test_data.test_labels, hyper_params["CLASS_SELECTED"]))[0],
        int((1 - hyper_params["TEST_NOISE"]) *
            hyper_params["TEST_SIZE"]),
        replace=False)
    id_min_test = numpy.random.choice(numpy.where(
        numpy.isin(test_data.test_labels, hyper_params["CLASS_CORRUPTED"]))[0],
        int(hyper_params["TEST_NOISE"] *
            hyper_params["TEST_SIZE"]),
        replace=False)
    id_test = numpy.concatenate((id_min_test, id_maj_test))

    train_data.data = train_data.data[id_train]
    train_data.targets = train_data.targets[id_train]

    test_data.data = test_data.data[id_test]
    test_data.targets = test_data.targets[id_test]

    train_data.targets = numpy.isin(train_data.train_labels,
                                    hyper_params["CLASS_CORRUPTED"])
    test_data.targets = numpy.isin(test_data.test_labels,
                                   hyper_params["CLASS_CORRUPTED"])

    # Flatten the data and transform to numpy array
    train_data.data = train_data.data.view(-1, 28 * 28).numpy()
    test_data.data = test_data.data.view(-1, 28 * 28).numpy()

    # Train kPCA
    # param_grid = [{"gamma": hyper_params["GAMMA"],
    #                "n_components": hyper_params["N_COMP"]}]

    param_grid = [{"n_components": hyper_params["N_COMP"]}]

    # kpca = KernelPCA(fit_inverse_transform=True,
    #                  kernel="rbf",
    #                  remove_zero_eig=True,
    #                  n_jobs=-1)

    kpca = PCA()

    #my_scorer2 = make_scorer(my_scorer, greater_is_better=True)
    # grid_search = GridSearchCV(kpca, param_grid, cv=ShuffleSplit(
    #     n_splits=3), scoring=my_scorer)
    kpca.fit(train_data.data)
    X_kpca = kpca.transform(train_data.data)
    X_train_back = kpca.inverse_transform(X_kpca)
    X_test_back = kpca.inverse_transform(
        kpca.transform(test_data.data))

    # Compute the distance between original data and reconstruction
    dist_train = numpy.linalg.norm(
        train_data.data - X_train_back, ord=2, axis=1)
    dist_test = numpy.linalg.norm(test_data.data - X_test_back, ord=2, axis=1)

    # Test performances on train
    train_anomalies_ind = numpy.argsort(dist_train)[int(
        (1 - hyper_params["ALPHA"]) *
        hyper_params["TRAIN_SIZE"]):int(hyper_params["TRAIN_SIZE"])]
    train_predictions = numpy.zeros(hyper_params["TRAIN_SIZE"])
    train_predictions[train_anomalies_ind] = 1

    train_recall = metrics.recall_score(train_data.targets, train_predictions)
    train_precision = metrics.precision_score(train_data.targets,
                                              train_predictions)
    train_f1_score = metrics.f1_score(train_data.targets, train_predictions)
    train_auc = metrics.roc_auc_score(train_data.targets, train_predictions)

    print(f"Train Precision: {train_precision}")
    print(f"Train Recall: {train_recall}")
    print(f"Train F1 Score: {train_f1_score}")
    print(f"Train AUC: {train_auc}")
    experiment.log_metric("train_precision", train_precision)
    experiment.log_metric("train_recall", train_recall)
    experiment.log_metric("train_f1_score", train_f1_score)
    experiment.log_metric("train_auc", train_auc)

    # Test performances on test
    test_probs = numpy.array(
        [numpy.sum(xi >= dist_train) / len(dist_train) for xi in dist_test],
        dtype=float)
    test_anomalies_ind = numpy.argwhere(
        test_probs >= 1 - hyper_params["ALPHA"])
    test_predictions = numpy.zeros(hyper_params["TEST_SIZE"])
    test_predictions[test_anomalies_ind] = 1

    test_recall = metrics.recall_score(test_data.targets, test_predictions)
    test_precision = metrics.precision_score(test_data.targets,
                                             test_predictions)
    test_f1_score = metrics.f1_score(test_data.targets, test_predictions)
    test_auc = metrics.roc_auc_score(test_data.targets, test_probs)
    test_average_precision = metrics.average_precision_score(
        test_data.targets, test_predictions)

    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1 Score: {test_f1_score}")
    print(f"Test AUC: {test_auc}")
    print(f"Test average Precision: {test_average_precision}")
    experiment.log_metric("test_precision", test_precision)
    experiment.log_metric("test_recall", test_recall)
    experiment.log_metric("test_f1_score", test_f1_score)
    experiment.log_metric("test_auc", test_auc)
    experiment.log_metric("test_average_precision", test_average_precision)

    # Save the results in the output file
    col_names = ["timestamp", "precision", "recall", "f1_score",
                 "average_precision", "auc"]
    results_file = os.path.join(folder, "results_" + file + ".csv")
    if os.path.exists(results_file):
        df_results = pandas.read_csv(results_file, names=col_names, header=0)
    else:
        df_results = pandas.DataFrame(columns=col_names)

    df_results = df_results.append(
        pandas.DataFrame(
            numpy.concatenate(
                (numpy.array(
                    datetime.datetime.fromtimestamp(
                        time.time()).strftime('%Y-%m-%d %H:%M:%S')).reshape(1),
                 test_precision.reshape(1), test_recall.reshape(1),
                 test_f1_score.reshape(1), test_average_precision.reshape(1),
                 test_auc.reshape(1))).reshape(1, -1), columns=col_names), ignore_index=True)

    df_results.to_csv(results_file)


def main():

    parser = argparse.ArgumentParser(
        prog="kPCA on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--normal-digit",
        type=int,
        help="Digit number considered normal class in training",
    ),
    parser.add_argument(
        "--anomalies",
        type=int,
        nargs='+',
        help="Digit number considered anomalies class in training",
    ),
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder to save the results",
    ),
    parser.add_argument(
        "--file",
        type=str,
        help="Filename to save the results",
    )
    parser.add_argument(
        "--p_train",
        type=float,
        help="Proportion of anomalies in train",
    )
    parser.add_argument(
        "--p_test",
        type=float,
        help="Proportion of anomalies in test",
    )
    args = parser.parse_args()
    train(args.normal_digit, args.anomalies, args.folder,
          args.file, args.p_train, args.p_test)


if __name__ == "__main__":
    main()
    #train(normal_digit=6, anomalies=[8], folder="", file="kpca", p_train=0.05, p_test=0.05)
