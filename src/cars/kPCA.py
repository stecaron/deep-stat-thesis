import os
import numpy
from comet_ml import Experiment
import torch
import argparse
import pandas
import time
import datetime
import math
import torch.utils.data as Data

from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import KernelPCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from src.cars.data import DataGenerator, define_filenames
from src.mnist.utils.pca_functions import my_scorer


def train(file):

    # Create an experiment
    experiment = Experiment(project_name="deep-stats-thesis",
                            workspace="stecaron",
                            disabled=False)
    experiment.add_tag("cars_dogs_kpca")

    # General parameters
    PATH_DATA_CARS = os.path.join(os.path.expanduser("~"), 'data/stanford_cars')
    PATH_DATA_DOGS = os.path.join(os.path.expanduser("~"), 'data/stanford_dogs2')
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Define training parameters
    hyper_params = {
        "NUM_WORKERS": 0,
        "IMAGE_SIZE": (128, 128),
        "TRAIN_SIZE": 10000,
        "TRAIN_NOISE": 0.01,
        "TEST_SIZE": 1000,
        "TEST_NOISE": 0.1,
        "ALPHA": 0.1,  # level of significance for the test
        "GAMMA": [0.001, 0.01, 0.1, 1, 10]  # hyperparameters gamma in rbf kPCA
    }

    # Log experiment parameterso0p
    experiment.log_parameters(hyper_params)

    # Define some transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        #transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Set the random seed
    numpy.random.seed(0)

    # Load data
    train_x_files, test_x_files, train_y, test_y = define_filenames(
        PATH_DATA_DOGS, PATH_DATA_CARS, hyper_params["TRAIN_SIZE"],
        hyper_params["TEST_SIZE"], hyper_params["TRAIN_NOISE"],
        hyper_params["TEST_NOISE"])

    train_data = DataGenerator(train_x_files,
                            train_y,
                            transform=transform,
                            image_size=hyper_params["IMAGE_SIZE"])

    test_data = DataGenerator(test_x_files,
                            test_y,
                            transform=transform,
                            image_size=hyper_params["IMAGE_SIZE"])

    train_loader = Data.DataLoader(dataset=train_data,
                                batch_size=hyper_params["TRAIN_SIZE"],
                                num_workers=hyper_params["NUM_WORKERS"],
                                shuffle=False)

    test_loader = Data.DataLoader(dataset=test_data,
                                batch_size=hyper_params["TEST_SIZE"],
                                num_workers=hyper_params["NUM_WORKERS"],
                                shuffle=False)

    # Flatten the data and transform to numpy array
    # X_train = next(enumerate(train_loader))[1][0].flatten().reshape(hyper_params["TRAIN_SIZE"], 3 * 128 * 128).numpy()
    # X_test = next(enumerate(test_loader))[1][0].flatten().reshape(hyper_params["TEST_SIZE"], 3 * 128 * 128).numpy()
    X_train = next(enumerate(train_loader))[1][0].flatten().reshape(
        hyper_params["TRAIN_SIZE"], 3 * 128 * 128).numpy()
    X_test = next(enumerate(test_loader))[1][0].flatten().reshape(
        hyper_params["TEST_SIZE"], 3 * 128 * 128).numpy()

    # Train kPCA
    param_grid = [{"gamma": hyper_params["GAMMA"]}]

    kpca = KernelPCA(n_components=100,
                    fit_inverse_transform=True,
                    remove_zero_eig=True,
                    kernel="rbf",
                    n_jobs=-1)
    # To know how many components to keep
    # kpca.fit(X_train)
    # var = kpca.lambdas_/numpy.sum(kpca.lambdas_)
    # cum_var = numpy.cumsum(var)
    grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer)
    grid_search.fit(X_train)
    X_kpca = grid_search.transform(X_train)
    X_train_back = grid_search.inverse_transform(X_kpca)
    X_test_back = grid_search.inverse_transform(grid_search.transform(X_test))

    # Compute the distance between original data and reconstruction
    dist_train = numpy.linalg.norm(X_train - X_train_back, axis=1)
    dist_test = numpy.linalg.norm(X_test - X_test_back, axis=1)

    # Test performances on train
    train_anomalies_ind = numpy.argsort(dist_train)[int(
        (1 - hyper_params["ALPHA"]) *
        hyper_params["TRAIN_SIZE"]):int(hyper_params["TRAIN_SIZE"])]
    train_predictions = numpy.zeros(hyper_params["TRAIN_SIZE"])
    train_predictions[train_anomalies_ind] = 1

    train_recall = metrics.recall_score(train_y, train_predictions)
    train_precision = metrics.precision_score(train_y, train_predictions)
    train_f1_score = metrics.f1_score(train_y, train_predictions)

    print(f"Train Precision: {train_precision}")
    print(f"Train Recall: {train_recall}")
    print(f"Train F1 Score: {train_f1_score}")
    experiment.log_metric("train_precision", train_precision)
    experiment.log_metric("train_recall", train_recall)
    experiment.log_metric("train_f1_score", train_f1_score)

    # Test performances on test
    test_probs = numpy.array(
        [numpy.sum(xi >= dist_train) / len(dist_train) for xi in dist_test],
        dtype=float)
    test_anomalies_ind = numpy.argwhere(test_probs >= 1 - hyper_params["ALPHA"])
    test_predictions = numpy.zeros(hyper_params["TEST_SIZE"])
    test_predictions[test_anomalies_ind] = 1

    test_recall = metrics.recall_score(test_y, test_predictions)
    test_precision = metrics.precision_score(test_y, test_predictions)
    test_f1_score = metrics.f1_score(test_y, test_predictions)
    test_auc = metrics.auc(test_y, test_probs)
    test_average_precision = metrics.average_precision_score(
        test_y, test_predictions)

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

    # Plot p-values
    preds_order = numpy.argsort(test_probs)

    x_line = numpy.arange(0, len(test_probs), step=1)
    y_line = numpy.linspace(0, 1, len(test_probs))
    y_adj = numpy.arange(0, len(test_probs),
                        step=1) / len(test_probs) * hyper_params["ALPHA"]
    zoom = int(0.2 * len(test_probs))  # nb of points to zoom

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.scatter(numpy.arange(0, len(test_probs), 1),
                test_probs[preds_order],
                c=test_y[preds_order].reshape(-1))
    ax1.plot(x_line, y_line, color="green")
    ax1.plot(x_line, y_adj, color="red")
    ax1.set_title(
        f'Entire test dataset with {int(hyper_params["TEST_NOISE"] * 100)}% of noise'
    )
    ax1.set_xticklabels([])

    ax2.scatter(numpy.arange(0, zoom, 1),
                test_probs[preds_order][0:zoom],
                c=test_y[preds_order].reshape(-1)[0:zoom])
    ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
    ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
    ax2.set_title('Zoomed in')
    ax2.set_xticklabels([])

    experiment.log_figure(figure_name="empirical_test_hypothesis",
                        figure=fig,
                        overwrite=True)
    plt.show()

    # Save the results in the output file
    col_names = ["timestamp", "precision", "recall", "f1_score",
            "average_precision", "auc"]
    if os.path.exists(file):
        df_results = pandas.read_csv(file, names=col_names, header=0)
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
                 test_auc.reshape(1))).reshape(1,-1), columns=col_names), ignore_index=True)

    df_results.to_csv(file)


def main():

    parser = argparse.ArgumentParser(
        prog="kPCA on ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--normal-digit",
        type=int,
        help="Digit number considered normal class in training",
    ),
    parser.add_argument(
        "--file",
        type=str,
        help="Filename to save the results",
    )
    args = parser.parse_args()
    train(args.file)


if __name__ == "__main__":
    main()