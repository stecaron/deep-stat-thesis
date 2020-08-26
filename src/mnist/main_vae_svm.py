import os
import numpy
from comet_ml import Experiment
import torch
import pandas
import datetime
import time
import argparse
import math
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.metrics as metrics

from src.mnist.data import load_mnist, load_mnist_fashion
from src.mnist.vae import VariationalAE
from src.mnist.vae import ConvVAE
from src.mnist.vae import ConvLargeVAE
from src.mnist.utils.train import train_mnist_vae
from src.mnist.utils.evaluate import to_img
from src.utils.empirical_pval import compute_pval_loaders_svm


def train(normal_digit, anomalies, file):

    # Create an experiment
    experiment = Experiment(project_name="deep-stats-thesis",
                            workspace="stecaron",
                            disabled=True)
    experiment.add_tag("mnist_vae_svm")

    # General parameters
    DOWNLOAD_MNIST = True
    PATH_DATA = os.path.join(os.path.expanduser("~"), 'Downloads/mnist')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # Define training parameters
    hyper_params = {
        "EPOCH": 75,
        "BATCH_SIZE": 500,
        "NUM_WORKERS": 0,
        "LR": 0.001,
        "TRAIN_SIZE": 4000,
        "TRAIN_NOISE": 0.01,
        "TEST_SIZE": 1000,
        "TEST_NOISE": 0.1,
        # on which class we want to learn outliers
        "CLASS_SELECTED": [normal_digit],
        # which class we want to corrupt our dataset with
        "CLASS_CORRUPTED": anomalies,
        "INPUT_DIM": 28 * 28,  # In the case of MNIST
        "HIDDEN_DIM": 500,  # hidden layer dimensions (before the representations)
        "LATENT_DIM": 25,  # latent distribution dimensions
        "ALPHA": 0.1,  # level of significance for the test
        "BETA_epoch": [5, 10, 25],
        "BETA": [0, 5, 1],  # hyperparameter to weight KLD vs RCL
        "MODEL_NAME": "mnist_vae_svm_model",
        "LOAD_MODEL": True,
        "LOAD_MODEL_NAME": "mnist_vae_svm_model"
    }

    # Log experiment parameterso0p
    experiment.log_parameters(hyper_params)

    # Set the random seed
    numpy.random.seed(0)

    # Load data
    train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

    # Train the autoencoder
    # model = VariationalAE(hyper_params["INPUT_DIM"], hyper_params["HIDDEN_DIM"],
    #                     hyper_params["LATENT_DIM"])
    model = ConvLargeVAE(z_dim=hyper_params["LATENT_DIM"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

    # Build "train" and "test" datasets
    id_maj_train = numpy.random.choice(numpy.where(
        numpy.isin(train_data.train_labels, hyper_params["CLASS_SELECTED"]))[0],
                                    int((1 - hyper_params["TRAIN_NOISE"]) *
                                        hyper_params["TRAIN_SIZE"]),
                                    replace=False)
    id_min_train = numpy.random.choice(numpy.where(
        numpy.isin(train_data.train_labels, hyper_params["CLASS_CORRUPTED"]))[0],
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

    train_data.targets = torch.from_numpy(
        numpy.isin(train_data.train_labels,
                hyper_params["CLASS_CORRUPTED"])).type(torch.int32)
    test_data.targets = torch.from_numpy(
        numpy.isin(test_data.test_labels,
                hyper_params["CLASS_CORRUPTED"])).type(torch.int32)

    train_loader = Data.DataLoader(dataset=train_data,
                                batch_size=hyper_params["BATCH_SIZE"],
                                shuffle=True,
                                num_workers=hyper_params["NUM_WORKERS"])

    test_loader = Data.DataLoader(dataset=test_data,
                                batch_size=test_data.data.shape[0],
                                shuffle=False,
                                num_workers=hyper_params["NUM_WORKERS"])

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyper_params["LR"], steps_per_epoch=len(train_loader), epochs=hyper_params["EPOCH"])

    if hyper_params["LOAD_MODEL"]:
        model = torch.load(hyper_params["LOAD_MODEL_NAME"])
    else:
        train_mnist_vae(
            train_loader,
            model,
            criterion=optimizer,
            n_epoch=hyper_params["EPOCH"],
            experiment=experiment,
            #scheduler=scheduler,
            beta_list=hyper_params["BETA"],
            beta_epoch=hyper_params["BETA_epoch"],
            model_name=hyper_params["MODEL_NAME"],
            device=device,
            loss_type="binary",
            flatten=False)

    # Compute p-values
    model.to(device)
    preds = compute_pval_loaders_svm(train_loader,
                                    test_loader,
                                    model,
                                    device=device,
                                    experiment=experiment,
                                    flatten=False)

    index = numpy.array(test_data.targets).astype(int)

    # Compute some stats
    precision = metrics.precision_score(index, preds)
    recall = metrics.recall_score(index, preds)
    f1_score = metrics.f1_score(index, preds)
    average_precision = metrics.average_precision_score(index, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Average Precision: {average_precision}")
    #print(f"AUC: {roc_auc}")
    experiment.log_metric("precision", precision)
    experiment.log_metric("recall", recall)
    experiment.log_metric("f1_score", f1_score)
    experiment.log_metric("average_precision", average_precision)
    #experiment.log_metric("auc", roc_auc)

    # Show some examples

    sample_erros = numpy.random.choice(numpy.where((index != preds) & (index == 1))[0], 25)
    sample_ok = numpy.random.choice(numpy.where((index == preds) & (index == 1))[0], 25)

    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(25):
        image = test_data.data[sample_erros[i]]
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')

    experiment.log_figure(figure_name="rejetcted_observations",
                        figure=fig,
                        overwrite=True)
    plt.show()

    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(25):
        image = test_data.data[sample_ok[i]]
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')

    experiment.log_figure(figure_name="better_observations",
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
                 precision.reshape(1), recall.reshape(1),
                 f1_score.reshape(1), average_precision.reshape(1),
                 numpy.array(numpy.nan).reshape(1))).reshape(1,-1), columns=col_names), ignore_index=True)

    df_results.to_csv(file)


def main():

    parser = argparse.ArgumentParser(
        prog="VAE-SVM on MNIST",
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
        "--file",
        type=str,
        help="Filename to save the results",
    )
    args = parser.parse_args()
    train(args.normal_digit, args.anomalies, args.file)


if __name__ == "__main__":
    main()
    #train([6], [3, 8], "train.csv")