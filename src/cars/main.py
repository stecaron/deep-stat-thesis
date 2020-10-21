import os
import numpy
from comet_ml import Experiment
import torch
import datetime
import time
import argparse
import pandas

import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms

from src.cars.data import DataGenerator, define_filenames
from src.cars.model import CarsConvVAE, SmallCarsConvVAE, SmallCarsConvVAE128, AlexNetVAE
from src.mnist.utils.train import train_mnist_vae
from src.utils.empirical_pval import compute_pval_loaders, compute_reconstruction_pval, compute_pval_loaders_mixture
from src.mnist.utils.stats import test_performances
from src.utils.denormalize import denormalize


def train(folder, file, p_train, p_test):

    # Create an experiment
    experiment = Experiment(project_name="deep-stats-thesis",
                            workspace="stecaron",
                            disabled=True)
    experiment.add_tag("cars_dogs")

    # General parameters
    PATH_DATA_CARS = os.path.join(
        os.path.expanduser("~"), 'data/stanford_cars')
    PATH_DATA_DOGS = os.path.join(
        os.path.expanduser("~"), 'data/stanford_dogs2')

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define training parameters
    hyper_params = {
        "IMAGE_SIZE": (128, 128),
        "NUM_WORKERS": 10,
        "EPOCH": 20,
        "BATCH_SIZE": 130,
        "LR": 0.001,
        "TRAIN_SIZE": 10000,
        "TRAIN_NOISE": p_train,
        "TEST_SIZE": 1000,
        "TEST_NOISE": p_test,
        "LATENT_DIM": 25,  # latent distribution dimensions
        "ALPHA": p_test,  # level of significance for the test
        "BETA_epoch": [5, 10, 15],
        "BETA": [0, 100, 10],  # hyperparameter to weight KLD vs RCL
        "MODEL_NAME": "vae_model_cars",
        "LOAD_MODEL": False,
        "LOAD_MODEL_NAME": "vae_model_carsscenario_cars_plus"
    }

    # Log experiment parameters
    experiment.log_parameters(hyper_params)

    # Define some transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        #transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

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
                                   batch_size=hyper_params["BATCH_SIZE"],
                                   shuffle=True,
                                   num_workers=hyper_params["NUM_WORKERS"])

    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=hyper_params["NUM_WORKERS"])

    # Load model
    model = SmallCarsConvVAE128(z_dim=hyper_params["LATENT_DIM"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

    model.to(device)

    model_save = os.path.join(folder, hyper_params["MODEL_NAME"] + file)

    # Train the model
    if hyper_params["LOAD_MODEL"]:
        model.load_state_dict(torch.load(
            f'{hyper_params["LOAD_MODEL_NAME"]}.h5'))
    else:
        train_mnist_vae(train_loader,
                        # test_loader,
                        model,
                        criterion=optimizer,
                        n_epoch=hyper_params["EPOCH"],
                        experiment=experiment,
                        beta_list=hyper_params["BETA"],
                        beta_epoch=hyper_params["BETA_epoch"],
                        model_name=model_save,
                        device=device,
                        loss_type="perceptual",
                        flatten=False)

    # Compute p-values
    model.to(device)
    pval, _ = compute_pval_loaders(train_loader,
                                   test_loader,
                                   model,
                                   device=device,
                                   experiment=experiment,
                                   file=file,
                                   folder=folder)

    pval = 1 - pval  # we test on the tail
    pval_order = numpy.argsort(pval)

    # Plot p-values
    x_line = numpy.arange(0, len(test_data), step=1)
    y_line = numpy.linspace(0, 1, len(test_data))
    y_adj = numpy.arange(0, len(test_data),
                         step=1) / len(test_data) * hyper_params["ALPHA"]
    zoom = int(0.2 * len(test_data))  # nb of points to zoom

    index = test_data.labels

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.scatter(numpy.arange(0, len(pval), 1),
                pval[pval_order],
                c=index[pval_order].reshape(-1))
    ax1.plot(x_line, y_line, color="green")
    ax1.axhline(hyper_params["ALPHA"], color="red")
    #ax1.plot(x_line, y_adj, color="red")
    ax1.set_ylabel(r"Score $(1 - \gamma)$")
    ax1.set_title(
        f'Jeu de données test avec {int(hyper_params["TEST_NOISE"] * 100)}% de contamination'
    )
    ax1.set_xticklabels([])

    ax2.scatter(numpy.arange(0, zoom, 1),
                pval[pval_order][0:zoom],
                c=index[pval_order].reshape(-1)[0:zoom])
    ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
    ax2.axhline(hyper_params["ALPHA"], color="red")
    #ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
    ax2.set_ylabel(r"Score $(1 - \gamma)$")
    ax2.set_title('Vue rapprochée')
    ax2.set_xticklabels([])

    experiment.log_figure(figure_name="empirical_test_hypothesis",
                          figure=fig,
                          overwrite=True)
    plt.savefig(os.path.join(folder, "pvalues_" + file + ".png"))
    plt.show()

    # Compute some stats
    precision, recall, f1_score, average_precision, roc_auc = test_performances(pval, index,
                                                                                hyper_params["ALPHA"])
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")
    print(f"Average precision: {average_precision}")
    print(f"AUC: {roc_auc}")
    experiment.log_metric("precision", precision)
    experiment.log_metric("recall", recall)
    experiment.log_metric("F1-Score", f1_score)
    experiment.log_metric("average_precision", average_precision)
    experiment.log_metric("AUC", roc_auc)

    # Show some examples

    plt.rcParams['figure.figsize'] = [10, 10]
    fig, axs = plt.subplots(4, 4)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(16):
        image = test_data[pval_order[i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    experiment.log_figure(figure_name="rejetcted_observations",
                          figure=fig,
                          overwrite=True)
    plt.savefig(os.path.join(folder, "rejected_observations_" + file + ".png"))
    plt.show()

    fig, axs = plt.subplots(4, 4)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(16):
        image = test_data[pval_order[int(
            len(pval) - 1) - i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    experiment.log_figure(figure_name="better_observations",
                          figure=fig,
                          overwrite=True)
    plt.savefig(os.path.join(folder, "better_observations_" + file + ".png"))
    plt.show()

    # Plot some errors
    preds = numpy.zeros(index.shape[0])
    preds[numpy.argwhere(pval <= hyper_params["ALPHA"])] = 1
    false_positive = numpy.where((index != preds) & (index == 1))[0]
    nb_errors = numpy.min([16, false_positive.shape[0]])

    sample_errors = numpy.random.choice(
        false_positive, nb_errors, replace=False)
    fig, axs = plt.subplots(4, 4)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(nb_errors):
        image = test_data[sample_errors[i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    plt.savefig(os.path.join(folder, "false_positive_sample_" + file + ".png"))
    plt.show()

    false_negative = numpy.where((index != preds) & (index == 0))[0]
    nb_errors = numpy.min([16, false_negative.shape[0]])

    sample_errors = numpy.random.choice(
        false_negative, nb_errors, replace=False)
    fig, axs = plt.subplots(4, 4)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(nb_errors):
        image = test_data[sample_errors[i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    plt.savefig(os.path.join(folder, "false_negative_sample_" + file + ".png"))
    plt.show()

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
                 precision.reshape(1), recall.reshape(1),
                 f1_score.reshape(1), average_precision.reshape(1),
                 roc_auc.reshape(1))).reshape(1, -1), columns=col_names), ignore_index=True)

    df_results.to_csv(results_file)


def main():

    parser = argparse.ArgumentParser(
        prog="DA-VAE on ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder to save the results",
    )
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
    train(args.folder, args.file, args.p_train, args.p_test)


if __name__ == "__main__":
    main()
    #train("", "test.csv", 0.01, 0.1)
