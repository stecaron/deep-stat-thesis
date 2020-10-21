import os
import numpy
from comet_ml import Experiment
import argparse
import torch
import datetime
import time
import pandas

import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms
import sklearn.metrics as metrics

from src.cars.data import DataGenerator, define_filenames
from src.cars.model import CarsConvVAE, SmallCarsConvVAE, SmallCarsConvVAE128, AlexNetVAE
from src.mnist.utils.train import train_mnist_vae
from src.utils.empirical_pval import compute_pval_loaders_svm
from src.utils.denormalize import denormalize


def train(folder, file, p_train, p_test):
    # Create an experiment
    experiment = Experiment(project_name="deep-stats-thesis",
                            workspace="stecaron",
                            disabled=False)
    experiment.add_tag("cars_dogs_svm")

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
        "MODEL_NAME": "vae_svm_model_cars",
        "LOAD_MODEL": False,
        "LOAD_MODEL_NAME": "vae_svm_model_cars_20200612-50dims"
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
                                  batch_size=hyper_params["BATCH_SIZE"],
                                  shuffle=False,
                                  num_workers=hyper_params["NUM_WORKERS"])

    # Load model
    model = SmallCarsConvVAE128(z_dim=hyper_params["LATENT_DIM"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

    model.to(device)

    # Train the model
    if hyper_params["LOAD_MODEL"]:
        model.load_state_dict(torch.load(
            f'{hyper_params["LOAD_MODEL_NAME"]}.h5'))
    else:
        train_mnist_vae(train_loader,
                        model,
                        criterion=optimizer,
                        n_epoch=hyper_params["EPOCH"],
                        experiment=experiment,
                        # scheduler=scheduler,
                        beta_list=hyper_params["BETA"],
                        beta_epoch=hyper_params["BETA_epoch"],
                        model_name=hyper_params["MODEL_NAME"],
                        device=device,
                        loss_type="perceptual",
                        flatten=False)

    # Compute p-values
    model.to(device)
    preds = compute_pval_loaders_svm(train_loader,
                                     test_loader,
                                     model,
                                     device=device,
                                     experiment=experiment,
                                     alpha=hyper_params["ALPHA"],
                                     flatten=False)

    index = test_data.labels

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

    sample_erros = numpy.random.choice(
        numpy.where((index != preds) & (index == 1))[0], 25)
    sample_ok = numpy.random.choice(numpy.where(
        (index == preds) & (index == 1))[0], 25)

    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(25):
        image = test_data[sample_erros[i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    experiment.log_figure(figure_name="Errors examples",
                          figure=fig,
                          overwrite=True)
    plt.show()

    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(25):
        image = test_data[sample_ok[i]][0].transpose_(0, 2)
        image = denormalize(image, MEAN, STD, device=device).numpy()
        axs[i].imshow(image)
        axs[i].axis('off')

    experiment.log_figure(figure_name="OK examples",
                          figure=fig,
                          overwrite=True)
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
                 numpy.array(numpy.nan).reshape(1))).reshape(1, -1), columns=col_names), ignore_index=True)

    df_results.to_csv(results_file)


def main():

    parser = argparse.ArgumentParser(
        prog="VAE-SVM on ImageNet",
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
