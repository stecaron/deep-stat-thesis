import os
import numpy
from comet_ml import Experiment
import torch
import pandas
import math
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from matplotlib import cm

from src.mnist.data import load_mnist
from src.mnist.vae import VariationalAE
from src.mnist.vae import ConvVAE
from src.mnist.vae import ConvLargeVAE
from src.mnist.utils.train import train_mnist_vae
from src.mnist.utils.evaluate import to_img
from src.utils.empirical_pval import compute_pval_loaders, compute_pval_loaders_mixture
from src.mnist.utils.stats import test_performances

# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=False)
experiment.add_tag("mnist_vae")

# General parameters
DOWNLOAD_MNIST = False
PATH_DATA = os.path.join(os.path.expanduser("~"),
                              'Downloads/mnist')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Define training parameters
hyper_params = {
    "EPOCH": 30,
    "BATCH_SIZE": 256,
    "NUM_WORKERS": 0,
    "LR": 0.001,
    "TRAIN_SIZE": 4000,
    "TRAIN_NOISE": 0.01,
    "TEST_SIZE": 500,
    "TEST_NOISE": 0.1,
    "CLASS_SELECTED": 7,  # on which class we want to learn outliers
    "CLASS_CORRUPTED": [0, 2, 3, 6],  # which class we want to corrupt our dataset with
    #"INPUT_DIM": 28 * 28,  # In the case of MNIST
    #"HIDDEN_DIM": 256,  # hidden layer dimensions (before the representations)
    "LATENT_DIM": 50,  # latent distribution dimensions
    "ALPHA": 0.1, # level of significance for the test
    "BETA_epoch": [5, 10, 25],
    "BETA": [0, 10, 1],  # hyperparameter to weight KLD vs RCL
    "MODEL_NAME": "mnist_vae_model",
    "LOAD_MODEL": False,
    "LOAD_MODEL_NAME": "mnist_vae_model"
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Load data
train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

# Train the autoencoder
# model = VariationalAE(hyper_params["INPUT_DIM"], hyper_params["HIDDEN_DIM"],
#                       hyper_params["LATENT_DIM"])
model = ConvLargeVAE(z_dim=hyper_params["LATENT_DIM"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])


# Build "train" and "test" datasets
id_maj_train = numpy.random.choice(
    numpy.where(numpy.isin(train_data.train_labels, hyper_params["CLASS_SELECTED"]))[0],
    int((1 - hyper_params["TRAIN_NOISE"]) * hyper_params["TRAIN_SIZE"]),
    replace=False
)
id_min_train = numpy.random.choice(
    numpy.where(numpy.isin(train_data.train_labels, hyper_params["CLASS_CORRUPTED"]))[0],
    int(hyper_params["TRAIN_NOISE"] * hyper_params["TRAIN_SIZE"]),
    replace=False
)
id_train = numpy.concatenate((id_maj_train, id_min_train))

id_maj_test = numpy.random.choice(
    numpy.where(numpy.isin(test_data.test_labels, hyper_params["CLASS_SELECTED"]))[0],
    int((1 - hyper_params["TEST_NOISE"]) * hyper_params["TEST_SIZE"]),
    replace=False
)
id_min_test = numpy.random.choice(
    numpy.where(numpy.isin(test_data.test_labels, hyper_params["CLASS_CORRUPTED"]))[0],
    int(hyper_params["TEST_NOISE"] * hyper_params["TEST_SIZE"]),
    replace=False
)
id_test = numpy.concatenate((id_min_test, id_maj_test))

train_data.data = train_data.data[id_train]
train_data.targets = train_data.targets[id_train]

test_data.data = test_data.data[id_test]
test_data.targets = test_data.targets[id_test]

train_data.targets = 1 - (train_data.targets == hyper_params["CLASS_SELECTED"]).type(torch.int32)
test_data.targets = 1 - (test_data.targets == hyper_params["CLASS_SELECTED"]).type(torch.int32)

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
else :
    train_mnist_vae(train_loader,
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
pval, _ = compute_pval_loaders_mixture(train_loader,
                               test_loader,
                               model,
                               device=device,
                               method="mean",
                               experiment=experiment)
#pval = 1 - pval #we test on the tail
pval_order = numpy.argsort(pval)

# Plot p-values
x_line = numpy.arange(0, len(test_data), step=1)
y_line = numpy.linspace(0, 1, len(test_data))
y_adj = numpy.arange(0, len(test_data),
                     step=1) / len(test_data) * hyper_params["ALPHA"]
zoom = int(0.2 * len(test_data))  # nb of points to zoom

index = numpy.isin(test_data.test_labels, hyper_params["CLASS_CORRUPTED"]).astype(int)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.scatter(numpy.arange(0, len(pval), 1),
            pval[pval_order],
            c=index[pval_order].reshape(-1))
ax1.plot(x_line, y_line, color="green")
ax1.plot(x_line, y_adj, color="red")
ax1.set_title(
    f'Entire test dataset with {int(hyper_params["TEST_NOISE"] * 100)}% of noise'
)
ax1.set_xticklabels([])

ax2.scatter(numpy.arange(0, zoom, 1),
            pval[pval_order][0:zoom],
            c=index[pval_order].reshape(-1)[0:zoom])
ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
ax2.set_title('Zoomed in')
ax2.set_xticklabels([])

experiment.log_figure(figure_name="empirical_test_hypothesis",
                      figure=fig,
                      overwrite=True)
plt.show()

# Compute some stats
precision, recall, f1_score, roc_auc = test_performances(pval, index, hyper_params["ALPHA"])
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"AUC: {roc_auc}")
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)
experiment.log_metric("f1_score", f1_score)
experiment.log_metric("auc", roc_auc)

# Show some examples

fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    image = test_data.data[pval_order[i]]
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
    image = test_data.data[pval_order[int(len(pval) - 1) - i]]
    axs[i].imshow(image, cmap='gray')
    axs[i].axis('off')

experiment.log_figure(figure_name="better_observations",
                      figure=fig,
                      overwrite=True)
plt.show()

