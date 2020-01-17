import numpy
from comet_ml import Experiment
import torch

import torch.utils.data as Data
import matplotlib.pyplot as plt

from src.cifar10.data import load_cifar10
from src.cifar10.vae import ConvLargeVAE
from src.mnist.utils.train import train_mnist_vae
from src.utils.empirical_pval import compute_empirical_pval
from src.mnist.utils.stats import test_performances


# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=False)
experiment.add_tag("cifar10_vae")

# General parameters
DOWNLOAD_CIFAR10 = False
PATH_DATA = '/Users/stephanecaron/Downloads/cifar-10'

# CIFAR-10 labels
# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck

# Define training parameters
hyper_params = {
    "EPOCH": 30,
    "BATCH_SIZE": 128,
    "LR": 0.001,
    "TRAIN_SIZE": 5000,
    "TRAIN_NOISE": 0.01,
    "TEST_SIZE": 1000,
    "TEST_NOISE": 0.1,
    "CLASS_SELECTED": 1,  # on which class we want to learn outliers
    "CLASS_CORRUPTED": [2, 3, 4],  # which class we want to corrupt our dataset with
    "LATENT_DIM": 5,  # latent distribution dimensions
    "ALPHA": 0.05, # level of significance for the test
    "BETA": 1, # hyperparameter to weight KLD vs RCL
    "MODEL_NAME": "vae_model_cifar10",
    "LOAD_MODEL": False,
    "LOAD_MODEL_NAME": "vae_model_cifar10"
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Load data
train_data, test_data = load_cifar10(PATH_DATA, download=DOWNLOAD_CIFAR10)

train_data.targets = numpy.array(train_data.targets)
test_data.targets = numpy.array(test_data.targets)

# Train the autoencoder
model = ConvLargeVAE(z_dim=hyper_params["LATENT_DIM"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

# Build "train" and "test" datasets
id_maj_train = numpy.random.choice(
    numpy.where(train_data.targets == hyper_params["CLASS_SELECTED"])[0],
    int((1 - hyper_params["TRAIN_NOISE"]) * hyper_params["TRAIN_SIZE"]),
    replace=False
)
id_min_train = numpy.random.choice(
    numpy.where(numpy.isin(train_data.targets, hyper_params["CLASS_CORRUPTED"]))[0],
    int(hyper_params["TRAIN_NOISE"] * hyper_params["TRAIN_SIZE"]),
    replace=False
)
id_train = numpy.concatenate((id_maj_train, id_min_train))

id_maj_test = numpy.random.choice(
    numpy.where(test_data.targets == hyper_params["CLASS_SELECTED"])[0],
    int((1 - hyper_params["TEST_NOISE"]) * hyper_params["TEST_SIZE"]),
    replace=False
)
id_min_test = numpy.random.choice(
    numpy.where(numpy.isin(test_data.targets, hyper_params["CLASS_CORRUPTED"]))[0],
    int(hyper_params["TEST_NOISE"] * hyper_params["TEST_SIZE"]),
    replace=False
)
id_test = numpy.concatenate((id_min_test, id_maj_test))

train_data.data = train_data.data[id_train]
train_data.targets = train_data.targets[id_train]

test_data.data = test_data.data[id_test]
test_data.targets = test_data.targets[id_test]

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True)

test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=test_data.data.shape[0],
                              shuffle=False)

# Train the model
if hyper_params["LOAD_MODEL"]:
    model = torch.load(f'hyper_params["LOAD_MODEL_NAME"].pt')
else :
    train_mnist_vae(train_loader,
                    model,
                    criterion=optimizer,
                    n_epoch=hyper_params["EPOCH"],
                    experiment=experiment,
                    beta=hyper_params["BETA"],
                    loss_type="mse",
                    flatten=False)

torch.save(model, f'hyper_params["MODEL_NAME"].pt')
model.save_weights(f'./{hyper_params["MODEL_NAME"]}.h5')
experiment.log_asset(file_data=f'./{hyper_params["MODEL_NAME"]}.h5', file_name='model.h5')

# Compute p-values
pval, _ = compute_empirical_pval(train_data.data, model, test_data.data)
pval_order = numpy.argsort(pval)

# Plot p-values
x_line = numpy.arange(0, test_data.data.shape[0], step=1)
y_line = numpy.linspace(0, 1, test_data.data.shape[0])
y_adj = numpy.arange(0, test_data.data.shape[0],
                     step=1) / test_data.data.shape[0] * hyper_params["ALPHA"]
zoom = int(0.2 * test_data.data.shape[0])  # nb of points to zoom
index = numpy.concatenate([
    numpy.repeat(True, len(id_min_test)),
    numpy.repeat(False, len(id_maj_test))
])

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

experiment.log_figure(figure_name="empirical_test_hypothesis", figure=fig, overwrite=True)
plt.show()

# Compute some stats
precision, recall = test_performances(pval, index, hyper_params["ALPHA"])
print(f"Precision: {precision}")
print(f"Recall: {recall}")
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)

# Show some examples
fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    axs[i].imshow(test_data.data[pval_order[i]], cmap='gray')
    axs[i].axis('off')

experiment.log_figure(figure_name="rejetcted_observations", figure=fig, overwrite=True)
plt.show()

fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    axs[i].imshow(test_data.data[pval_order[int(0.75 * len(pval)) + i]], cmap='gray')
    axs[i].axis('off')

experiment.log_figure(figure_name="better_observations", figure=fig, overwrite=True)
plt.show()
