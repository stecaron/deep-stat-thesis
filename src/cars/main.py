import os
import numpy
from comet_ml import Experiment
import torch

import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms

from src.cars.data import DataGenerator
from src.cars.model import CarsConvVAE
from src.mnist.utils.train import train_mnist_vae
from src.utils.empirical_pval import compute_pval_loaders
from src.mnist.utils.stats import test_performances

# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=True)
experiment.add_tag("cars_dogs")

# General parameters
PATH_DATA_CARS = os.path.join(os.path.expanduser("~"), 'Downloads/stanford_cars')
PATH_DATA_DOGS = os.path.join(os.path.expanduser("~"), 'Downloads/stanford_dogs')

# Define training parameters
hyper_params = {
    "IMAGE_SIZE": (224, 224),
    "GPU": False,
    "NUM_WORKERS": 4,
    "EPOCH": 3,
    "BATCH_SIZE": 128,
    "LR": 0.001,
    "TRAIN_SIZE": 5000,
    "TRAIN_NOISE": 0.01,
    "TEST_SIZE": 100,
    "TEST_NOISE": 0.1,
    "LATENT_DIM": 5,  # latent distribution dimensions
    "ALPHA": 0.05,  # level of significance for the test
    "BETA": 1,  # hyperparameter to weight KLD vs RCL
    "MODEL_NAME": "vae_model_cars",
    "LOAD_MODEL": False,
    "LOAD_MODEL_NAME": "vae_model_cars"
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Define some transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_data = DataGenerator(PATH_DATA_DOGS,
                           PATH_DATA_CARS,
                           size=hyper_params["TRAIN_SIZE"],
                           noise_perc=hyper_params["TRAIN_NOISE"],
                           transform=transform,
                           image_size=hyper_params["IMAGE_SIZE"])

test_data = DataGenerator(PATH_DATA_DOGS,
                          PATH_DATA_CARS,
                          size=hyper_params["TEST_SIZE"],
                          noise_perc=hyper_params["TEST_NOISE"],
                          transform=transform,
                          image_size=hyper_params["IMAGE_SIZE"])

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True,
                               num_workers=hyper_params["NUM_WORKERS"])

test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=len(test_data),
                              shuffle=False,
                              num_workers=hyper_params["NUM_WORKERS"])

# Load model
model = CarsConvVAE(z_dim=hyper_params["LATENT_DIM"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

if hyper_params["GPU"]:
    model.cuda()

# Train the model
if hyper_params["LOAD_MODEL"]:
    model.load_state_dict(torch.load(f'{hyper_params["LOAD_MODEL_NAME"]}.h5'))
else :
    train_mnist_vae(train_loader,
                    model,
                    criterion=optimizer,
                    n_epoch=hyper_params["EPOCH"],
                    experiment=experiment,
                    beta=hyper_params["BETA"],
                    gpu=hyper_params["GPU"],
                    loss_type="mse",
                    flatten=False)

if not hyper_params["LOAD_MODEL"]:
    torch.save(model, f'{hyper_params["MODEL_NAME"]}.pt')
    model.save_weights(f'./{hyper_params["MODEL_NAME"]}.h5')
    experiment.log_asset(file_data=f'./{hyper_params["MODEL_NAME"]}.h5', file_name='model.h5')

# Compute p-values
if not hyper_params["GPU"]:
    model.cpu()
pval, _ = compute_pval_loaders(train_loader, test_loader, model, gpu=hyper_params["GPU"])
pval_order = numpy.argsort(pval)

# Plot p-values
x_line = numpy.arange(0, len(test_data), step=1)
y_line = numpy.linspace(0, 1, len(test_data))
y_adj = numpy.arange(0, len(test_data),
                     step=1) / len(test_data) * hyper_params["ALPHA"]
zoom = int(0.2 * len(test_data))  # nb of points to zoom
index = numpy.concatenate([
    numpy.repeat(True, numpy.sum(test_data.labels == 1)),
    numpy.repeat(False, numpy.sum(test_data.labels == 0))
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
    axs[i].imshow(test_data[pval_order[i]][0].transpose_(0,2).numpy())
    axs[i].axis('off')

experiment.log_figure(figure_name="rejetcted_observations", figure=fig, overwrite=True)
plt.show()

fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    axs[i].imshow(test_data[pval_order[int(0.75 * len(pval)) + i]][0].transpose_(0,2).numpy())
    axs[i].axis('off')

experiment.log_figure(figure_name="better_observations", figure=fig, overwrite=True)
plt.show()
