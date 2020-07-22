import os
import numpy
from comet_ml import Experiment
import torch

import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms

from src.cars.data import DataGenerator, define_filenames
from src.cars.model import CarsConvAE
from src.mnist.utils.train import train_mnist
from src.utils.empirical_pval import compute_reconstruction_pval
from src.mnist.utils.stats import test_performances
from src.utils.denormalize import denormalize

# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=False)
experiment.add_tag("cars_dogs_ae")

# General parameters
PATH_DATA_CARS = os.path.join(os.path.expanduser("~"),
                              'data/stanford_cars')
PATH_DATA_DOGS = os.path.join(os.path.expanduser("~"),
                              'data/stanford_dogs2')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define training parameters
hyper_params = {
    "IMAGE_SIZE": (128, 128),
    "NUM_WORKERS": 0,
    "EPOCH": 2,
    "BATCH_SIZE": 132,
    "LR": 0.001,
    "TRAIN_SIZE": 2000,
    "TRAIN_NOISE": 0.01,
    "TEST_SIZE": 1000,
    "TEST_NOISE": 0.1,
    "ALPHA": 0.1,
    "MODEL_NAME": "classic_ae_model_cars",
    "LOAD_MODEL": False,
    "LOAD_MODEL_NAME": "vae_model_cars"
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Define some transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)])

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
model = CarsConvAE()
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

model.to(device)

# Train the model
if hyper_params["LOAD_MODEL"]:
    model.load_state_dict(torch.load(f'{hyper_params["LOAD_MODEL_NAME"]}.h5'))

train_mnist(train_loader,
            model,
            criterion=optimizer,
            n_epoch=hyper_params["EPOCH"],
            #loss_func=loss_func,
            experiment=experiment,
            device=device,
            model_name=hyper_params["MODEL_NAME"],
            perceptual_ind=True)

# Compute p-values
model.to(device)
pval, _ = compute_reconstruction_pval(train_loader, model, test_loader, device)
pval = 1 - pval #we test on the tail
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
print(f"F1-Score: {f1_score}")
print(f"AUC: {roc_auc}")
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)
experiment.log_metric("f1_score", f1_score)
experiment.log_metric("roc_auc", roc_auc)

# Show some examples

fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    image = test_data[pval_order[i]][0].transpose_(0, 2)
    image = denormalize(image, MEAN, STD, device=device).numpy()
    axs[i].imshow(image)
    axs[i].axis('off')

experiment.log_figure(figure_name="rejetcted_observations",
                      figure=fig,
                      overwrite=True)
plt.show()

fig, axs = plt.subplots(5, 5)
fig.tight_layout()
axs = axs.ravel()

for i in range(25):
    image = test_data[pval_order[int(len(pval) - 1) - i]][0].transpose_(
        0, 2)
    image = denormalize(image, MEAN, STD, device=device).numpy()
    axs[i].imshow(image)
    axs[i].axis('off')

experiment.log_figure(figure_name="better_observations",
                      figure=fig,
                      overwrite=True)
plt.show()
