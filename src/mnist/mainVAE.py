import numpy
from comet_ml import Experiment
import torch
import pandas
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.mnist.data import load_mnist
from src.mnist.autoencoder import VariationalAE
from src.mnist.utils.train import train_mnist_vae
from src.mnist.utils.evaluate import to_img
from src.utils.chi2 import compute_pvalues

# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=True)
experiment.add_tag("mnist_vae")

# General parameters
DOWNLOAD_MNIST = False
PATH_DATA = '/Users/stephanecaron/Downloads/mnist'

# Define training parameters
hyper_params = {
    "EPOCH": 20,
    "BATCH_SIZE": 128,
    "LR": 0.001,
    "CLASS_SELECTED": 6,  # on which class we want to learn outliers
    "CLASS_CORRUPTED": 4,  # which class we want to corrupt our dataset with
    "POURC_CORRUPTED": 0.05,  # percentage of corruption we wasnt to induce
    "INPUT_DIM": 28 * 28,  # In the case of MNIST
    "HIDDEN_DIM": 256,  # hidden layer dimensions (before the representations)
    "LATENT_DIM": 75  # latent distribution dimensions
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Load data
train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

# Train the autoencoder

model = VariationalAE(hyper_params["INPUT_DIM"], hyper_params["HIDDEN_DIM"],
                      hyper_params["LATENT_DIM"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])
loss_func = nn.MSELoss()

idx_selected = numpy.where(
    train_data.train_labels == hyper_params["CLASS_SELECTED"])[0]
idx_corrupted = numpy.random.choice(
    numpy.where(train_data.train_labels == hyper_params["CLASS_CORRUPTED"])[0],
    int(len(idx_selected) * hyper_params["POURC_CORRUPTED"]))
idx_final = numpy.concatenate((idx_selected, idx_corrupted))

train_data.data = train_data.data[idx_final]
train_data.targets = train_data.targets[idx_final]

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True)

train_mnist_vae(train_loader,
                model,
                criterion=optimizer,
                n_epoch=hyper_params["EPOCH"],
                experiment=experiment)

# create a random latent vector
model.eval()
z = torch.randn(1, hyper_params["LATENT_DIM"])

reconstructed_img = model.decoder(z)
img = reconstructed_img.view(28, 28).data

fig = plt.figure()
plt.imshow(img, cmap='gray')
experiment.log_figure(figure_name="image_reconstruite",
                      figure=fig,
                      overwrite=True)
plt.show()

# Feed a wrong input
idx_test = numpy.random.choice(
    numpy.where(test_data.targets == hyper_params["CLASS_CORRUPTED"])[0],
    1)
test_data.data = test_data.data[idx_test]
test_data.targets = test_data.targets[idx_test]
test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
test_image = next(iter(test_loader))[0]
test_image = test_image.view(-1, 28 * 28)
reconstructed_img, _, _, _ = model(test_image)
img = reconstructed_img.view(28, 28).data
fig = plt.figure()
plt.imshow(img, cmap='gray')
experiment.log_figure(figure_name="wrong_input", figure=fig, overwrite=True)
plt.show()

# Test each encoded input
test_loader = Data.DataLoader(dataset=train_data,
                              batch_size=train_data.data.shape[0],
                              shuffle=False)
test_images, test_labels = next(iter(test_loader))
test_images = test_images.view(-1, 28 * 28)
_, z_mu, z_sigma, encoded_data = model(test_images)

# mu_6 = numpy.mean(z_mu.detach().numpy()[torch.squeeze(numpy.argwhere(train_data.targets == 6), 0)], axis=0)
# mu_4 = numpy.mean(z_mu.detach().numpy()[torch.squeeze(numpy.argwhere(train_data.targets == 4), 0)], axis=0)
# sigma_6 = numpy.mean(z_sigma.detach().numpy()[torch.squeeze(numpy.argwhere(train_data.targets == 6), 0)], axis=0)
# sigma_4 = numpy.mean(z_sigma.detach().numpy()[torch.squeeze(numpy.argwhere(train_data.targets == 4), 0)], axis=0)

# pval = compute_pvalues(z_mu.detach().numpy(),
#                        mean=numpy.median(z_mu.detach().numpy(), axis=0).reshape(1, -1),
#                        sigma=numpy.diag(numpy.exp(numpy.median(z_sigma.detach().numpy(), axis=0))))

pval = compute_pvalues(encoded_data.detach().numpy(),
                       mean=numpy.repeat(0, hyper_params["LATENT_DIM"]).reshape(1, -1),
                       sigma=numpy.identity(hyper_params["LATENT_DIM"]))

pval_sort = numpy.sort(pval, axis=0)
index = []
for i in range(test_images.shape[0]):
    pval_selected = pval_sort[i]
    index_original = int(numpy.where(pval_selected == pval)[0])
    if index_original in numpy.argwhere(train_data.targets == hyper_params["CLASS_CORRUPTED"]):
        index.append(True)
    else:
        index.append(False)

N_OBS = test_images.shape[0]
x_line = numpy.arange(0, N_OBS, step=1)
y_line = numpy.linspace(0, 1, N_OBS)
y_adj = numpy.arange(
    0, N_OBS, step=1) / N_OBS * 0.05  # 0.05 means the alpha value of my test
zoom = 40  # nb of points to zoom

ax1 = plt.subplot(2, 1, 1)
ax1.scatter(numpy.arange(0, len(pval), 1), pval_sort, c=index)
ax1.plot(x_line, y_line, color="green")
ax1.plot(x_line, y_adj, color="red")
ax1.set_title('Entire dataset')
ax1.set_xticklabels([])

ax2 = plt.subplot(2, 1, 2)
ax2.scatter(numpy.arange(0, zoom, 1), pval_sort[0:zoom], c=index[0:zoom])
ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
ax2.set_title('Zoomed in')
ax2.set_xticklabels([])

plt.show()
