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
from src.mnist.autoencoder import AutoEncoder, ConvAutoEncoder2
from src.mnist.utils.plot import plot_n_images, plot_outliers_idx
from src.mnist.utils.train import train_mnist
from src.mnist.outliers import lof_scoring
from src.mnist.utils.evaluate import plot_comparisons, to_img

# Create an experiment
experiment = Experiment(project_name="deep-stats-thesis", workspace="stecaron")
experiment.add_tag("mnist_conv_ae")

# General parameters
DOWNLOAD_MNIST = False
PATH_DATA = '/Users/stephanecaron/Downloads/mnist'

# Define training parameters
hyper_params = {
    "EPOCH" : 5,
    "BATCH_SIZE" : 1024,
    "LR" : 0.01,
    "CLASS_SELECTED" : 6,  # on which class we want to learn outliers
    "CLASS_CORRUPTED" : 9,
    "POURC_CORRUPTED" : 0.05,
    "POURC_OUTLIER" : 0.01
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Load data
train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

# Plot 10 randoms "4"
#plot_n_images(train_data, target=3)

# Train the autoencoder
model = ConvAutoEncoder2()
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])
loss_func = nn.MSELoss()

idx_selected = numpy.where(train_data.train_labels == hyper_params["CLASS_SELECTED"])[0]
idx_corrupted = numpy.random.choice(
    numpy.where(train_data.train_labels == hyper_params["CLASS_CORRUPTED"])[0],
    int(len(idx_selected) * hyper_params["POURC_CORRUPTED"]))
idx_final = numpy.concatenate((idx_selected, idx_corrupted))

train_data.data = train_data.data[idx_final]
train_data.targets = train_data.targets[idx_final]

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True)
model.train()
train_mnist(train_loader,
            model,
            criterion=optimizer,
            loss_func=loss_func,
            n_epoch=hyper_params["EPOCH"],
            experiment=experiment)

# Evaluate the AE with few examples
id_test = numpy.where(test_data.train_labels == hyper_params["CLASS_SELECTED"])[0]
test_data.data = test_data.data[id_test]
test_data.targets = test_data.targets[id_test]
test_loader = Data.DataLoader(dataset=test_data, batch_size=8, shuffle=True)
test_images = next(iter(test_loader))
inputs, _ = test_images
inputs = inputs.float()
model.eval()
_, decoded_images_test = model(inputs)
pic = to_img(decoded_images_test)
pic = torch.squeeze(pic, 1)
fig = plot_comparisons(torch.squeeze(inputs, 1), pic)
experiment.log_figure(figure_name="plot_comparison", figure=fig, overwrite=True)

# Run the autoencoder on training data
view_data = train_data.data.type(torch.FloatTensor)
view_data = torch.unsqueeze(view_data, 1)
encoded_data, _ = model(view_data)
encoded_data = torch.flatten(encoded_data, start_dim=1)
pandas.DataFrame(encoded_data.detach().numpy()).to_csv(
    "data/mnist/encoded_data.csv", index=False)

# Compute a score of anomaly
lof_score = lof_scoring(encoded_data.data.numpy(),
                        n_neighbors=20,
                        pourc=hyper_params["POURC_OUTLIER"])

# Plot the encoded representations with outlier score
colors = ("red", "blue")
groups = numpy.unique(lof_score).tolist()
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(
), encoded_data.data[:, 2].numpy()
for group, color in zip(groups, colors):
    idx = lof_score == group
    x, y, z = encoded_data.data[idx, 0].numpy(
    ), encoded_data.data[idx, 1].numpy(), encoded_data.data[idx, 2].numpy()
    ax.scatter(x, y, z, c=color)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()

# Show some examples of inliers and outliers
idx_inliers = numpy.random.choice(numpy.where(lof_score == 1)[0], 20)
idx_outliers = numpy.random.choice(numpy.where(lof_score == -1)[0], 5)

plot_outliers_idx(train_data.data,
                  idx_in=idx_inliers,
                  idx_out=idx_outliers,
                  shape=(5, 5))
