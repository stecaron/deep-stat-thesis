import numpy
import torch
import torch.nn as nn
import torch.utils.data as Data


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from src.mnist.data import load_mnist
from src.mnist.autoencoder import AutoEncoder, ConvAutoEncoder
from src.mnist.utils.plot import plot_n_images, plot_outliers_idx
from src.mnist.utils.train import train_mnist
from src.mnist.outliers import lof_scoring
from src.mnist.utils.evaluate import plot_comparisons

# General parameters
DOWNLOAD_MNIST = False
PATH_DATA = '/Users/stephanecaron/Downloads/mnist'

# Define training parameters
EPOCH = 2
BATCH_SIZE = 64
LR = 0.005
CLASS_SELECTED = 6  # on which class we want to learn outliers
CLASS_CORRUPTED = 9
POURC_CORRUPTED = 0.05
POURC_OUTLIER=0.01

# Load data
train_data, test_data = load_mnist(PATH_DATA, download=DOWNLOAD_MNIST)

# Plot 10 randoms "4"
#plot_n_images(train_data, target=3)

# Train the autoencoder

model = ConvAutoEncoder()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss_func = nn.MSELoss()

idx_selected = numpy.where(train_data.train_labels == CLASS_SELECTED)[0]
idx_corrupted = numpy.random.choice(numpy.where(train_data.train_labels == CLASS_CORRUPTED)[0], int(len(idx_selected) * POURC_CORRUPTED))
idx_final = numpy.concatenate((idx_selected, idx_corrupted))

train_data.data = train_data.data[idx_final]
train_data.targets = train_data.targets[idx_final]
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

train_mnist(train_loader,
            model,
            criterion=optimizer,
            loss_func=loss_func,
            n_epoch=EPOCH)


# Evaluate the AE with few examples
images_test = train_data.data.type(torch.FloatTensor)[0:10,:,:]
images_test = torch.unsqueeze(images_test, 1)
_, decoded_images_test = model(images_test)
plot_comparisons(train_data.data[0:10], torch.squeeze(decoded_images_test, 1))

# Run the autoencoder on training data
view_data = train_data.data.type(torch.FloatTensor)/255.
view_data = torch.unsqueeze(view_data, 1)
encoded_data, _ = model(view_data)
encoded_data = torch.squeeze(encoded_data, 2)

# Compute a score of anomaly
lof_score = lof_scoring(encoded_data.data.numpy(), n_neighbors=20, pourc=POURC_OUTLIER)

# Plot the encoded representations with outlier score
colors = ("red", "blue")
groups = numpy.unique(lof_score).tolist()
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
for group, color in zip(groups, colors):
    idx = lof_score == group
    x, y, z = encoded_data.data[idx, 0].numpy(), encoded_data.data[idx, 1].numpy(), encoded_data.data[idx, 2].numpy()
    ax.scatter(x,y,z,c=color)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()

# Show some examples of inliers and outliers
idx_inliers = numpy.random.choice(numpy.where(lof_score == 1)[0], 20)
idx_outliers = numpy.random.choice(numpy.where(lof_score == -1)[0], 5)

plot_outliers_idx(data, idx_in=idx_inliers, idx_out=idx_outliers, shape=(5,5))