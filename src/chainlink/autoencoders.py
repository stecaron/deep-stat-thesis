import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering

import numpy
import pickle

from src.chainlink.import_data import dt
from src.chainlink.import_data import plot_chainlink
from src.utils.train import train
from src.utils.predict import predict

X = numpy.array(dt[["x", "y", "z"]])

# Define configurations

#GPU_DEVICE = False
BATCH_SIZE = 32
LEARNING_WEIGHT = 0.001
NUM_EPOCHS = 25


# Define architecture

class SimpleAE(nn.Module):

    def __init__(self):
        super(SimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 100)
        )

        self.decoder = nn.Sequential(
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Train 
train_loader = Data.DataLoader(dataset=X, batch_size=BATCH_SIZE, shuffle=True)
SimpleAE = SimpleAE()

optimizer = optim.Adam(SimpleAE.parameters(), lr=LEARNING_WEIGHT)
loss_function = nn.MSELoss()

train(SimpleAE, train_loader, optimizer=optimizer, loss_function=loss_function, num_epochs=NUM_EPOCHS)

# Predict 
embeddings = predict(SimpleAE, X).detach().numpy()
svd_embeddings = TruncatedSVD(3).fit_transform(embeddings)
svd_spectral = SpectralClustering(n_clusters=2).fit(svd_embeddings)
plot_chainlink(dt[["x", "y", "z"]], svd_spectral.labels_, show=True, animate=True, name='autoencoder')