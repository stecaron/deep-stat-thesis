import numpy
from comet_ml import Experiment
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data

from scipy.stats import multivariate_normal

from src.mnist.autoencoder import VariationalAE
from src.simulations.utils.dataset import MyDataset
from src.mnist.utils.train import train_mnist_vae
from src.utils.kl import compute_kl_divergence


experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=True)
experiment.add_tag("experience_betas")

# Set experience parameters
hyper_params = {
    "N_DIM": 25,
    "N_OBS": 5000,
    "SIMULATIONS" : 2000,
    "EPOCH": 50,
    "BATCH_SIZE": 128,
    "LR": 0.001,
    "HIDDEN_DIM": 10,  # hidden layer dimensions (before the representations)
    "LATENT_DIM": 3,  # latent distribution dimensions
    "BETA": [0, 0.5, 1, 1.5, 2, 5]
}

experiment.log_parameters(hyper_params)

# Simulate N(0, 1)
MU = numpy.repeat(0, hyper_params["N_DIM"])
SIGMA = numpy.diag(numpy.repeat(1, hyper_params["N_DIM"]))
dt_norm = numpy.random.multivariate_normal(mean=MU, cov=SIGMA, size=hyper_params["N_OBS"] * hyper_params["SIMULATIONS"])
dt_norm = dt_norm.reshape(hyper_params["N_OBS"], hyper_params["SIMULATIONS"], hyper_params["N_DIM"])

# Simule non normals
SHAPE = 1
SCALE = 1
simulations = numpy.random.gamma(SHAPE,
                                 SCALE,
                                 size=hyper_params["N_OBS"] * hyper_params["N_DIM"])
dt_non_norm = simulations.reshape((hyper_params["N_OBS"], hyper_params["N_DIM"]))

# Entraine mes vaes (plusieurs beta)
dataset = MyDataset(dt_non_norm)
train_loader = Data.DataLoader(dataset=dataset,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True)

kl_rep = numpy.empty((len(hyper_params["BETA"]), hyper_params["N_OBS"]))
for i, beta in enumerate(hyper_params["BETA"]):
    model = VariationalAE(hyper_params["N_DIM"], hyper_params["HIDDEN_DIM"],
                        hyper_params["LATENT_DIM"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

    model.train()
    train_mnist_vae(train_loader,
                model,
                criterion=optimizer,
                n_epoch=hyper_params["EPOCH"],
                experiment=experiment,
                loss_type="mse",
                beta = beta,
                mnist=False)
    
    dt_train_torch = torch.from_numpy(dt_non_norm).float()
    generated, mu_train, logvar_train, z = model(dt_train_torch)
    mu_train = mu_train.detach().numpy()
    logvar_train = logvar_train.detach().numpy()

    kl_rep[i,] = compute_kl_divergence(mu_train, logvar_train)

# Calcule KL pour les normales simulées
kl_norm = []
for i in range(dt_norm.shape[0]):
    mu = numpy.mean(dt_norm[i], axis=0).reshape(1, -1)
    logvar = numpy.log(numpy.std(dt_norm[i], axis=0)**2).reshape(1, -1)
    kl_norm.append(compute_kl_divergence(mu, logvar))
kl_norm = numpy.array(kl_norm).reshape(-1)
kl_norm_sorted = kl_norm[numpy.argsort(kl_norm)]

# Plot les KL (y = KL_sim, x = KL_rep)

fig, axs = plt.subplots(2, 3)
axs = axs.ravel()

for i in range(len(hyper_params["BETA"])):
    kl_rep_sorted = kl_rep[i][numpy.argsort(kl_rep[i])]
    axs[i].scatter(kl_rep_sorted, kl_norm_sorted)
    axs[i].set_title(hyper_params["BETA"][i])

plt.show()

