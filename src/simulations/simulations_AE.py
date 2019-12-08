import numpy
from comet_ml import Experiment
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data

from mpl_toolkits.mplot3d import Axes3D
from sklearn.covariance import MinCovDet
from scipy.spatial import distance
from scipy.stats import multivariate_normal

from src.utils.chi2 import compute_pvalues
from src.mnist.autoencoder import VariationalAE
from src.simulations.utils.dataset import MyDataset
from src.mnist.utils.train import train_mnist_vae

experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=False)
experiment.add_tag("simulations_chi2_AE")

# Set distributions parameters
hyper_params = {
    "N_DIM": 20,
    "N_OBS": 2000,
    "NOISE_PRC": 0.025,
    "EPOCH": 50,
    "BATCH_SIZE": 64,
    "LR": 0.001,
    "HIDDEN_DIM": 25,  # hidden layer dimensions (before the representations)
    "LATENT_DIM": 3,  # latent distribution dimensions
    "ALPHA": 0.05 # alpha value of my test
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Simulate the "majority" class
SHAPE = 1
SCALE = 1
simulations = numpy.random.gamma(SHAPE,
                                 SCALE,
                                 size=hyper_params["N_OBS"] *
                                 hyper_params["N_DIM"])
dt_maj = simulations.reshape((hyper_params["N_OBS"], hyper_params["N_DIM"]))

# Simulate the "minority" class
MU = numpy.repeat(-5, hyper_params["N_DIM"])
SIGMA = numpy.diag(numpy.repeat(10, hyper_params["N_DIM"]))
dt_min = numpy.random.multivariate_normal(mean=MU,
                                          cov=SIGMA,
                                          size=int(hyper_params["N_OBS"] *
                                                   hyper_params["NOISE_PRC"]))

# Visualise complete dataset
dt = numpy.concatenate((dt_maj, dt_min))
id_maj = numpy.arange(start=0, stop=hyper_params["N_OBS"], step=1)
id_min = numpy.arange(start=hyper_params["N_OBS"], stop=dt.shape[0], step=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dt[id_maj, 0], dt[id_maj, 1], dt[id_maj, 2])
ax.scatter(dt[id_min, 0], dt[id_min, 1], dt[id_min, 2], color="red")
experiment.log_figure(figure_name="plot_out_vs_in", figure=fig, overwrite=True)
plt.show()

# Train VAE
model = VariationalAE(hyper_params["N_DIM"], hyper_params["HIDDEN_DIM"],
                      hyper_params["LATENT_DIM"])
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["LR"])

dataset = MyDataset(dt)
train_loader = Data.DataLoader(dataset=dataset,
                               batch_size=hyper_params["BATCH_SIZE"],
                               shuffle=True)
train_mnist_vae(train_loader,
                model,
                criterion=optimizer,
                n_epoch=hyper_params["EPOCH"],
                experiment=experiment,
                loss_type="mse",
                mnist=False)

# Encode data
dt_torch = torch.from_numpy(dt).float()
generated, z_mu, z_sigma, encoded_data = model(dt_torch)
z_mu = z_mu.detach().numpy()
z_sigma = z_sigma.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z_mu[id_maj, 0], z_mu[id_maj, 1], z_mu[id_maj, 2])
ax.scatter(z_mu[id_min, 0], z_mu[id_min, 1], z_mu[id_min, 2], color="red")
experiment.log_figure(figure_name="representations", figure=fig, overwrite=True)
plt.show()

# Compute p-values based on chi2
mu_tot = numpy.mean(z_mu, axis=0).reshape(1, -1)
var_tot = numpy.exp(numpy.mean(z_sigma, axis=0))

pval_chi2 = compute_pvalues(z_mu, mean=mu_tot, sigma=numpy.diag(var_tot))
order = numpy.argsort(pval_chi2, axis=0)

x_line = numpy.arange(0, dt.shape[0], step=1)
y_line = numpy.linspace(0, 1, dt.shape[0])
y_adj = numpy.arange(
    0, dt.shape[0],
    step=1) / dt.shape[0] * hyper_params["ALPHA"]
zoom = 50  # nb of points to zoom
index = numpy.concatenate([numpy.repeat(False, len(id_maj)), numpy.repeat(True, len(id_min))])

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.scatter(numpy.arange(0, len(pval_chi2), 1), pval_chi2[order], c=index[order].reshape(-1))
ax1.plot(x_line, y_line, color="green")
ax1.plot(x_line, y_adj, color="red")
ax1.set_title('Entire dataset')
ax1.set_xticklabels([])

ax2.scatter(numpy.arange(0, zoom, 1), pval_chi2[order][0:zoom], c=index[order].reshape(-1)[0:zoom])
ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
ax2.set_title('Zoomed in')
ax2.set_xticklabels([])

experiment.log_figure(figure_name="chi2_test", figure=fig, overwrite=True)
plt.show()
