import numpy
from comet_ml import Experiment
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data

from mpl_toolkits.mplot3d import Axes3D
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from scipy.stats import multivariate_normal

from src.utils.empirical_pval import compute_empirical_pval
from src.mnist.autoencoder import VariationalAE
from src.simulations.utils.dataset import MyDataset
from src.mnist.utils.train import train_mnist_vae

experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron",
                        disabled=True)
experiment.add_tag("simulations_chi2_AE")

# Set distributions parameters
hyper_params = {
    "N_DIM": 25,
    "TRAIN_SIZE": 10000,
    "TEST_SIZE": 5000,
    "TRAIN_NOISE": 0.01,
    "TEST_NOISE": 0.1,
    "EPOCH": 30,
    "BATCH_SIZE": 256,
    "LR": 0.01,
    "HIDDEN_DIM": 10,  # hidden layer dimensions (before the representations)
    "LATENT_DIM": 3,  # latent distribution dimensions
    "ALPHA": 0.05,  # alpha value of my test
    "BETA": 1
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

# Build "train" dataset
# Simulate the "majority" class
SHAPE = 1
SCALE = 1
maj_size = int(hyper_params["TRAIN_SIZE"] * (1 - hyper_params["TRAIN_NOISE"]))
simulations = numpy.random.gamma(SHAPE,
                                 SCALE,
                                 size=maj_size * hyper_params["N_DIM"])
dt_maj = simulations.reshape((maj_size, hyper_params["N_DIM"]))

# Simulate the "minority" class
min_size = int(hyper_params["TRAIN_SIZE"] * hyper_params["TRAIN_NOISE"])
MU = numpy.repeat(-2, hyper_params["N_DIM"])
SIGMA = numpy.diag(numpy.repeat(5, hyper_params["N_DIM"]))
dt_min = numpy.random.multivariate_normal(mean=MU, cov=SIGMA, size=min_size)

# Visualise complete dataset
dt = numpy.concatenate((dt_maj, dt_min))
#dt_normalized = normalize(dt)
id_maj = numpy.arange(start=0, stop=maj_size, step=1)
id_min = numpy.arange(start=maj_size, stop=dt.shape[0], step=1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dt[id_maj, 0], dt[id_maj, 1], dt[id_maj, 2])
# ax.scatter(dt[id_min, 0], dt[id_min, 1], dt[id_min, 2], color="red")
# experiment.log_figure(figure_name="plot_out_vs_in", figure=fig, overwrite=True)
# plt.show()

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
                beta = hyper_params["BETA"],
                mnist=False)

# Build "test" dataset
# Simulate the "majority" class
SHAPE = 1
SCALE = 1
maj_size = int(hyper_params["TEST_SIZE"] * (1 - hyper_params["TEST_NOISE"]))
simulations = numpy.random.gamma(SHAPE,
                                 SCALE,
                                 size=maj_size * hyper_params["N_DIM"])
dt_maj_test = simulations.reshape((maj_size, hyper_params["N_DIM"]))

# Simulate the "minority" class
min_size = int(hyper_params["TEST_SIZE"] * hyper_params["TEST_NOISE"])
MU = numpy.repeat(-5, hyper_params["N_DIM"])
SIGMA = numpy.diag(numpy.repeat(10, hyper_params["N_DIM"]))
dt_min_test = numpy.random.multivariate_normal(mean=MU,
                                               cov=SIGMA,
                                               size=min_size)

dt_test = numpy.concatenate((dt_maj_test, dt_min_test))
id_maj_test = numpy.arange(start=0, stop=maj_size, step=1)
id_min_test = numpy.arange(start=maj_size, stop=dt_test.shape[0], step=1)

pval, kld_train = compute_empirical_pval(dt, model, dt_test)
pval_order = numpy.argsort(pval)

# Plot p-values
x_line = numpy.arange(0, dt_test.shape[0], step=1)
y_line = numpy.linspace(0, 1, dt_test.shape[0])
y_adj = numpy.arange(0, dt_test.shape[0],
                     step=1) / dt_test.shape[0] * hyper_params["ALPHA"]
zoom = int(0.2 * dt_test.shape[0])  # nb of points to zoom
index = numpy.concatenate([
    numpy.repeat(False, len(id_maj_test)),
    numpy.repeat(True, len(id_min_test))
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


