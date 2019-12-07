import numpy
import matplotlib.pyplot as plt
from comet_ml import Experiment

from mpl_toolkits.mplot3d import Axes3D
from sklearn.covariance import MinCovDet

from src.utils.pvalues import compute_pvalues


experiment = Experiment(project_name="deep-stats-thesis",
                        workspace="stecaron", disabled=False)
experiment.add_tag("simulations_chi2")

# Set distributions parameters
hyper_params = {
    "N_DIM" : 25,
    "N_OBS" : 1000,
    "NOISE_PRC" : 0.01
}

# Log experiment parameters
experiment.log_parameters(hyper_params)

MU = numpy.repeat(0, hyper_params["N_DIM"])
SIGMA = numpy.identity(hyper_params["N_DIM"])

# Simulate data
dt = numpy.random.multivariate_normal(mean=MU, cov=SIGMA, size=hyper_params["N_OBS"])

# Plot normal data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dt[:, 0], dt[:, 1], dt[:, 2])
plt.show()

# Add random noise
idx_noisy = numpy.random.choice(dt.shape[0], int(dt.shape[0] * hyper_params["NOISE_PRC"]))
noise = numpy.tile(numpy.random.normal(2.5, 2.5, len(MU)), (len(idx_noisy), 1))
dt[idx_noisy] = dt[idx_noisy] + noise

# Plot data with noise
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(numpy.delete(dt[:, 0], idx_noisy, axis=0),
           numpy.delete(dt[:, 1], idx_noisy, axis=0),
           numpy.delete(dt[:, 2], idx_noisy, axis=0))
ax.scatter(dt[idx_noisy, 0], dt[idx_noisy, 1], dt[idx_noisy, 2], color="red")
plt.show()

# Estimate new data parameters
estimated_mean = numpy.mean(dt, axis=0).reshape(1, -1)
estimated_median = numpy.median(dt, axis=0).reshape(1, -1)
estimated_sigma = numpy.cov(dt, rowvar=False)
estimated_robust_cov = MinCovDet(random_state=0).fit(dt)
estimated_robust_cov = estimated_robust_cov.covariance_

# p-value on data and plot "outliers"
pval = compute_pvalues(dt, estimated_mean, estimated_sigma)

pval_sort = numpy.sort(pval, axis=0)
index = []
for i in range(hyper_params["N_OBS"]):
    pval_selected = pval_sort[i]
    index_original = int(numpy.where(pval_selected == pval)[0])
    if index_original in idx_noisy:
        index.append(True)
    else:
        index.append(False)

x_line = numpy.arange(0, hyper_params["N_OBS"], step=1)
y_line = numpy.linspace(0, 1, hyper_params["N_OBS"])
y_adj = numpy.arange(
    0, hyper_params["N_OBS"], step=1) / hyper_params["N_OBS"] * 0.05  # 0.05 means the alpha value of my test
zoom = 40  # nb of points to zoom

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.scatter(numpy.arange(0, len(pval), 1), pval_sort, c=index)
ax1.plot(x_line, y_line, color="green")
ax1.plot(x_line, y_adj, color="red")
ax1.set_title('Entire dataset')
ax1.set_xticklabels([])

ax2.scatter(numpy.arange(0, zoom, 1), pval_sort[0:zoom], c=index[0:zoom])
ax2.plot(x_line[0:zoom], y_line[0:zoom], color="green")
ax2.plot(x_line[0:zoom], y_adj[0:zoom], color="red")
ax2.set_title('Zoomed in')
ax2.set_xticklabels([])

experiment.log_figure(figure_name="chi2_test", figure=fig, overwrite=True)
plt.show()