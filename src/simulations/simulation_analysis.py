import numpy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.covariance import MinCovDet
from scipy.spatial import distance

# Set distributions parameters
N_DIM = 3
MU = numpy.repeat(0, N_DIM)
SIGMA = numpy.identity(N_DIM)
N_OBS = 1000
NOISE_PRC = 0.01

# Simulate data
dt = numpy.random.multivariate_normal(mean=MU, cov=SIGMA, size=N_OBS)

# Plot normal data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dt[:, 0], dt[:, 1], dt[:, 2])
plt.show()

# Add random noise
idx_noisy = numpy.random.choice(dt.shape[0], int(dt.shape[0] * NOISE_PRC))
noise = numpy.tile(numpy.random.normal(.5, 1.5, len(MU)), (len(idx_noisy), 1))
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
estimated_mean = numpy.mean(dt, axis=0)
estimated_median = numpy.median(dt, axis=0)
estimated_sigma = numpy.cov(dt, rowvar=False)
estimated_robust_cov = MinCovDet(random_state=0).fit(dt)
estimated_robust_cov = estimated_robust_cov.covariance_

# p-value on data and plot "outliers"
loss = distance.cdist(dt,
                      estimated_mean.reshape(1, -1),
                      'mahalanobis',
                      VI=numpy.linalg.inv(estimated_sigma))
loss = loss**2
pval = 1 - scipy.stats.chi2.cdf(loss, df=len(MU))  # we want 1 - Pr (X < x)
pval_sort = numpy.sort(pval, axis=0)
index = []
for i in range(N_OBS):
    pval_selected = pval_sort[i]
    index_original = int(numpy.where(pval_selected == pval)[0])
    if index_original in idx_noisy:
        index.append(True)
    else:
        index.append(False)

x_line = numpy.arange(0, N_OBS, step = 1)
y_line = numpy.linspace(0, 1, N_OBS)
y_adj = numpy.arange(0, N_OBS, step = 1)/N_OBS*0.05 # 0.05 means the alpha value of my test
zoom = 40 # nb of points to zoom

ax1 = plt.subplot(2, 1, 1)
ax1.scatter(numpy.arange(0, len(pval), 1), pval_sort, c=index)
ax1.plot(x_line, y_line, color = "green")
ax1.plot(x_line, y_adj, color = "red")
ax1.set_title('Entire dataset')
ax1.set_xticklabels([])

ax2 = plt.subplot(2, 1, 2)
ax2.scatter(numpy.arange(0, zoom, 1), pval_sort[0:zoom], c=index[0:zoom])
ax2.plot(x_line[0:zoom], y_line[0:zoom], color = "green")
ax2.plot(x_line[0:zoom], y_adj[0:zoom], color = "red")
ax2.set_title('Zoomed in')
ax2.set_xticklabels([])

plt.show()