import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.covariance import MinCovDet

# Set distributions parameters
MU = [0, 0, 0]
SIGMA = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
N_OBS = 5000
NOISE_PRC = 0.025

# Simulate data
dt = numpy.random.multivariate_normal(mean=MU, cov=SIGMA, size=N_OBS)

# Plot normal data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dt[:,0], dt[:,1], dt[:,2])
plt.show()

# Add random noise
idx_noisy = numpy.random.choice(dt.shape[0], int(dt.shape[0] * NOISE_PRC))
noise = numpy.tile(numpy.random.uniform(-5, 5, len(MU)), (len(idx_noisy), 1))
dt[idx_noisy] = dt[idx_noisy] + noise

# Plot data with noise
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dt[:,0], dt[:,1], dt[:,2])
plt.show()

# Estimate new data parameters
estimated_mean = numpy.mean(dt, axis=0)
estimated_median = numpy.median(dt, axis=0)
estimated_sigma = numpy.cov(dt, rowvar=0)
estimated_robust_cov = MinCovDet(random_state=0).fit(dt)
estimated_robust_cov = estimated_robust_cov.covariance_ 

# p-value on data and plot "outliers"
