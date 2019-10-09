import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from src.chainlink.import_data import dt
from src.chainlink.import_data import plot_chainlink

X = numpy.array(dt[["x", "y", "z"]])

# Real data
plot_chainlink(dt[["x", "y", "z"]], dt["class"], show=True)

# K-Means method
kmeans = KMeans(n_clusters = 2, random_state=0).fit(X)
plot_chainlink(dt[["x", "y", "z"]], kmeans.labels_, show=True)

# Hierarchical Ward
hclust_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X)
plot_chainlink(dt[["x", "y", "z"]], hclust_ward.labels_, show=True)

# Spectral clustering
spectral = SpectralClustering(n_clusters=2).fit(X)
plot_chainlink(dt[["x", "y", "z"]], spectral.labels_, show=True)
