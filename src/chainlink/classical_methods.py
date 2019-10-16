import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from src.chainlink.import_data import dt
from src.chainlink.import_data import plot_chainlink

X = numpy.array(dt[["x", "y", "z"]])

# Real data
plot_chainlink(dt[["x", "y", "z"]], dt["class"], show=True, animate=False, name='real')

# K-Means method
kmeans = KMeans(n_clusters = 2, random_state=0).fit(X)
plot_chainlink(dt[["x", "y", "z"]], kmeans.labels_, show=True, animate=False, name='kmeans')

# Hierarchical Ward
hclust_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X)
plot_chainlink(dt[["x", "y", "z"]], hclust_ward.labels_, show=True, animate=False, name='hclust')

hclust_single = AgglomerativeClustering(n_clusters=2, linkage="single").fit(X)
plot_chainlink(dt[["x", "y", "z"]], hclust_single.labels_, show=True, animate=False, name='hclust')

# Spectral clustering
spectral = SpectralClustering(n_clusters=2).fit(X)
plot_chainlink(dt[["x", "y", "z"]], spectral.labels_, show=True, animate=False, name='spectral')
