import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate a sample dataset
n_samples = 300
data, labels = make_blobs(n_samples=n_samples, centers=3, random_state=0)

# Perform Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(data)

# Create linkage matrix for dendrogram
linkage_matrix = linkage(data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], c=agg_clustering.labels_, cmap='rainbow')
plt.title('Agglomerative Hierarchical Clustering')

plt.subplot(122)
dendrogram(linkage_matrix, orientation='top', labels=agg_clustering.labels_)
plt.title('Dendrogram')

plt.show()
