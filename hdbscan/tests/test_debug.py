import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from hdbscan import (
	HDBSCAN,
	approximate_predict
)
from sklearn.datasets import make_circles

clusterer = HDBSCAN(min_cluster_size=20, min_samples = 5, prediction_data=True, algorithm='prims_kdtree')
data, _ = make_circles(n_samples=100, factor=0.4, noise=0.05)
print("data", data)
X_new = np.random.rand(5, 2) * 2 - 1
cluster_labels = clusterer.fit_predict(data)
print(cluster_labels)
label_counts = Counter(cluster_labels)

# # Display frequencies
# for label, count in sorted(label_counts.items()):
#     print(f"Cluster {label}: {count} points")
# if prediction_data:
# 	labels, strengths = approximate_predict(clusterer, X_new)

# 	print("Cluster labels:", labels)
# 	print("Membership strengths:", strengths)

# # Plot clusters
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
# plt.colorbar(scatter, label="Cluster Labels")

# if prediction_data:
# 	plt.scatter(X_new[:, 0], X_new[:, 1], c='red', edgecolors='k', marker='o', s=100, label="New Points")

# plt.title("HDBSCAN Clustering")
# plt.legend()
# plt.show()