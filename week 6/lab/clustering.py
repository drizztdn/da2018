import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50);


from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50,
#     cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1],
#     c='black', s=200, alpha=0.5);
#
# labels = KMeans(6, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#     s=50, cmap='viridis')

# plt.show()

from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#     s=50, cmap='viridis')
# plt.show()


from sklearn.cluster import SpectralClustering
# model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
#     assign_labels='kmeans')
# labels = model.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#     s=50, cmap='viridis')
# plt.show()


from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)
# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
    # axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
# plt.show()



from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target, labels))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
    xticklabels=digits.target_names,
    yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()
