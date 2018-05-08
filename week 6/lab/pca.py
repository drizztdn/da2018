import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
pca = PCA(n_components=2, whiten=True)
pca.fit(X)


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# plot data
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal')


# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# # plot data
# ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
# ax[0].axis('equal');
# ax[0].set(xlabel='x', ylabel='y', title='input')
#
# X_pca = pca.transform(X)
# ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
# draw_vector([0, 0], [0, 3], ax=ax[1])
# draw_vector([0, 0], [3, 0], ax=ax[1])
# ax[1].axis('equal')
# ax[1].set(xlabel='component 1', ylabel='component 2',
# title='principal components',
# xlim=(-5, 5), ylim=(-3, 3.1))
#
#
# plt.show()


# pca = PCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# print("original shape: ", X.shape)
# print("transformed shape:", X_pca.shape)
# X_new = pca.inverse_transform(X_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
# plt.axis('equal');
# plt.show()


digits = load_digits()
# pca = PCA(2) # project from 64 to 2 dimensions
# projected = pca.fit_transform(digits.data)
# print(digits.data.shape)
# print(projected.shape)

# pca = PCA().fit(digits.data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.show()

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
            cmap='binary', interpolation='nearest',
            clim=(0, 16))

# plot_digits(noisy)
# plt.show()

# pca = PCA(0.30).fit(noisy)
# print(pca.n_components_)
# components = pca.transform(noisy)
# filtered = pca.inverse_transform(components)
# plot_digits(filtered)
# plt.show()

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)