import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm


class ConstantShiftEmbedding(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Template class for Constant Shift Embedding (CSE)

    Attributes:
        PMAT (np.ndarray): Proximity matrix used for calculating the embeddings.
        S (np.ndarray): Similarity matrix.
        D (np.ndarray): Dissimilarity matrix.
        V (np.ndarray): Eigenvectors of S^~c
        eigvs (np.ndarray): Eigenvalue diagonal matrix of S^~c

    """

    def __init__(self):
        self.PMAT = None
        self.S = None
        self.D = None
        self.V = None
        self.eigvs = None
        # Add/change parameters, if necessary.

    def _center_matrix(self, arr):
        # TODO:  it's much faster to centralize subtracting from each row and column the respective mean, instead of
        # perfomring two matrix multiplications
        one = np.eye(len(arr))
        Q = one - 1 / len(arr)
        return np.linalg.multi_dot([Q, arr, Q])


    def plot_S_spectrum(self, plot_expl_var=False, values=False):
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(self.eigvs)), self.eigvs)

        if plot_expl_var:
            ax2 = ax1.twinx()
            plt.plot(np.arange(len(self.eigvs)), np.cumsum(self.eigvs) / np.sum(self.eigvs))

    def fit(self, PMAT, is_dist=False):
        """ Calculate similarity/dissimiliarity matrix and all
        the necessary variables for calculating the embeddings.

        Args:
            PMAT (np.ndarray): proximity matrix
        """
	
        if not is_dist:
             # Save data
             self.PMAT = PMAT

             # create proper dissimilarity matrix, use shortest path
             # self.D = graph.shortest_path(self.PMAT)
             self.D = - self.PMAT
        else:
            self.PMAT = None
            self.D = PMAT

        # make sure no infs in D
        inf_inds = np.where(np.isinf(self.D))
        infs = np.zeros(self.D.shape, dtype=bool)
        infs[inf_inds] = True

        inf_replace = np.max(self.D[~infs]) * 10
        self.D[infs] = inf_replace

        # get centered S^c
        Sc = - 0.5 * self._center_matrix(self.D)

        # compute D^~
        eigv = np.linalg.eigvalsh(Sc)[0]
        D = self.D - 2 * eigv * (np.ones(self.D.shape) - np.eye(len(self.D)))

        #Calculate S^~c
        self.S = - 0.5 * self._center_matrix(D)

        # Spectral Decomposition of S^~c
        eigvs, V = np.linalg.eigh(self.S)
        self.eigvs, self.V = eigvs[::-1], np.flip(V, axis=1)

        return self

    def get_embedded_vectors(self, p):
        """Return embeddings

        Args:
            p (np.ndarray): cut-off value in eigenspectrum

        Returns:
            Xp (np.ndarray): embedded vectors

        """
        Vp = self.V[:, :p]
        eigvs_p = self.eigvs[:p]
        return np.linalg.multi_dot([Vp, np.diag(np.sqrt(eigvs_p))])

    def kmeans(self, k, p):
        X = self.get_embedded_vectors(p)
        return cl.KMeans(k).fit(X), self.get_distance_matrix(X)

    def get_distance_matrix(self, X):
        ret = X[:, None] - X
        return np.einsum('ijk -> ij', ret**2)

    def plot_kmeans_dist(self, ks, p, figsize=(15, 35), vmin=0, vmax=3, vmin_orig=0, vmax_orig=3, labels=None, linthresh=0.5):
        fig, axarr = plt.subplots(len(ks), 2, figsize=figsize)
        axarr = np.atleast_2d(axarr)

        for i,k in enumerate(ks):
            kmeans, dist = self.kmeans(k, p)
            sort = np.argsort(kmeans.labels_)

            orig_dist = self.D
            orig_sorted = orig_dist[sort, :][:, sort]
            denoised_sorted = dist[sort, :][:, sort]

            im0 = axarr[i, 0].matshow(denoised_sorted, norm=SymLogNorm(vmin=vmin,
                                                                       vmax=vmax,
                                                                       linthresh=linthresh))
            im1 = axarr[i, 1].matshow(orig_sorted, norm=SymLogNorm(vmin=vmin,
                                                                   vmax=vmax,
                                                                   linthresh=linthresh))

            sorted_labels = kmeans.labels_[sort]
            diff = np.diff(sorted_labels)
            ticks = np.where(diff == 1)[0]
            yticklabels = np.array(sorted_labels[ticks] + 1, dtype=str)

            axarr[i, 0].set_yticks(ticks)
            axarr[i, 0].set_yticklabels(yticklabels)
            axarr[i, 1].set_yticks(ticks)
            axarr[i, 1].set_yticklabels(yticklabels)

            divider = make_axes_locatable(axarr[i, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im0, cax=cax)
            axarr[i, 0].set_title('k = %d, p = %d' % (k, p))

            divider = make_axes_locatable(axarr[i, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax)
            axarr[i, 1].set_title('k = %d, Original' % k)

