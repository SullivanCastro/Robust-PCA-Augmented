import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import json
from sklearn.preprocessing import StandardScaler


class Robust_PCA_augmented():

    def __init__(self, M, gamma=0) -> None:
        scaler = StandardScaler()
        M = scaler.fit_transform(M)
        self.M           = M
        self.k           = 0
        self.gamma       = gamma
        self.r1          = 1
        self.r2          = 1
        self.epsilon     = 1e-7
        self._lambda     = 1/np.sqrt(np.max(M.shape))
        self.n_neighbors = 10
        self.corruption  = False
        self.corruption_type = 'occlusion'

        self.laplacien = self.laplacian_computation(self.M)

        self.L           = np.zeros(M.shape)
        self.W           = np.zeros(M.shape)
        self.S           = np.zeros(M.shape)
        self.Z1          = self.M - self.L - self.S
        self.Z2          = self.W - self.L
        self.P1          = self._nulcear_norm(self.L)
        self.P2          = self._lambda * np.sum(np.abs(self.S))
        self.P3          = self.gamma * np.trace(self.L.T @ self.laplacien @ self.L)

    def pairwaise_distance(self, X):
        """
        Compute the pairwise distance between the data points
        When the corruption is set to occlusion
        Then the occlusion is applied to the other data point in order to compute the pairwise distance

        Parameters
        ----------
        X : np.array
            Input data matrix
        
        Returns
        -------
        np.array
            Pairwise distance matrix
        """
        O = np.zeros((X.shape[0], X.shape[0]))
        annotation_dataset = json.load(open("C:\\MVA\\1er Semestre\\G Data Analysis\\RPCA\\Robust-PCA-Augmented\\Corrupted_Datasets\\occlusion\\Cyprien\\annotation.json"), "r")
        if self.corruption_type == 'occlusion':
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    image_first_pixel, image_patch_size = annotation_dataset[str(i)]
                    image_first_pixel2, image_patch_size2 = annotation_dataset[str(j)]
                    mask = np.ones((X.shape[1], X.shape[1]))
                    mask[image_first_pixel[0]:image_first_pixel[0] + image_patch_size, image_first_pixel[1]:image_first_pixel[1] + image_patch_size] = 0
                    mask[image_first_pixel2[0]:image_first_pixel2[0] + image_patch_size2, image_first_pixel2[1]:image_first_pixel2[1] + image_patch_size2] = 0

                    O[i, j] = np.linalg.norm(mask * (X[i] - X[j]), 2) / np.sum(mask)
        
        return O


    def laplacian_computation(self, X):
        """
        Compute the laplacian matrix

        Parameters
        ----------
        X : np.array
            Input data matrix
        
        Returns
        -------
        np.array
            Laplacian matrix
        """
        # Library based technique to compute the laplacian matrix, working a priori
        # knn_weights = kneighbors_graph(X, self.n_neighbors, mode='distance', include_self=True)
        # knn_weights_square = np.square(knn_weights.toarray())        
        # self.laplacien = laplacian(knn_weights_square, normed=False)

        # Manual technique to compute the laplacian matrix
        if not self.corruption:
            distances = pairwise_distances(X, metric='euclidean')
        else:
            distances = self.pairwaise_distance(X)

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)
        A = np.zeros((X.shape[0], X.shape[0]))
        omega = np.min(distances[distances != 0])
        for i in range(X.shape[0]):
            for j in indices[i]:
                A[i, j] = np.exp(-(distances[i, j] - omega) ** 2 / 0.05)
            A = (A + A.T) / 2
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))
        self.laplacien = np.eye(X.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

        return self.laplacien

    
    def _nulcear_norm(self, X):
        return np.linalg.norm(X, 'nuc')

    def _squared_frobenius_norm(self, X):
        return np.linalg.norm(X, 'fro')**2
    
    def _prox_l1_operator(self, X, coeff):
        """
        Proximal operator for L1 norm

        Parameters
        ----------
        X : np.array
            Input matrix

        Returns
        -------
        np.array
            Proximal operator for L1 norm
        """
        
        return np.sign(X) * np.maximum(np.abs(X) - coeff, 0)
    
    def _prox_nuclear_operator(self, X, coeff):
        """
        Proximal operator for nuclear norm

        Parameters
        ----------
        X : np.array
            Input matrix

        Returns
        -------
        np.array
            Proximal operator for nuclear norm
        """

        U, S, V = np.linalg.svd(X, full_matrices=False)
        D = np.dot(U, np.dot(np.diag(self._prox_l1_operator(S, coeff)), V))
        return D

    def _compute_L_next(self) -> np.array:    
        """
        Compute the next L matrix

        Returns
        -------
        np.array
            L matrix
        """
        H1 = self.M - self.S + self.Z1 / self.r1
        H2 = self.W + self.Z2 / self.r2
        A = (self.r1 * H1 + self.r2 * H2) / (self.r1 + self.r2)
        return self._prox_nuclear_operator(A, 1 / (self.r1 + self.r2))
    
    def _compute_W_next(self) -> np.array:
        """
        Compute the next W matrix

        Returns
        -------
        np.array
            W matrix
        """
        return self.r2 * np.linalg.inv(self.gamma * self.laplacien + self.r2 * np.eye(self.W.shape[0])) @ (self.L - self.Z2 / self.r2)

    def _compute_S_next(self) -> np.array:
        """
        Compute the next S matrix

        Returns
        -------
        np.array
            S matrix
        """
        return self._prox_l1_operator(self.M - self.L + self.Z1 / self.r1, self._lambda / self.r1)
    
    def stopping_criterion(self, P_prev, P_next):
        """
        Compute the stopping criterion

        Returns
        -------
        bool
            True if the stopping criterion is reached, False otherwise
        """        
        return self._squared_frobenius_norm(P_prev - P_next) / self._squared_frobenius_norm(P_prev) > self.epsilon

    def fit(self) -> np.array:
        """
        Fit the model by sperating outliers (Sparse matrix) and low-rank matrix (L matrix)

        Returns
        -------
        np.array
            L matrix
        np.array
            S matrix
        """

        P1_prev, P2_prev, P3_prev = self.P1, self.P2, self.P3
        Z1_prev, Z2_prev = self.Z1, self.Z2
        
        while self.k < 100 or (np.square(self.P1 - P1_prev) / np.square(P1_prev) > self.epsilon and np.square(self.P2- P2_prev) / np.square(P2_prev) > self.epsilon and np.square(self.P3 - P3_prev) / np.square(P3_prev) > self.epsilon and self.stopping_criterion(self.Z1, Z1_prev) and self.stopping_criterion(self.Z2, Z2_prev)):
            P1_prev, P2_prev, P3_prev = self.P1, self.P2, self.P3
            Z1_prev, Z2_prev = self.Z1, self.Z2
            self.L = self._compute_L_next()
            self.S = self._compute_S_next()
            self.W = self._compute_W_next()
            self.Z1 += self.r1 * (self.M - self.L - self.S)
            self.Z2 += self.r2 * (self.W - self.L)
            self.P1 = self._nulcear_norm(self.L)
            self.P2 = self._lambda * np.sum(np.abs(self.S))
            # self.P3 = self.gamma * np.trace(self.L.T @ self.laplacien @ self.L)
            temp = self.laplacien @ self.L
            self.P3 = self.gamma * np.sum(self.L * temp)
            self.k += 1
            print(f"Iteration {self.k} - P1: {self.P1} - P2: {self.P2} - P3: {self.P3}, Difference between S and L: {np.linalg.norm(self.S - self.L)}")
        return self.L, self.S

    def pca_predict(self, k, true_labels):
        """
        Test the pca model for clustering data

        Returns
        -------
        float
            Clustering error
        """
        U, S, V = np.linalg.svd(self.L, full_matrices=False)
        for i in range(10):
            kmeans = KMeans(n_clusters=k, n_init=10).fit(U[:, :k])
            if i == 0:
                min_error = np.sum(kmeans.labels_ != true_labels) / len(true_labels) * 100
            else:
                error = np.sum(kmeans.labels_ != true_labels) / len(true_labels) * 100
                if error < min_error:
                    min_error = error
        return min_error


