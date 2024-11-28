import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment as hungarian
import json
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds

class Robust_PCA_augmented():

    def __init__(self, M, corruption=False, annotation_path = None, gamma=1e-3, nb_iter=100, knn=0) -> None:
        self.M           = M
        self.k           = 0
        self.gamma       = gamma
        self.r1          = 1
        self.r2          = 1
        self.annotation_path = annotation_path
        self.epsilon     = 1e-7
        self._lambda     = 1/(gamma * np.sqrt(np.max(M.shape)))
        self.corruption  = corruption
        self.nb_iter     = nb_iter
        self.corruption_type = 'occlusion'
        self.knn        = knn

        self.laplacien = self.laplacian_computation()

        init_array       = np.random.random(M.shape)
        self.L           = init_array
        self.W           = init_array
        self.S           = init_array
        self.Z1          = self.M - self.L - self.S
        self.Z2          = self.W - self.L
        self.P1          = self._nulcear_norm(self.L)
        self.P2          = self._lambda * np.sum(np.abs(self.S))
        temp = self.laplacien @ self.L
        self.P3 = self.gamma * np.sum(self.L * temp)

    def custom_pairwise_distance(self):
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
        num_samples = self.M.shape[0]
        O = np.zeros((num_samples, num_samples))
        with open(self.annotation_path, "r") as file:
            annotation_dataset = json.load(file) 

        image_shape = (92, 112) 

        if self.corruption_type == 'occlusion':
            for i in range(self.M.shape[0]):
                for j in range(self.M.shape[0]):
                    if i == j:
                        O[i, j] = 0
                    else:
                        (x1, y1), patch_size1 = annotation_dataset[str(i)]
                        (x2, y2), patch_size2 = annotation_dataset[str(j)]

                        image_i = self.M[i].reshape(image_shape)
                        image_j = self.M[j].reshape(image_shape)

                        mask = np.ones(image_shape)

                        mask[x1:x1+patch_size1, y1:y1+patch_size1] = 0
                        mask[x2:x2+patch_size2, y2:y2+patch_size2] = 0

                        masked_image_i = (mask * image_i).flatten()
                        masked_image_j = (mask * image_j).flatten()

                        O[i, j] = np.linalg.norm(masked_image_i - masked_image_j, 2) / np.sum(mask)

        return O

    def laplacian_computation(self):
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

        if not self.corruption:
            distances = pairwise_distances(self.M, metric='euclidean')
        else:
            distances = self.custom_pairwise_distance()

        A = np.zeros((self.M.shape[0], self.M.shape[0]))
        omega = np.min(distances[distances != 0])
        if self.knn == 0:
            for i in range(self.M.shape[0]):
                for j in range(self.M.shape[0]):
                    scale_factor = 1e-10
                    A[i, j] = np.exp(-(distances[i, j] - omega) ** 2 / 0.05 * scale_factor)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='ball_tree').fit(self.M)
            _, indices = nbrs.kneighbors(self.M)
            for i in range(self.M.shape[0]):
                for j in indices[i]:
                    scale_factor = 1e-10
                    A[i, j] = np.exp(-(distances[i, j] - omega) ** 2 / 0.05 * scale_factor)

        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D) + 1e-9)
        self.laplacien = np.eye(self.M.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))

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

        U, S, V = svds(X)
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
        
        U, Sigma, VT = svds(A)  # Truncated SVD
        r = (self.r1 + self.r2) / 2
        Sigma_thresholded = np.maximum(Sigma - 1 / r, 0)
        return U @ np.diag(Sigma_thresholded) @ VT
    
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
        # print(self._lambda / self.r1)
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
        
        while self.k < self.nb_iter or (np.square(self.P1 - P1_prev) / np.square(P1_prev) > self.epsilon and np.square(self.P2- P2_prev) / np.square(P2_prev) > self.epsilon and np.square(self.P3 - P3_prev) / np.square(P3_prev) > self.epsilon and self.stopping_criterion(self.Z1, Z1_prev) and self.stopping_criterion(self.Z2, Z2_prev)):
            P1_prev, P2_prev, P3_prev = self.P1, self.P2, self.P3
            Z1_prev, Z2_prev = self.Z1, self.Z2
            self.L = self._compute_L_next()
            self.S = self._compute_S_next()
            # break
            self.W = self._compute_W_next()
            self.Z1 += self.r1 * (self.M - self.L - self.S)
            self.Z2 += self.r2 * (self.W - self.L)
            self.P1 = self._nulcear_norm(self.L)
            self.P2 = self._lambda * np.linalg.norm(self.S, ord=1)
            # self.P3 = self.gamma * np.trace(self.L.T @ self.laplacien @ self.L)
            temp = self.laplacien @ self.L
            self.P3 = self.gamma * np.sum(self.L * temp)
            self.k += 1
            print(f"Iteration {self.k} - P1: {self.P1} - P2: {self.P2} - P3: {self.P3}, Difference between S and L: {np.linalg.norm(self.S - self.L)}")
            print(f"Iteration {self.k} - Difference between M and S+L: {np.linalg.norm(self.M - self.S - self.L)}")
        
        return self.L, self.S
    
    def compute_clustering_error(self, labels, y_true):
        cost_matrix = np.zeros((np.max(labels) + 1, np.max(y_true) + 1))
        for i in range(len(labels)):
            cost_matrix[labels[i], y_true[i]] += 1

        row_ind, col_ind = hungarian(-cost_matrix)
        mapping = dict(zip(row_ind, col_ind))

        aligned_labels = np.array([mapping[label] for label in labels])
        return 100 * (1 - accuracy_score(y_true, aligned_labels))


    def pca_predict(self, true_labels):
        """
        Test the pca model for clustering data

        Returns
        -------
        float
            Clustering error
        """
        U, S, V = svds(self.L)
        cumulated_variance = np.cumsum(S**2) / np.sum(S**2)
        n_components = np.argmax(cumulated_variance > 0.95) + 1

        errors = []
        for _ in range(10):
            kmeans = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10, random_state=None)
            kmeans.fit(U[:, :n_components])
            errors.append(self.compute_clustering_error(kmeans.labels_, true_labels))
        return np.min(errors)