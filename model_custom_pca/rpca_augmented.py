import numpy as np
from sklearn.decomposition import PCA

class Robust_PCA_augmented():

    def __init__(self, X, gamma=2.8) -> None:
        self.X           = X
        self.k           = 0
        self.gamma       = gamma
        self.r1          = 1
        self.r2          = 1
        self.epsilon     = 1e-7
        self._lambda     = 1/np.sqrt(np.max(X.shape))

        self.laplacien = self.laplacian(X)

        self.L           = np.random.rand(*X.shape)
        self.W           = np.random.rand(*X.shape)
        self.S           = np.random.rand(*X.shape)
        self.Z1          = X - self.L - self.S
        self.Z2          = self.W - self.L
    
    def laplacian(self, X):
        # useful because the laplaician matrix will be different
        # for corrupted data, to be computed manually for these datasets
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
        vectorized_samples = X.reshape(X.shape[0], -1)

        # Adjacency matrix
        A = np.exp(-np.linalg.norm(vectorized_samples[:, None] - vectorized_samples, axis=-1) ** 2 / 0.05)

        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        self.laplacian = np.eye(A.shape[0]) - D ** -0.5 @ A @ D ** -0.5
        return self.laplacian
    
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
        return np.dot(U, np.dot(np.diag(self._prox_l1_operator(S, coeff)), V))

    def _compute_L_next(self) -> np.array:    
        """
        Compute the next L matrix

        Returns
        -------
        np.array
            L matrix
        """
        H1 = self.X - self.S + self.Z1 / self.r1
        H2 = self.W + self.Z2 / self.r2
        return self._prox_nuclear_operator((self.r1 * H1 + self.r2 * H2) / (self.r1 + self.r2), 1 / (self.r1 + self.r2))

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
        return self._prox_l1_operator(self.X - self.L + self.Z1 / self.r1, self._lambda / self.r1)
    
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
        P1 = self._nulcear_norm(self.L)
        P2 = self._lambda * np.sum(np.abs(self.S))
        P3 = self.gamma * np.trace(self.L.T @ self.laplacien @ self.L)

        P1_prev, P2_prev, P3_prev = P1, P2, P3
        Z1_prev, Z2_prev = self.Z1, self.Z2
        
        while self.k == 0 or (np.square(P1 - P1_prev) / np.square(P1_prev) > self.epsilon and  np.square(P2- P2_prev) / np.square(P2_prev) > self.epsilon and np.square(P3 - P3_prev) / np.square(P3_prev) > self.epsilon and self.stopping_criterion(self.Z1, Z1_prev) and self.stopping_criterion(self.Z2, Z2_prev)):
            P1_prev, P2_prev, P3_prev = P1, P2, P3
            Z1_prev, Z2_prev = self.Z1, self.Z2
            self.L = self._compute_L_next()
            self.S = self._compute_S_next()
            self.W = self._compute_W_next()
            self.Z1 += self.r1 * (self.X - self.L - self.S)
            self.Z2 += self.r2 * (self.W - self.L)
            P1 = self._nulcear_norm(self.L)
            P2 = self._lambda * np.sum(np.abs(self.S))
            P3 = self.gamma * np.trace(self.L.T @ self.laplacien @ self.L)
            self.k += 1
            print(f"Iteration {self.k} - P1: {P1.shape} - P2: {P2.shape} - P3: {P3.shape}, Z1: {self.Z1.shape}, Z2: {self.Z2.shape}")

        return self.L, self.S


