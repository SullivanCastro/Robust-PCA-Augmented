import numpy as np
from sklearn.cluster import KMeans


class custom_PCA():

    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components_  = None
        self.explained_variance_ = None

    def _eigen_values(self, X):
        """
        Compute eigen values

        Parameters
        ----------
        X : np.array
            Input matrix

        Returns
        -------        
        np.array
            Eigen vectors
        np.array
            Eigen values
        """
        X_corr = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(X_corr)

        return S, V

    def _project_data(self, X):
        """
        Project data

        Parameters
        ----------
        X : np.array
            Input matrix

        Returns
        -------
        np.array
            Projected matrix
        """
        return np.dot(X, self.components_.T)


    def fit(self, X):
        """
        Fit PCA model

        Parameters
        ----------
        X : np.array
            Input matrix

        Returns
        -------
        np.array
            Reduced input matrix
        """
        eig_values, eig_vectors = self._eigen_values(X)
        self.explained_variance_ = eig_values / eig_values.sum()
        self.components_ = eig_vectors[:self.n_components]
        return - self._project_data(X)
    
    def predict(self, k, X, true_labels):
        """
        Predict using PCA

        Parameters
        ----------
        k : int
            Number of clusters
        X : np.array
            Input matrix
        true_labels : np.array
            True labels
        
        Returns
        -------
        float
            Clustering error
        """
        def calculate_min_error(X_proj, true_labels, k):
            errors = [
                np.mean(KMeans(n_clusters=k).fit(X_proj).labels_ != true_labels) * 100
                for _ in range(10)
            ]
            return min(errors)

        min_errors = [
            calculate_min_error(custom_PCA(k).fit(X), true_labels, k)
            for _ in range(10)
        ]

        return min(min_errors)