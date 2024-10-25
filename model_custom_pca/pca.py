import numpy as np

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