import sys
import os
import numpy as np

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_custom_pca.pca import custom_PCA
from model_custom_pca.rpca_augmented import Robust_PCA_augmented

class Robust_PCA():

    def __init__(self, M, delta=1e-7) -> None:
        self.S           = 0
        self.Y           = 0
        self.M           = M
        self.n1, self.n2 = M.shape
        self.mu          = (self.n1*self.n2) / (4*np.linalg.norm(M, 1))
        self._lambda     = 1/np.sqrt(np.max(M.shape))
        self.D           = self._singular_value_thresholding(M, 1/self.mu)
        self.delta       = delta
        self.L           = np.zeros_like(M)

    
    def _shrink_operator(self, X, tau):
        """
        Shrinkage operator

        Parameters
        ----------
        X : np.array
            Input matrix
        tau : float

        Returns
        -------
        np.array
            Shrinked matrix
        """
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    

    def _singular_value_thresholding(self, X: np.array, tau: float) -> np.array:
        """
        Singular value thresholding operator

        Parameters
        ----------
        X : np.array
            Input matrix
        tau : float

        Returns
        -------
        np.array
            Singular value thresholded matrix
        """
        U, S, V = np.linalg.svd(X, full_matrices=False)
        self.D = np.dot(U, np.dot(np.diag(self._shrink_operator(S, tau)), V))
        return self.D


    def _compute_L_next(self) -> np.array:
        """
        Compute the next L matrix

        Returns
        -------
        np.array
            L matrix
        """
        return self._singular_value_thresholding(self.M - self.S + self.Y/self.mu, self.mu)
    

    def _compute_S_next(self) -> np.array:
        """
        Compute the next S matrix

        Returns
        -------
        np.array
            S matrix
        """
        return self._shrink_operator(self.M - self.L + self.Y/self.mu, self._lambda*self.mu) 
    

    def _compute_Y_next(self) -> np.array:
        """
        Compute the next Y matrix
        
        Returns
        -------
        np.array
            Y matrix
        """
        return self.Y + self.mu * (self.M - self.L - self.S)
    
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
        self.S ,self.Y = np.zeros_like(self.M), np.zeros_like(self.M)
        while self.delta < np.linalg.norm(self.M - self.L - self.S) / np.linalg.norm(self.M):
            self.L = self._compute_L_next()
            self.S = self._compute_S_next()
            self.Y = self._compute_Y_next()
        return self.L, self.S
    

if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets.squeeze()

    # Create outliers
    idx_outliers = np.random.randint(0, len(y), len(y)//100) 
    y[idx_outliers] = np.where(y[idx_outliers] == "M", "B", "M")

    # standardize data
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    
    import matplotlib.pyplot as plt

    plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("PCA n=2")
    pca = custom_PCA(n_components=2)
    X_pca = pca.fit(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.where(y=="M", "red", "blue"), alpha=0.5)
    plt.xlabel("Principal Component Analysis 1")
    plt.ylabel("Principal Component Analysis 2")

    rpca = Robust_PCA(X)
    L, S = rpca.fit()

    # Plot Low Rank Matrix PCA
    plt.subplot(1, 3, 2)
    plt.title("RPCA Low-Rank matrix n=2")
    pca = custom_PCA(n_components=2)
    L_pca = pca.fit(L)
    plt.scatter(L_pca[:, 0], L_pca[:, 1], c=np.where(y=="M", "red", "blue"), alpha=0.5, label="Low Rank Matrix")
    plt.xlabel("Principal Component Analysis 1")
    plt.ylabel("Principal Component Analysis 2")

    # Plot Sparse Matrix PCA
    # plt.subplot(1, 3, 3)
    # S_pca = pca.fit(S)
    # plt.title("RPCA Sparse matrix n=2")
    # plt.scatter(S_pca[:, 0], S_pca[:, 1], c=np.where(y=="M", "red", "blue"), alpha=0.3, label="Sparse Matrix")
    # plt.xlabel("Principal Component Analysis 1")
    # plt.ylabel("Principal Component Analysis 2")

    rpca_augmented = Robust_PCA_augmented(X)
    L, S = rpca_augmented.fit()

    plt.subplot(1, 3, 3)
    plt.title("RPCA Augmented Low-Rank matrix n=2")
    pca = custom_PCA(n_components=2)
    
    L_pca = pca.fit(L)
    plt.scatter(L_pca[:, 0], L_pca[:, 1], c=np.where(y=="M", "red", "blue"), alpha=0.5, label="Low Rank Matrix")
    plt.xlabel("Principal Component Analysis 1")
    plt.ylabel("Principal Component Analysis 2")


    plt.tight_layout()
    plt.show()