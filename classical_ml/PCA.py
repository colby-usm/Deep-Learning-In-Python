import numpy as np
from neural import Tensor


class PCA:
    def __init__(self):
        pass

    @staticmethod
    def decompose_with_cov(X: np.ndarray | Tensor, dims: int) -> np.ndarray:
        """
        Given an input nxm Matrix X with 'n' observations and 'm' features, we can decompose it into nxd where d, 'dims' << m

        Steps used in decomposition:
            1. Assign a matrix A with X centered around 0 (by subtracting the mean values from of each feature)
            2. Compute the covariance matrix C
            3. Eigendecompose C and assign it to matrix W
            4. Order the eigenvectors in W by decreasing order of their eigen values
            5. Select top k eigenvectors, W_k
            6. Calculate the PCA matrix Z = XW
        """

        if isinstance(X, np.ndarray):
            if np.ndim(X) != 2:
                raise ValueError(
                    f"X must be a 2D np arary or Tensor, got {np.ndim(X)} dimensions"
                )

            X = Tensor(X)

        elif isinstance(X, Tensor):
            if np.ndim(X.data) != 2:
                raise ValueError(
                    f"X must be a 2D np arary or Tensor, got {np.ndim(X.data)} dimensions"
                )

        A = X - X.mean(axis=0)  # center columns (features) around 0

        # TODO: implement
        C = PCA._generate_covariance(A)
        W = PCA._eigen_decompose(C)
        W = PCA._sort_eigens(W)
        W_k = W[:, :dims]
        Z = A @ W_k

        return Z

    @staticmethod
    def _generate_covariance(X: Tensor):
        pass

    @staticmethod
    def _eigen_decompose(X: Tensor):
        pass

    @staticmethod
    def _sort_eigens(X: Tensor):
        pass


pca = PCA()

data = np.random.rand(4, 5)
print(f"shape:{data.shape}")
print(f"data:{data}")
pca.decompose_with_cov(data, 2)
