from neural.Initializers import Initializer, UniformInitializer
import numpy as np

class NeuralLayer():
    def __init__(
        self,
        size: tuple[int, int],
        data: tuple[np.ndarray, np.ndarray] | None = None,
        initializer: Initializer | None = None,
        dtype=np.float32
    ):
        assert isinstance(size, tuple) and len(size) == 2, (
            f"size must be a tuple of length 2, got {size}"
        )

        out_features, in_features = size
        assert out_features > 0 and in_features > 0, (
            f"size values must be positive, got {size}"
        )

        self.size = size
        self.out_features_size = out_features
        self.in_features_size = in_features
        self.dtype = dtype


        if data is None:
            if initializer is None:
                initializer = UniformInitializer()

            self.weights = initializer(size, self.dtype)
            self.biases = initializer((out_features,), self.dtype)
        else:
            self.weights, self.biases = data

            assert isinstance(self.weights, np.ndarray), "weights must be a numpy array"
            assert isinstance(self.biases, np.ndarray), "biases must be a numpy array"

            assert self.weights.shape == size, (
                f"Unmatched size {size} with weight matrix {self.weights.shape}"
            )

            assert self.biases.shape == (out_features,), (
                f"Bias shape must be ({out_features},), got {self.biases.shape}"
            )

            assert self.weights.dtype == dtype, (
                f"weights dtype {self.weights.dtype} does not match {dtype}"
            )

            assert self.biases.dtype == dtype, (
                f"biases dtype {self.biases.dtype} does not match {dtype}"
            )


    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x):
        return x @ self.weights.T + self.biases

