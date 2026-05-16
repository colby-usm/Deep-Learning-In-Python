from neural.Tensor import Tensor
import numpy as np

class LinearLayer():
    def __init__(
        self,
        data: tuple[Tensor, Tensor],
        dtype=np.float32
    ):

        self.weights, self.biases = data
        self.weights = self.weights.astype(dtype)
        self.biases = self.biases.astype(dtype)
        self.shape = self.weights.data.shape



    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return x @ self.weights.T + self.biases

