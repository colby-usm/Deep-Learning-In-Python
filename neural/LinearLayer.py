from .Tensor import Tensor
from .NeuralModule import NeuralModule
import numpy as np


class LinearLayer(NeuralModule):
    def __init__(
        self, data: tuple[Tensor, Tensor], dtype=np.float32, name="LinearLayer"
    ):
        super().__init__(name=name)
        self.weights, self.biases = data
        self.shape = self.weights.data.shape

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return x @ self.weights + self.biases
