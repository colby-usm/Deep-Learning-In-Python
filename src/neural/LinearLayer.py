from .Tensor import Tensor
from .NeuralModule import NeuralModule
from .Initializers import HeNormal
import numpy as np


class LinearLayer(NeuralModule):
    def __init__(
        self,
        data: tuple[Tensor, Tensor] | tuple[int, int],
        dtype=np.float32,
        initializer=None,
        name="LinearLayer",
    ):

        super().__init__(name=name)

        assert len(data) == 2, f"data must be a tuple of length 2, received {len(data)}"
        assert all(isinstance(d, Tensor) for d in data) or all(
            isinstance(d, int) for d in data
        ), (
            f"data must be either a tuple of Tensors or a tuple of ints, received: {type(data[0])} and {type(data[1])}"
        )

        if isinstance(data[0], int) and isinstance(data[1], int):
            in_features: int = data[0]
            out_features: int = data[1]
            init = initializer or HeNormal()
            self.weights = Tensor(init((in_features, out_features)), dtype=dtype)
            self.biases = Tensor(np.zeros(out_features, dtype=dtype))
        elif isinstance(data[0], Tensor) and isinstance(data[1], Tensor):
            self.weights, self.biases = data[0], data[1]
        else:
            raise ValueError(
                f"data must be either both ints or both Tensors, received {type(data[0])}, and {type(data[1])}"
            )

    @property
    def in_features(self) -> int:
        return self.weights.data.shape[0]

    @property
    def out_features(self) -> int:
        return self.weights.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.in_features, self.out_features)

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.biases
