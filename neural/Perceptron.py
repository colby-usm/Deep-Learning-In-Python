import numpy as np
from typing import Tuple, Union

class Perceptron():

    def __init__(self,
                    size: np.uint16 = np.uint16(32),
                    data: Tuple[np.ndarray, Union[int,float]] | None = None,
                    dtype=np.float32
        ):

        self.dtype=dtype
        self.size = size

        if data is None:
            self.weights = np.random.default_rng().random(size, dtype=self.dtype)
            self.bias = np.array(0, dtype=dtype).item()

        else:
            self.weights, self.bias = data
            assert size == self.weights.shape[0], f"weights.shape does not match size, got size: {size}, and weights.shape: {self.weights.shape}"
            assert np.isscalar(self.bias), f"bias must be scalar, got {type(self.bias)}"

            self.bias = np.array(self.bias, dtype=self.weights.dtype).item()
 

 
    def __call__(self, x: np.ndarray):
        return self.forward(x)


    def forward(self, x: np.ndarray):
        assert x.shape == (self.size,), f"Input shape {x.shape} does not match weights size {self.size}"
        return np.dot(self.weights, x) + self.bias
