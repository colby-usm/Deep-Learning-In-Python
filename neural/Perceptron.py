from neural.Tensor import Tensor

class Perceptron():

    def __init__(self, data: tuple[Tensor, Tensor]):
        self.weights, self.bias = data
        self.size = self.weights.data.shape[0]

        assert self.weights.data.ndim == 1, f"Weights must be 1D, got {self.weights.shape}"
        assert self.bias.shape == (1,), f"bias tensor is of wrong shape.  Expected: (1,), Received: {self.bias.shape}"

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        assert x.shape == self.weights.data.shape, f"Input shape {x.shape} does not match weights size {self.weights.data.size}"

        bias: Tensor = self.bias
        return x.dot(self.weights) + bias
