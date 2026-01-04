from abc import ABC, abstractmethod
import numpy as np
from neural.Tensor import Tensor

class Activation(ABC):
 
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
 
    def __call__(self, x: Tensor):
        return self.forward(x)


class Relu(Activation):
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)


class Softmax(Activation):
    def __init__(self):
        pass


    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        shift_x = x.data - np.max(x.data)
        exp_x = np.exp(shift_x)
        return Tensor(exp_x / exp_x.sum(), requires_grad=x.requires_grad)
