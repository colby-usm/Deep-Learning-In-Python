from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
 
    @abstractmethod
    def forward(self, x):
        pass
 
    def __call__(self, x) -> np.ndarray:
        return self.forward(x)


class Relu(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)


class Softmax(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / exp_x.sum()
