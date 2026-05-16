from abc import ABC, abstractmethod
import numpy as np
from neural.Tensor import Tensor

class Activation(ABC):
    '''
    An abstract Activation class for activation functions

    All child classes must implement the forward method

    it uses the neural.Tensor class
    '''
 
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        '''
        abstract forward method for an activation class

        all child classes must implement the forward 
        '''
        pass
 

    def __call__(self, x: Tensor):
        return self.forward(x)


class Relu(Activation):

    def __init__(self):
        pass


    @TODO fix this to perform on the tensor operations
    def forward(self, x: Tensor):
        '''
        Relu is defined as max(0, x) for any scalar value
        it returns a Tensor object
        '''
        return Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    @TODO implement me
    def __backward__():
        pass


class Softmax(Activation):

    def __init__(self):
        '''
        Softmax is defined as 
        '''
        pass


    def __call__(self, x: Tensor):
        return self.forward(x)


    @TODO fix this to perform on the tensor operations
    def forward(self, x: Tensor):
        '''
        softmax is invariant to scalar shifts, we shift to ensure numeric stability — no overflow
        '''
        shift_x = x.data - np.max(x.data)
        exp_x = np.exp(shift_x)
        return Tensor(exp_x / exp_x.sum(), requires_grad=x.requires_grad)


    @TODO implement me
    def __backward__():
        pass
