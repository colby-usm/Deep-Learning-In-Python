import numpy as np


class SGD:
    """
    A Stochastic Gradient Descent Algorithm
    """

    def __init__(self, params, lr: float):
        params = list(params)
        print(f"num params: {len(params)}")
        self.params = params
        self.lr = lr

    def step(self, clip=1.0):
        for p in self.params:
            if p.grad is not None:
                np.clip(p.grad, -clip, clip, out=p.grad)
                p.data -= self.lr * p.grad

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.zero_grad()
