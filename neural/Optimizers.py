class SGD:
    """
    A Stochastic Gradient Descent Algorithm
    """

    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.zero_grad()
