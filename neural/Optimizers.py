class SGD:
    """
    A Stochastic Gradient Descent Algorithm
    """

    def __init__(self, params, lr: float):
        params = list(params)
        print(f"num params: {len(params)}")
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                before = p.data[0, 0]
                p.data -= self.lr * p.grad
                after = p.data[0, 0]
                print(
                    f"before: {before:.6f}, after: {after:.6f}, grad: {p.grad[0, 0]:.6f}"
                )
                break

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.zero_grad()
