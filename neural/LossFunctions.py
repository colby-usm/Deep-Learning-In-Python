from neural.Tensor import Tensor

class SimpleCrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        return -1 * (y_true * y_pred.log()).sum()

