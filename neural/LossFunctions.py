import numpy as np

class SimpleCrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):

        y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon)
        return -np.sum(y_true * np.log(y_pred))
