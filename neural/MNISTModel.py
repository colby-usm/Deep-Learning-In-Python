from .NeuralModule import NeuralModule
from .LinearLayer import LinearLayer
from .Tensor import Tensor


class MNISTModel(NeuralModule):
    def __init__(self, hl1: LinearLayer, hl2: LinearLayer, o: LinearLayer):
        super().__init__(modules={"hl1": hl1, "hl2": hl2, "o": o}, name="MNISTModel")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._modules["hl1"](x).relu()
        x = self._modules["hl2"](x).relu()
        logits = self._modules["o"](x)
        return logits.softmax(), logits
