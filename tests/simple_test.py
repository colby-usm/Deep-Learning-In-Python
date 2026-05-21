from neural.NeuralModule import NeuralModule
from neural.LinearLayer import LinearLayer
from neural.Tensor import Tensor


hl1 = LinearLayer(
    (
        Tensor.randn(784, 64),
        Tensor.randn(
            64,
        ),
    ),
    name="hl1",
)
hl2 = LinearLayer(
    (
        Tensor.randn(64, 64),
        Tensor.randn(
            64,
        ),
    ),
    name="hl2",
)
o = LinearLayer(
    (
        Tensor.randn(64, 10),
        Tensor.randn(
            10,
        ),
    ),
    name="o",
)


ffn = NeuralModule(
    modules={
        hl1.name: hl1,
        hl2.name: hl2,
        o.name: o,
    }
)

print(ffn)
