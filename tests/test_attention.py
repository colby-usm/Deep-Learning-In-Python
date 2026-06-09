from neural.Initializers import HeNormal
from neural.Attention import Attention
from neural.LinearLayer import LinearLayer
from neural.NeuralModule import NeuralModule
from neural.Tensor import Tensor


class AttenNet(NeuralModule):
    def __init__(self, attention, hl1, hl2, o, name="AttenNet"):
        super().__init__(
            modules={"attention": attention, "hl1": hl1, "hl2": hl2, "o": o}, name=name
        )

    def forward():
        pass

    @property
    def in_features(self) -> int:
        return self._modules["attention"]._modules["q_proj"].in_features

    @property
    def out_features(self) -> int:
        return self._modules["o"].out_features


he_normal = HeNormal()


ATTENTION_DIMENSION = 128

IMAGE_H = 224
IMAGE_L = 224
PATCH_STRIDE = 14
TOKEN_H = IMAGE_H // PATCH_STRIDE
TOKEN_L = IMAGE_L // PATCH_STRIDE
TOKEN_DIM = TOKEN_H * TOKEN_L


attention = Attention(
    linear_layers=(
        LinearLayer(
            (TOKEN_DIM, ATTENTION_DIMENSION), initializer=HeNormal(), name="q_proj"
        ),
        LinearLayer(
            (TOKEN_DIM, ATTENTION_DIMENSION), initializer=HeNormal(), name="k_proj"
        ),
        LinearLayer(
            (TOKEN_DIM, ATTENTION_DIMENSION), initializer=HeNormal(), name="v_proj"
        ),
    ),
    dimension=ATTENTION_DIMENSION,
    name="Attention 1",
)

# Feed Forward Network
hl1 = LinearLayer((ATTENTION_DIMENSION, 64))
hl2 = LinearLayer((64, 64))
o = LinearLayer((64, 10))

atten_net_1 = AttenNet(attention, hl1, hl2, o, name="Attention Net 1")

print(atten_net_1)
