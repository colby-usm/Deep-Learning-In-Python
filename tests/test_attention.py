from neural.Initializers import HeNormal
from neural.Attention import Attention
from neural.LinearLayer import LinearLayer

he_normal = HeNormal()

attention = Attention(
        linear_layers=(
            LinearLayer((784, 64), initializer=HeNormal(), name="q_proj"),
            LinearLayer((784, 64), initializer=HeNormal(), name="k_proj"),
            LinearLayer((784, 64), initializer=HeNormal(), name="v_proj")
        ),
        dimension=64,
        name="Attention 1"
)

print(attention)
