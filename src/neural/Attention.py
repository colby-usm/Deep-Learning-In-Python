import numpy as np

from .Tensor import Tensor
from .LinearLayer import LinearLayer
from .NeuralModule import NeuralModule


class Attention(NeuralModule):
    def __init__(
        self,
        linear_layers: tuple[LinearLayer, LinearLayer, LinearLayer],
        dimension,
        name="Attention",
    ):

        q_proj, k_proj, v_proj = linear_layers

        assert q_proj.shape == k_proj.shape and k_proj.shape == v_proj.shape, (
            "Linear layers in Attention module must all have the same shape. "
            f"Got: q_proj={q_proj.shape}, "
            f"k_proj={k_proj.shape}, "
            f"v_proj={v_proj.shape}"
        )

        super().__init__(
            modules={"q_proj": q_proj, "k_proj": k_proj, "v_proj": v_proj}, name=name
        )
        self.d = dimension

    def forward(self, x: Tensor, mask=None):
        """
        x has shape (B, T, D)
        B: number of batches
        T: each sequence of tokens (all padded to the same length)
        D: dimension of each sequence
        """

        q = self._modules["q_proj"](x)
        k = self._modules["k_proj"](x)
        v = self._modules["v_proj"](x)

        # (B, T, D) @ (B, D, T) -> (B, T, T)
        scores = (q @ k.T) / np.sqrt(self.d)

        if mask is not None:
            scores += mask

        A = scores.softmax(axis=-1)

        # (B, T, T) @ (B, T, D) → (B, T, D)
        return A @ v

    @property
    def in_features(self) -> int:
        return self._modules["q_proj"].in_features

    @property
    def out_features(self) -> int:
        return self._modules["q_proj"].out_features
