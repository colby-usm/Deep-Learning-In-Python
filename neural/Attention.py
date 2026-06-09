import numpy as np

from.Tensor import Tensor
from .LinearLayer import LinearLayer
from .NeuralModule import NeuralModule

class Attention(NeuralModule):

    def __init__(
            self,
            linear_layers: tuple[LinearLayer, LinearLayer, LinearLayer],
            dimension,
            name="Attention"
    ):

        super().__init__(name=name)
        self.d = dimension
        self.q_proj, self.k_proj, self.v_proj = linear_layers


        assert (
            self.q_proj.shape == self.k_proj.shape
            and self.k_proj.shape == self.v_proj.shape
        ), (
            "Linear layers in Attention module must all have the same shape. "
            f"Got: q_proj={self.q_proj.shape}, "
            f"k_proj={self.k_proj.shape}, "
            f"v_proj={self.v_proj.shape}"
        )

    def forward(self, x: Tensor):
        '''
        x has shape (B, T, D)
        B: number of independent batches
        T: each independent sequence of tokens (all padded to the same length)
        D: dimension of each sequence
        '''

        """
        x: (B, T, D)
        """

        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        # (B, T, D) @ (B, D, T) → (B, T, T)
        scores = (q @ k.T) / np.sqrt(self.d)

        # optional mask (future use)
        # scores = scores + mask

        # softmax over last axis (keys)
        A = scores.softmax(axis=-1)

        # (B, T, T) @ (B, T, D) → (B, T, D)
        return A @ v


    @property
    def in_features(self) -> int:
        return self.q_proj.in_features

    @property
    def out_features(self) -> int:
        return self.q_proj.out_features
