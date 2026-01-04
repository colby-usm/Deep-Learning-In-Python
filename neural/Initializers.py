import numpy as np
from typing import Tuple

class Initializer:
    """Base class for weight initializers."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        raise NotImplementedError("Initializer subclasses must implement __call__")


class UniformInitializer(Initializer):
    """Uniform random initializer U(low, high)."""
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high

    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        return np.random.default_rng().uniform(self.low, self.high, size=shape).astype(dtype)


class XavierUniform(Initializer):
    """Glorot/Xavier uniform initializer (good for tanh/sigmoid/linear)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in, n_out = shape[1], shape[0]
        elif len(shape) == 1:
            n_in, n_out = shape[0], 1
        else:
            raise ValueError(f"Unsupported shape for XavierUniform: {shape}")
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.default_rng().uniform(-limit, limit, size=shape).astype(dtype)


class XavierNormal(Initializer):
    """Glorot/Xavier normal initializer (good for tanh/sigmoid/linear)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in, n_out = shape[1], shape[0]
        elif len(shape) == 1:
            n_in, n_out = shape[0], 1
        else:
            raise ValueError(f"Unsupported shape for XavierNormal: {shape}")
        std = np.sqrt(2 / (n_in + n_out))
        return np.random.default_rng().normal(0, std, size=shape).astype(dtype)


class HeNormal(Initializer):
    """He / Kaiming normal initializer (good for ReLU)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in = shape[1]
        elif len(shape) == 1:
            n_in = shape[0]
        else:
            raise ValueError(f"Unsupported shape for HeNormal: {shape}")
        std = np.sqrt(2 / n_in)
        return np.random.default_rng().normal(0, std, size=shape).astype(dtype)


class HeUniform(Initializer):
    """He / Kaiming uniform initializer (good for ReLU)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in = shape[1]
        elif len(shape) == 1:
            n_in = shape[0]
        else:
            raise ValueError(f"Unsupported shape for HeUniform: {shape}")
        limit = np.sqrt(6 / n_in)
        return np.random.default_rng().uniform(-limit, limit, size=shape).astype(dtype)


class LeCunNormal(Initializer):
    """LeCun normal initializer (good for SELU)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in = shape[1]
        elif len(shape) == 1:
            n_in = shape[0]
        else:
            raise ValueError(f"Unsupported shape for LeCunNormal: {shape}")
        std = np.sqrt(1 / n_in)
        return np.random.default_rng().normal(0, std, size=shape).astype(dtype)


class LeCunUniform(Initializer):
    """LeCun uniform initializer (good for SELU)."""
    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        if len(shape) == 2:
            n_in = shape[1]
        elif len(shape) == 1:
            n_in = shape[0]
        else:
            raise ValueError(f"Unsupported shape for LeCunUniform: {shape}")
        limit = np.sqrt(3 / n_in)
        return np.random.default_rng().uniform(-limit, limit, size=shape).astype(dtype)
