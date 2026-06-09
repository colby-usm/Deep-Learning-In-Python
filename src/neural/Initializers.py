import numpy as np
from typing import Tuple


class Initializer:
    """
    Abstract base class for parameter initialization strategies.

    Initializers define how tensors such as weights or biases are populated
    before training begins. Proper initialization is critical for stable
    gradient flow and convergence during optimization.

    Subclasses implement __call__ and return a NumPy array with the requested
    shape and dtype.
    """

    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Generate an initialized tensor.

        Args:
            shape:
                Shape of the tensor to initialize.

            dtype:
                NumPy dtype of the returned tensor.

        Returns:
            np.ndarray:
                Initialized tensor values.
        """
        raise NotImplementedError("Initializer subclasses must implement __call__")


class UniformInitializer(Initializer):
    """
    Initialize parameters from a uniform distribution U(low, high).

    This initializer is primarily useful for experimentation or debugging.
    In practice, Xavier, He, or LeCun initializers are usually preferred
    because they preserve activation and gradient variance more effectively.

    Unlike scale-aware initializers, UniformInitializer does not account
    for fan_in or fan_out, meaning activation variance will grow or shrink
    with layer size.

    """

    def __init__(self, low: float = -1.0, high: float = 1.0):
        """
        Args:
            low:
                Lower bound of the uniform distribution.

            high:
                Upper bound of the uniform distribution.
        """
        self.low = low
        self.high = high

    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Sample tensor values uniformly from [low, high].

        Args:
            shape:
                Shape of the tensor to initialize.

            dtype:
                NumPy dtype of the returned tensor.

        Returns:
            np.ndarray:
                Tensor initialized with uniformly distributed values.
        """
        return (
            np.random.default_rng()
            .uniform(self.low, self.high, size=shape)
            .astype(dtype)
        )


class XavierUniform(Initializer):
    """
    Xavier / Glorot uniform initialization.

    Samples weights from:

        U(-sqrt(6 / (fan_in + fan_out)),
           sqrt(6 / (fan_in + fan_out)))

    Designed to preserve activation variance across layers and
    help prevent vanishing/exploding gradients by keeping activation
    variance approximately constant across layers

    Commonly used with:
        - tanh
        - sigmoid
        - linear activations
    """

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
    """
    Xavier / Glorot normal initialization.

    Samples weights from:

        N(0, sqrt(2 / (fan_in + fan_out)))

    Preserves signal variance across layers and is effective for
    symmetric activation functions.

    Commonly used with:
        - tanh
        - sigmoid
        - linear activations
    """

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
    """
    He / Kaiming normal initialization.

    Samples weights from:

        N(0, sqrt(2 / fan_in))

    Specifically designed for ReLU-family activations, where many
    activations become zero during forward propagation.


    The factor of 2 in sqrt(2 / fan_in) compensates for ReLU zeroing
    approximately half of all activations during the forward pass,
    which would otherwise cause variance to halve at each layer.

    Helps maintain gradient magnitude in deep networks.

    Commonly used with:
        - ReLU
        - LeakyReLU
        - GELU
    """

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
    """
    He / Kaiming uniform initialization.

    Samples weights from:

        U(-sqrt(6 / fan_in),
           sqrt(6 / fan_in))

    Uniform counterpart to HeNormal.

    Effective for deep networks using ReLU-family activations.
    """

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
    """
    LeCun normal initialization.

    Samples weights from:

        N(0, sqrt(1 / fan_in))

    Designed to preserve activation variance for self-normalizing
    networks.

    Commonly used with:
        - SELU
    """

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
    """
    LeCun uniform initialization.

    Samples weights from:

        U(-sqrt(3 / fan_in),
           sqrt(3 / fan_in))

    Uniform counterpart to LeCunNormal.

    Intended for self-normalizing neural networks using SELU
    activations.
    """

    def __call__(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:

        if len(shape) == 2:
            n_in = shape[1]

        elif len(shape) == 1:
            n_in = shape[0]

        else:
            raise ValueError(f"Unsupported shape for LeCunUniform: {shape}")

        limit = np.sqrt(3 / n_in)

        return np.random.default_rng().uniform(-limit, limit, size=shape).astype(dtype)
