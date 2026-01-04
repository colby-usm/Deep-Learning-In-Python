import numpy as np

class Tensor():
    def __init__(
        self,
        data: float | int | np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype=np.float32,
        requires_grad: bool = True
    ):

        assert not (data is not None and shape is not None), "Provide either data or shape, not both."
        assert not (data is None and shape is None), "Must provide either data or shape."

        self.dtype = dtype

        self.requires_grad = requires_grad
        self.grad = None 
        self.parents = []
        self.grad_fn = None

        if data is not None:
            # Always convert to np.ndarray
            self.data: np.ndarray = np.atleast_1d(np.array(data, dtype=self.dtype))
        else:
            self.data: np.ndarray = np.random.default_rng().random(shape, dtype=self.dtype)

        self.shape = self.data.shape


    def dot(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)

        x = np.dot(self.data, other.data)
        t = Tensor(x, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad)

        t.parents = [self, other]

        return t


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        x = self.data + other.data
        t = Tensor(x)
        t.parents = [self, other]
        
        return t


    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        t = Tensor(np.dot(self.data, other.data))
        t.grad_fn = lambda grad: ...

        return t
