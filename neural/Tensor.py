import numpy as np

class Tensor():
    def __init__(
        self,
        data: float | int | np.floating | np.ndarray | None = None,
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
            self.data = np.atleast_1d(np.array(data, dtype=self.dtype))
        else:
            self.data = np.array(
                np.random.default_rng().random(shape),
                dtype=self.dtype
)

        self.shape = self.data.shape



    def __repr__(self):
            return f"Tensor(data={self.data}, shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

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


    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype, requires_grad=False)

        out_data = self.data * other.data
        t = Tensor(out_data, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad)

        t.parents = [self, other]

        if self.requires_grad:
            def grad_fn(grad):
                pass

        t.grad_fn = grad_fn
        return t

    # right-hand multiplication (scalar * Tensor)
    __rmul__ = __mul__


    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        t = Tensor(np.dot(self.data, other.data))
 
        if self.requires_grad:
            def grad_fn(grad):
                pass

        t.grad_fn = grad_fn
        return t


    def log(self):
        x = self.data
        t = Tensor(np.log(x), dtype=self.dtype, requires_grad=self.requires_grad)

        if self.requires_grad:
            def grad_fn(grad):
                pass

        t.grad_fn = grad_fn
        return t

    def sum(self):
        x = self.data
        t = Tensor(np.sum(x), dtype=self.dtype, requires_grad=self.requires_grad)
 
        if self.requires_grad:
            def grad_fn(grad):
                pass
        return t


    @property
    def T(self):
        t = Tensor(self.data.T, dtype=self.dtype, requires_grad=self.requires_grad)
        t.parents = [self]

        if self.requires_grad:
            def grad_fn(grad):
                pass

        t.grad_fn = grad_fn
        return t

    def astype(self, dtype):
        self.data = self.data.astype(dtype)
        self.dtype = dtype
        return self
