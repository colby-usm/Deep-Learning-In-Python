import numpy as np

class Tensor():
    def __init__(
        self,
        data: float | int | np.floating | np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype=np.float32,
        requires_grad: bool = True,
        parents = None
    ):

        assert not (data is not None and shape is not None), "Provide either data or shape, not both."
        assert not (data is None and shape is None), "Must provide either data or shape."

        self.dtype = dtype

        self.requires_grad = requires_grad
        self.grad = None 
        self.parents =parents
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


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        t = Tensor(self.data + other.data,
                   dtype=self.dtype,
                   requires_grad=self.requires_grad or other.requires_grad,
                   parents=[self,other]
        )


        #TODO: implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t


    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        t = Tensor(self.data - other.data,
                   dtype=self.dtype,
                   requires_grad=self.requires_grad or other.requires_grad,
                   parents=[self,other]
        )


        #TODO: implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        t = Tensor(self.data * other.data,
                   dtype=self.dtype,
                   requires_grad=self.requires_grad or other.requires_grad,
                   parents=[self,other]
        )


        #TODO: implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t

    def __rmul__(self, other):
        return self.__mul__(other)


    def log(self):
        t = Tensor(np.log(self.data),
                   dtype=self.dtype,
                   requires_grad=self.requires_grad,
                   parents=[self]
        )


        #TODO implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t


    def dot(self, other):
        return self.__matmul__(other)


    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        t = Tensor(np.dot(self.data, other.data),
                   dtype=self.dtype,
                   requires_grad=self.requires_grad or other.requires_grad,
                   parents=[self, other]
        )

        #TODO: implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t


    def sum(self):
        t = Tensor(np.sum(self.data),
                   dtype=self.dtype,
                   requires_grad=self.requires_grad,
                   parents=[self]
                   )

        #TODO: implement grad_fn
        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t


    @property
    def T(self):
        t = Tensor(self.data.T,
                   dtype=self.dtype,
                   requires_grad=self.requires_grad,
                   parents=[self]
        )

        def grad_fn(grad):
            pass

        t.grad_fn = grad_fn
        return t
