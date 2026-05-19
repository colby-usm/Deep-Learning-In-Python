import numpy as np


class Tensor:
    """
    A multidimensional Tensor class:

    supports standard arithmeic, linear algebra, and nonlinear operations


    backward() builds a topological ordering via DFS, then performs a reverse
    topological traversal to propagate gradients through the computation graph.
    """

    _rng = np.random.default_rng()

    @staticmethod
    def randn(*shape, dtype=np.float32, requires_grad=True):
        return Tensor(
            Tensor._rng.random(shape), dtype=dtype, requires_grad=requires_grad
        )

    def __init__(
        self,
        data: float | int | np.floating | np.ndarray,
        dtype=np.float32,
        requires_grad: bool = True,
        parents=None,
        zero_epsilon=1e-6,
    ):
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.grad_fn = None
        self.zero_epsilon = zero_epsilon
        self.data = np.atleast_1d(np.array(data, dtype=self.dtype))
        self.grad = np.zeros_like(self.data, dtype=self.dtype)

    def __repr__(self):

        return f"Tensor(data={self.data}, shape={self.data.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def backward(self):
        """
        Two-pass backward:

        1. Build a topological ordering of the computation graph using DFS.
        2. Traverse nodes in reverse topological order, calling each node's grad_fn
           to accumulate gradients.
        """

        visited = set()
        graph = []

        def build(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v.parents:
                    build(p)
                graph.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(graph):
            if node.grad_fn:
                node.grad_fn(node.grad)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        t = Tensor(
            self.data + other.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A/∂x (A+B) = g
            ∂B/∂x (A+B) = g
            """
            if self.requires_grad:
                self.grad += g
            if other.requires_grad:
                other.grad += g

        t.grad_fn = grad_fn
        return t

    def __sub__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        t = Tensor(
            self.data - other.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A/∂x (A-B) = g
            ∂B/∂x (A-B) = -g
            """
            if self.requires_grad:
                self.grad += g
            if other.requires_grad:
                other.grad -= g

        t.grad_fn = grad_fn
        return t

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )
        t = Tensor(
            self.data * other.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A*∂x (A*B) = g * B
            ∂B*∂x (A*B) = g * A
            """
            if self.requires_grad:
                self.grad += g * other.data
            if other.requires_grad:
                other.grad += g * self.data

        t.grad_fn = grad_fn
        return t

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        d = np.where(
            np.abs(other.data) < self.zero_epsilon, self.zero_epsilon, other.data
        )

        t = Tensor(
            self.data / d,
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            Given Tensors A,B and upstream gradient g:
            ∂L/∂A = g / B
            ∂L/∂B = -g * A / B^2
            """

            d = np.where(
                np.abs(other.data) < self.zero_epsilon, self.zero_epsilon, other.data
            )

            if self.requires_grad:
                self.grad += g / d
            if other.requires_grad:
                other.grad += -g * self.data / (d**2)

        t.grad_fn = grad_fn
        return t

    def __rtruediv__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        d = np.where(
            np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
        )
        t = Tensor(
            other.data / d,
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[other, self],
        )

        def grad_fn(g):
            """
            Given Tensors A,B and upstream gradient g:
            ∂A/∂x (A/B)
                       = ∂A/∂x AB^{-1}
                       = B^{-1}

            ∂B/∂x (A/B)
                       = ∂B/∂x AB^{-1}
                       = -AB^{-2}
            """
            d = np.where(
                np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
            )
            if self.requires_grad:
                self.grad += -g * other.data / (d**2)
            if other.requires_grad:
                other.grad += g / d

        t.grad_fn = grad_fn
        return t

    def __neg__(self):
        t = Tensor(
            -self.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            ∂A/∂x (-A) = -1
            """
            if self.requires_grad:
                self.grad -= g

        t.grad_fn = grad_fn
        return t

    def __pow__(self, exp):
        t = Tensor(
            self.data**exp,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given Tensor A and an upstream gradient g:
            ∂A/∂x A^{exp} = g * exp * A^{exp}
            """
            if self.requires_grad:
                self.grad += g * (exp * (self.data ** (exp - 1)))

        t.grad_fn = grad_fn
        return t

    def max(self, other):
        """
        Calculates the rowwise max between two tensors of the same shape
        """
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        if self.data.shape != other.data.shape:
            raise ValueError(f"shape mismatch {self.data.shape} vs {other.data.shape}")

        mask = self.data > other.data  # True where self wins

        t = Tensor(
            np.maximum(self.data, other.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            Given Tensor a and b, and an upstream gradient g:
            ∂A/∂x max(A,B) is calculated for 'self' by zero-masking the entries in its data where the self.data < other.data
            ∂B/∂x max(A,B) is calculated for 'other' with the NOT of ∂A/∂x max(A,B)
            """

            if self.requires_grad:
                self.grad += g * mask

            if other.requires_grad:
                other.grad += g * (~mask)

        t.grad_fn = grad_fn
        return t

    def min(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        if self.data.shape != other.data.shape:
            raise ValueError(f"shape mismatch {self.data.shape} vs {other.data.shape}")

        mask = self.data < other.data  # True where self wins

        t = Tensor(
            np.minimum(self.data, other.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            Given Tensor A and B, and an upstream gradient g:
            ∂A/∂x min(A,B) is calculated for 'self' by zero-masking the entries in its data where the self.data > other.data
            ∂B/∂x min(A,B) is calculated for 'other' with the NOT of ∂A/∂x min(A,B)
            """

            if self.requires_grad:
                self.grad += g * mask

            if other.requires_grad:
                other.grad += g * (~mask)

        t.grad_fn = grad_fn
        return t

    def mean(self):
        t = Tensor(
            np.mean(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g:
            ∂A/∂x mean(x) = g / size(A)
            """

            if self.requires_grad:
                self.grad += (g / self.data.size) * np.ones_like(self.data)

        t.grad_fn = grad_fn
        return t

    def exp(self):
        t = Tensor(
            np.exp(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor Aand upstream gradient g:
            dL/dx = g * exp(A)
            """

            if self.requires_grad:
                self.grad += g * t.data

        t.grad_fn = grad_fn
        return t

    def log(self):
        t = Tensor(
            np.log(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor Aand upstream gradient g:
            ∂A/∂x log(A) = 1/A
            """
            d = np.where(
                np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
            )
            if self.requires_grad:
                self.grad += g / d

        t.grad_fn = grad_fn
        return t

    @property
    def T(self):
        """
        Standard Transpose function
        """
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """∂A/∂x A.T = g.T"""
            if self.requires_grad:
                self.grad += g.T

        out.grad_fn = grad_fn
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        t = Tensor(
            np.dot(self.data, other.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def grad_fn(g):
            """
            Given two Tensors A,B and upstream gradient g:

            ∂L/∂A = g @ B^T
            ∂L/∂B = A^T @ g
            """

            if self.requires_grad:
                self.grad += g @ other.T.data

            if other.requires_grad:
                other.grad += self.T.data @ g

        t.grad_fn = grad_fn
        return t

    def dot(self, other):
        return self.__matmul__(other)

    def sum(self):
        out = Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g

            Since y = sum(A):
            ∂L/∂A = g broadcast to all elements
            """

            if self.requires_grad:
                self.grad += g * np.ones_like(self.data)

        out.grad_fn = grad_fn
        return out

    def tanh(self):

        d = (np.exp(self.data) - np.exp(-self.data)) / (
            np.exp(self.data) + np.exp(-self.data)
        )

        out = Tensor(
            d,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g

            ∂L/∂A = g(1 - tanh(A)^2)
            """

            if self.requires_grad:
                self.grad += g * (1 - out.data**2)

        out.grad_fn = grad_fn
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0), requires_grad=self.requires_grad, parents=[self]
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g

            ∂L/∂A = g * (1 if A > 0 , 0 if A <= 0)
            """
            if self.requires_grad:
                self.grad += g * (self.data > 0)

        out.grad_fn = grad_fn
        return out

    def softmax(self):

        shifted_data = np.exp(self.data - np.max(self.data))

        out = Tensor(
            shifted_data / np.sum(shifted_data),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g
            ∂L/∂A = A * (g - sum(g * A))
            """
            if self.requires_grad:
                self.grad += out.data * (g - np.sum(g * out.data))

        out.grad_fn = grad_fn
        return out

    def sigmoid(self):

        # numerically stable sigmoid calculation
        x = self.data
        pos = 1 / (1 + np.exp(-x))
        neg = np.exp(x) / (1 + np.exp(x))
        d = np.where(x >= 0, pos, neg)

        out = Tensor(
            d,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given a Tensor A, and an upstream gradient g
            ∂L/∂A = g * (sigmoid(A) * (1 - sigmoid(A)))
            """
            if self.requires_grad:
                self.grad += g * (out.data * (1 - out.data))

        out.grad_fn = grad_fn
        return out

    def binary_cross_entropy(self, target):
        target = (
            target
            if isinstance(target, Tensor)
            else Tensor(target, requires_grad=False)
        )

        d = np.where(
            np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
        )
        out = Tensor(
            -np.mean(target.data * np.log(d) + (1 - target.data) * np.log(1 - d)),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given predictions P, targets Y, and upstream gradient g:
            ∂L/∂P = g * (-(Y/P) + (1-Y)/(1-P)) / N
            """
            if self.requires_grad:
                self.grad += (
                    g
                    * (-(target.data / d) + (1 - target.data) / (1 - d))
                    / self.data.size
                )

        out.grad_fn = grad_fn
        return out

    def categorical_cross_entropy(self, target):
        target = (
            target
            if isinstance(target, Tensor)
            else Tensor(target, requires_grad=False)
        )
        d = np.where(
            np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
        )

        out = Tensor(
            -np.mean(np.sum(target.data * np.log(d), axis=-1)),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def grad_fn(g):
            """
            Given predictions P, one-hot targets Y, and upstream gradient g:
            ∂L/∂P = g * (-Y/P) / N
            where N is the batch size (number of samples)
            """
            if self.requires_grad:
                self.grad += g * (-target.data / d) / self.data.shape[0]

        out.grad_fn = grad_fn
        return out
