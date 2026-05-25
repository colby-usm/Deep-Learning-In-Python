import numpy as np


class Tensor:
    """
    A multidimensional Tensor class:

    supports standard arithmeic, linear algebra, and nonlinear operations


    backward() builds a topological ordering via DFS, then performs a reverse
    topological traversal to propagate gradients through the computation graph.
    """

    @staticmethod
    def _reduce_grad(g: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Reduces an upstream gradient g back to the target shape, handling two cases:
            1. g has more dimensions than target (leading dims are summed away)
            2. g has size-1 dimensions in target that were broadcast (those dims are summed and kept)

        Args:
            g: upstream gradient, may have been expanded via broadcasting
            target_shape: the shape of the original tensor to reduce back to

        Returns:
            gradient reduced to target_shape
        """

        if g.shape == target_shape:
            return g

        # sum over leading dimensions if g has more dims
        ndim_diff = g.ndim - len(target_shape)
        axes = list(range(ndim_diff))

        # also sum over dimensions that were size 1 in the target (broadcast dims)
        for i, size in enumerate(target_shape):
            if size == 1:
                axes.append(i + ndim_diff)

        if axes:
            g = g.sum(axis=tuple(axes), keepdims=True)

        return g.reshape(target_shape)

    def __init__(
        self,
        data: float | int | np.floating | np.ndarray,
        dtype=np.float32,
        requires_grad: bool = True,
        parents=None,
        zero_epsilon=1e-6,
        name=None,
    ):
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.zero_epsilon = zero_epsilon
        self.data = np.atleast_1d(np.array(data, dtype=self.dtype))
        self.grad = np.zeros_like(self.data, dtype=self.dtype)
        self.name = name
        self.grad_fn = None

    def __repr__(self):

        return f"Tensor(data={self.data.shape}, shape={self.data.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __str__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def backward(self):
        """
        Two-pass backward:

        1. Build a topological ordering of the computation graph using DFS.
        2. Traverse nodes in reverse topological order, calling each node's grad_fn
           to accumulate gradients.
        returns graph: the built graph of Tensor objects
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

        return graph

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

        def AddBackward(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A/∂x (A+B) = g
            ∂B/∂x (A+B) = g
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._reduce_grad(g, other.data.shape)

        t.grad_fn = AddBackward
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

        def SubBackward(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A/∂x (A-B) = g
            ∂B/∂x (A-B) = -g
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g, self.data.shape)
            if other.requires_grad:
                other.grad -= Tensor._reduce_grad(g, other.data.shape)

        t.grad_fn = SubBackward
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

        def MulBackward(g):
            """
            given Tensors A,B and an upstream gradient g
            ∂A*∂x (A*B) = g * B
            ∂B*∂x (A*B) = g * A
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._reduce_grad(g * self.data, other.data.shape)

            t.grad_fn = MulBackward
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

        def TrueDivBackward(g):
            """
            Given Tensors A,B and upstream gradient g:
            ∂L/∂A = g / B
            ∂L/∂B = -g * A / B^2
            """

            d = np.where(
                np.abs(other.data) < self.zero_epsilon, self.zero_epsilon, other.data
            )
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g / d, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._reduce_grad(
                    -g * self.data / (d**2), other.data.shape
                )

        t.grad_fn = TrueDivBackward
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

        def RTrueDivBackward(g):
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
                self.grad += Tensor._reduce_grad(
                    -g * other.data / (d**2), self.data.shape
                )
            if other.requires_grad:
                other.grad += Tensor._reduce_grad(g / d, other.data.shape)

        t.grad_fn = RTrueDivBackward
        return t

    def __neg__(self):
        t = Tensor(
            -self.data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def NegBackward(g):
            """
            ∂A/∂x (-A) = -1
            """
            if self.requires_grad:
                self.grad -= Tensor._reduce_grad(g, self.data.shape)

        t.grad_fn = NegBackward
        return t

    def __pow__(self, exp):
        t = Tensor(
            self.data**exp,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def PowerBackward(g):
            """
            Given Tensor A and an upstream gradient g:
            ∂A/∂x A^{exp} = g * exp * A^{exp}
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * (exp * (self.data ** (exp - 1))), self.data.shape
                )

        t.grad_fn = PowerBackward
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

        def MaxBackward(g):
            """
            Given Tensor a and b, and an upstream gradient g:
            ∂A/∂x max(A,B) is calculated for 'self' by zero-masking the entries in its data where the self.data < other.data
            ∂B/∂x max(A,B) is calculated for 'other' with the NOT of ∂A/∂x max(A,B)
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * mask, self.data.shape)

            if other.requires_grad:
                other.grad += Tensor._reduce_grad(g * (~mask), other.data.shape)

        t.grad_fn = MaxBackward
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

        def MinBackward(g):
            """
            Given Tensor A and B, and an upstream gradient g:
            ∂A/∂x min(A,B) is calculated for 'self' by zero-masking the entries in its data where the self.data > other.data
            ∂B/∂x min(A,B) is calculated for 'other' with the NOT of ∂A/∂x min(A,B)
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * mask, self.data.shape)

            if other.requires_grad:
                other.grad += Tensor._reduce_grad(g * (~mask), other.data.shape)

        t.grad_fn = MinBackward
        return t

    def mean(self, axis=None):
        t = Tensor(
            np.mean(self.data, axis=axis),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def MeanBackward(g):
            if self.requires_grad:
                if axis is None:
                    # global mean, gradient is uniform
                    grad = (g / self.data.size) * np.ones_like(self.data)
                else:
                    # number of elements that were averaged along the axis
                    n = self.data.shape[axis]
                    # restore the reduced dimension so broadcasting works
                    grad = np.expand_dims(g, axis=axis) * np.ones_like(self.data) / n
                self.grad += Tensor._reduce_grad(grad, self.data.shape)

        t.grad_fn = MeanBackward
        return t

    def exp(self):
        t = Tensor(
            np.exp(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def ExpBackward(g):
            """
            Given a Tensor Aand upstream gradient g:
            dL/dx = g * exp(A)
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * t.data, self.data.shape)

        t.grad_fn = ExpBackward
        return t

    def log(self):
        t = Tensor(
            np.log(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def LogBackward(g):
            """
            Given a Tensor Aand upstream gradient g:
            ∂A/∂x log(A) = 1/A
            """
            d = np.where(
                np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
            )
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g / d, self.data.shape)

        t.grad_fn = LogBackward
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

        def TransposeBackward(g):
            """∂A/∂x A.T = g.T"""
            if self.requires_grad:
                self.grad += g.T

        out.grad_fn = TransposeBackward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        t = Tensor(
            np.dot(self.data, other.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=[self, other],
        )

        def MatMulBackward(g):
            """
            Given two Tensors A,B and upstream gradient g:

            ∂L/∂A = g @ B^T
            ∂L/∂B = A^T @ g
            """
            if self.requires_grad:
                if g.ndim == 1 and self.data.ndim == 1:
                    self.grad += other.data @ g

                else:
                    self.grad += g @ other.T.data

            if other.requires_grad:
                if g.ndim == 1 and self.data.ndim == 1:
                    other.grad += np.outer(self.data, g)
                else:
                    other.grad += self.T.data @ g

        t.grad_fn = MatMulBackward
        return t

    def dot(self, other):
        return self.__matmul__(other)

    def sum(self):
        out = Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def SumBackward(g):
            """
            Given a Tensor A, and an upstream gradient g

            Since y = sum(A):
            ∂L/∂A = g broadcast to all elements
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * np.ones_like(self.data), self.data.shape
                )

        out.grad_fn = SumBackward
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

        def TanHBackward(g):
            """
            Given a Tensor A, and an upstream gradient g

            ∂L/∂A = g(1 - tanh(A)^2)
            """

            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * (1 - out.data**2), self.data.shape)

        out.grad_fn = TanHBackward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0), requires_grad=self.requires_grad, parents=[self]
        )

        def ReluBackward(g):
            """
            Given a Tensor A, and an upstream gradient g

            ∂L/∂A = g * (1 if A > 0 , 0 if A <= 0)
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(g * (self.data > 0), self.data.shape)

        out.grad_fn = ReluBackward
        return out

    def softmax(self):

        shifted_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))

        out = Tensor(
            shifted_data / np.sum(shifted_data, axis=-1, keepdims=True),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def SoftmaxBackward(g):
            """
            Given a Tensor A, and an upstream gradient g
            ∂L/∂A = A * (g - sum(g * A))
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    out.data * (g - np.sum(g * out.data, axis=-1, keepdims=True)),
                    self.data.shape,
                )

        out.grad_fn = SoftmaxBackward
        return out

    def softmax_cross_entropy(self, target):
        target = (
            target.data
            if isinstance(target, Tensor)
            else np.array(target, dtype=np.float32)
        )

        # forward: softmax then CCE
        shifted = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        probs = shifted / np.sum(shifted, axis=-1, keepdims=True)
        N = self.data.shape[0] if self.data.ndim > 1 else 1
        loss = -np.mean(np.sum(target * np.log(probs + 1e-8), axis=-1))

        out = Tensor(loss, requires_grad=self.requires_grad, parents=[self])

        def SoftmaxCCEBackward(g):
            """
            Combined softmax + CCE backward:
            ∂L/∂x = (softmax(x) - y) / N
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * (probs - target) / N, self.data.shape
                )

        out.grad_fn = SoftmaxCCEBackward
        return out

    def mse(self, target):
        target = (
            target
            if isinstance(target, Tensor)
            else Tensor(target, requires_grad=False)
        )

        d = np.mean((self.data - target.data) ** 2)

        out = Tensor(
            d,
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def MSEBackward(g):
            """
            Given predictions P, targets Y, and upstream gradient g:
            ∂L/∂P = g * 2(P - Y) / N
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * (2 * (self.data - target.data) / self.data.size),
                    self.data.shape,
                )

        out.grad_fn = MSEBackward
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

        def SigmoidBackward(g):
            """
            Given a Tensor A, and an upstream gradient g
            ∂L/∂A = g * (sigmoid(A) * (1 - sigmoid(A)))
            """
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * (out.data * (1 - out.data)), self.data.shape
                )

        out.grad_fn = SigmoidBackward
        return out

    def binary_cross_entropy(self, target):
        target = (
            target.data
            if isinstance(target, Tensor)
            else np.array(target, dtype=np.float32)
        )
        d = np.where(
            np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
        )
        out = Tensor(
            -np.mean(target * np.log(d) + (1 - target) * np.log(1 - d)),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def BCEBackward(g):
            """
            Given predictions P, targets Y, and upstream gradient g:
            ∂L/∂P = g * (-(Y/P) + (1-Y)/(1-P)) / N
            """
            N = self.data.shape[0] if self.data.ndim > 1 else 1
            if self.requires_grad:
                self.grad += Tensor._reduce_grad(
                    g * (-(target / d) + (1 - target) / (1 - d)) / N,
                    self.data.shape,
                )

        out.grad_fn = BCEBackward
        return out

    def categorical_cross_entropy(self, target):
        target = (
            target.data
            if isinstance(target, Tensor)
            else np.array(target, dtype=np.float32)
        )
        d = np.where(
            np.abs(self.data) < self.zero_epsilon, self.zero_epsilon, self.data
        )
        out = Tensor(
            -np.mean(np.sum(target * np.log(d), axis=-1)),
            requires_grad=self.requires_grad,
            parents=[self],
        )

        def CCEBackward(g):
            """
            Given predictions P, one-hot targets Y, and upstream gradient g:
            ∂L/∂P = g * (-Y/P) / N
            where N is the batch size (number of samples)
            """
            if self.requires_grad:
                N = self.data.shape[0] if self.data.ndim > 1 else 1
                self.grad += Tensor._reduce_grad(g * (-target / d) / N, self.data.shape)

        out.grad_fn = CCEBackward
        return out
