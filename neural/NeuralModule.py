from typing import Any
from abc import ABC, abstractmethod

from .Tensor import Tensor


class NeuralModule(ABC):
    """
    The base class for all neural based modules within this library.
    It collects parameters and submodules for easy gradient updates.
    """

    def __init__(
        self,
        params: dict[str, Tensor] | None = None,
        modules: dict[str, "NeuralModule"] | None = None,
        name="NeuralModule",
    ):
        self._parameters: dict[str, Tensor] = params or {}
        self._modules: dict[str, "NeuralModule"] = modules or {}
        self.name = name

        assert all(
            isinstance(k, str) and isinstance(v, Tensor)
            for k, v in self._parameters.items()
        ), "params must be a dict[str, Tensor]"
        assert all(
            isinstance(k, str) and isinstance(v, NeuralModule)
            for k, v in self._modules.items()
        ), "modules must be a dict[str, NeuralModule]"

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, NeuralModule):
            self._modules[name] = value
        super().__setattr__(name, value)

    def num_params(self):
        total = 0

        for p in self._parameters.values():
            total += p.data.size

        for m in self._modules.values():
            total += m.num_params()

        return total

    def __str__(self):

        def build(module, depth=0):
            indent = "  " * depth

            s = f"{indent}{module.name}(total_params={module.num_params()})\n"

            # parameters
            for name, param in module._parameters.items():
                s += (
                    f"{indent}  {name}: "
                    f"shape={param.data.shape}, "
                    f"params={param.data.size}\n"
                )

            # child modules
            for name, child in module._modules.items():
                s += f"{indent}  [{name}]\n"
                s += build(child, depth + 2)

            return s

        return build(self)

    def parameters(self):
        for p in self._parameters.values():
            if p.requires_grad:
                yield p
        for module in self._modules.values():
            yield from module.parameters()

    @abstractmethod
    def forward(self, x : Tensor) -> Tensor | tuple[Tensor, ...]:
        pass


    @property
    @abstractmethod
    def in_features(self) -> int:
        pass

    @property
    @abstractmethod
    def out_features(self) -> int:
        pass

    def __call__(self, x: Tensor):
        return self.forward(x)
