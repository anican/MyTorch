from abc import ABC
from nn.modules import Module


class Criterion(Module, ABC):
    def forward(self, *args, **kwargs) -> float:
        raise NotImplementedError
