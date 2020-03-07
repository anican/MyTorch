from abc import ABC

from nn.modules import Criterion
from nn.modules import MetaModule


class Network(MetaModule, ABC):
    def __init__(self, criterion: Criterion):
        super(Network, self).__init__()
        self.criterion: Criterion = criterion

    @property
    def final_layer(self):
        return self.criterion

    def loss(self, *args, **kwargs) -> float:
        raise NotImplementedError
