from typing import Optional, Callable
import numpy as np

from nn import Parameter
from .module import Module


class Linear(Module):
    def __init__(self, input_features: int, output_features: int, is_bias=True, parent=None):
        super(Linear, self).__init__(parent)
        self.data = None
        # TODO: make bias parameter optional
        # self.is_bias = is_bias
        self.bias = Parameter(np.zeros((1, output_features), dtype=np.float64))
        self.weight = Parameter(np.random.randn(input_features, output_features), dtype=np.float64)
        self.initialize()

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        TODO
        :return:
        """
        # Linear conv_layers (fully connected) forward pass
        # :param data: n X d array (batch x features)
        # :return: n X c array (batch x channels)
        # :math:`A^T x + b`
        output = np.matmul(data, self.weight.data) + self.bias.data
        self.data = data
        return output

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        """
        Does the backwards computation of gradients wrt weights and inputs
        :param previous_partial_gradient: n X c partial gradients wrt future conv_layers
        :return: gradients wrt inputs
        """
        n, c = previous_partial_gradient.shape
        inputs_gradient = np.matmul(previous_partial_gradient, self.weight.data.T)
        self.weight.grad = np.matmul(self.data.T, previous_partial_gradient)
        self.bias.grad = np.sum(previous_partial_gradient, axis=0)
        return inputs_gradient

    def selfstr(self):
        return str(self.weight.data.shape)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(Linear, self).initialize()
