from typing import Optional, Callable
import numpy as np
from nn import Parameter, Module


class Linear(Module):
    def __init__(self, input_features: int, output_features: int, is_bias=True, parent=None):
        """

        :param input_features:
        :param output_features:
        :param is_bias:
        :param parent:
        """
        super(Linear, self).__init__(parent)
        self.data = None
        self.is_bias = is_bias
        if is_bias:
            self.bias = Parameter(np.zeros((1, output_features), dtype=np.float64))
        self.weight = Parameter(np.random.randn(input_features, output_features), dtype=np.float64)
        self.initialize()

    def forward(self, data: np.ndarray) -> np.ndarray:
        """

        :param data:
        :return:
        """
        self.data = data
        output = np.matmul(self.data, self.weight.data)
        output += self.bias.data if self.is_bias else 0
        return output

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        """

        :param previous_partial_gradient:
        :return:
        """
        inputs_gradient = np.matmul(previous_partial_gradient, self.weight.data.T)
        self.weight.grad = np.matmul(self.data.T, previous_partial_gradient)
        if self.is_bias:
            self.bias.grad = np.sum(previous_partial_gradient, axis=0)
        return inputs_gradient

    def selfstr(self):
        return str(self.weight.data.shape)

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            if self.is_bias:
                self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(Linear, self).initialize()
