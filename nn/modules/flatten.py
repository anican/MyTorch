from nn import Module
import numpy as np


class Flatten(Module):
    """
    TODO:
    """
    def __init__(self, parent=None):
        super(Module, self).__init__(parent)
        self.start_dim = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.start_dim = data.shape
        batch_size, num_channels, kernel_height, kernel_width = self.start_dim
        end_dim = (batch_size, num_channels * kernel_height * kernel_width)
        return np.reshape(data, end_dim)

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        return np.reshape(previous_partial_gradient, self.start_dim)
