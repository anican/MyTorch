import numpy as np
from numba import njit, prange
from nn import Module


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class ReLU(Module):
    # TODO: requires testing!
    def __init__(self, parent=None, use_numba=False):
        super(ReLU, self).__init__(parent)
        self.use_numba = use_numba
        self.data = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        # TODO: add if block for `if self.use_numba == True`
        self.data = data
        output = np.maximum(self.data, 0, dtype=np.float64)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        pass

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        inputs_loss_gradient = np.zeros(self.data.shape, dtype=np.float64)
        inputs_loss_gradient[self.data > 0] = 1
        return inputs_loss_gradient * previous_partial_gradient

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        for idx, datum in enumerate(data):
            if datum <= 0:
                data[idx] = 0
            else:
                data[idx] = grad[idx]
        return data


class Sigmoid(Module):
    # TODO: requires testing!
    def __init__(self, parent=None):
        super(Sigmoid, self).__init__(parent)
        self.output = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.output = sigmoid(data)
        return self.output

    def backward(self, previous_partial_gradient: np.ndarray) -> np.ndarray:
        return previous_partial_gradient * self.output * (1 - self.output)


class Softplus(Module):
    # TODO: requires testing!
    def __init__(self, parent=None):
        super(Softplus, self).__init__(parent)
        self.data = None

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.data = data
        output = np.log(1 + np.exp(self.data))
        return output

    def backward(self, previous_partial_gradient):
        return previous_partial_gradient * sigmoid(self.data)








