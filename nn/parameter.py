import numpy as np


class Parameter(object):
    """
    :class Parameter is used to create wrappers for the data and gradients of parameters
    in a network. Such parameters include but are not limited to the weights and biases.

    If for some reason you ever got around to building mytorch on top of a tensor related library instead of numpy, then
    you'd want Parameter to inherit from the tensor building block instead of just object. 
    """
    def __init__(self, data: np.ndarray, dtype=np.float32):
        self._data = data.astype(dtype)
        self._grad = np.zeros_like(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data[:] = data

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad[:] += grad

    def zero_grad(self):
        self._grad[:] = 0

