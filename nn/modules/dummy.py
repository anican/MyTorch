from nn import Module


class _DummyLayer(Module):
    def __init__(self):
        super(_DummyLayer, self).__init__(None)

    def forward(self, data):
        return data

    def backward(self, previous_partial_gradient):
        return previous_partial_gradient
