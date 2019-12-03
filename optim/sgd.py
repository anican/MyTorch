from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum).
    """
    def __init__(self, parameters, learning_rate, momentum=0, dampening=0, weight_decay=0,
            nesterov=False):
        super(Optimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.prev_deltas = [0] * len(parameters)

    def step(self):
        for idx, parameter in enumerate(self.parameters):
            curr_delta = self.momentum * self.prev_deltas[idx] + (parameter.grad +
                    self.weight_decay * parameter.data)
            self.prev_deltas[idx] = curr_delta
            parameter.data = parameter.data - self.learning_rate * curr_delta
