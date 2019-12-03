from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Implements Adam algorithm.
    """
    def __init__(self, parameters, learning_rate, weight_decay):
        # TODO: add betas, etc. to construct
        super(Optimizer, self).__init__(parameters)
        pass

    def step(self):
        pass
