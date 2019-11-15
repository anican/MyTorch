class Optimizer(object):
    """
    Base class for all optimizers.
    """
    def __init__(self, parameters: Iterable):
        self.parameters = parameters

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        pass

    def zero_grad(self):
        """
        Sets gradients of all model parameters to zero.
        """
        for parameter in self.parameters:
            parameter.zero_grad()
