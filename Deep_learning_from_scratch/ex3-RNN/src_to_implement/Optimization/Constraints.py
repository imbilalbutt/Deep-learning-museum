import numpy as np


class L2_Regularizer(object):
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha*weights

    def norm(self, weights):
        return self.alpha*np.sum(np.power(np.abs(weights), 2))

class L1_Regularizer(object):
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights):
        return self.alpha*np.sign(weights)

    def norm(self, weights):
        return self.alpha*np.sum(np.abs(weights))