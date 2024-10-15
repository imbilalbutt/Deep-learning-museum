import numpy as np
from .Constraints import L2_Regularizer, L1_Regularizer


class Optimizer(object):
    def __init__(self) -> None:
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if isinstance(self.regularizer, (L1_Regularizer, L2_Regularizer)):
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(
                weight_tensor) - self.learning_rate * gradient_tensor
        else:
            weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        if isinstance(self.regularizer, (L1_Regularizer, L2_Regularizer)):
            updated_weight = weight_tensor + self.v - self.learning_rate * self.regularizer.calculate_gradient(
                weight_tensor)
        else:
            updated_weight = weight_tensor + self.v
        return updated_weight


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        g = gradient_tensor
        mu = self.mu
        rho = self.rho
        self.k += 1
        self.v = mu * self.v + (1 - mu) * g
        self.r = rho * self.r + (1 - rho) * g * g

        v_hat = self.v / (1 - np.power(mu, self.k))
        r_hat = self.r / (1 - np.power(rho, self.k))
        if isinstance(self.regularizer, (L1_Regularizer, L2_Regularizer)):
            updated_weight = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(
                weight_tensor) - self.learning_rate * (v_hat) / (np.sqrt(r_hat) + np.finfo(float).eps)
        else:
            updated_weight = weight_tensor - self.learning_rate * (v_hat) / (np.sqrt(r_hat) + np.finfo(float).eps)
        return updated_weight