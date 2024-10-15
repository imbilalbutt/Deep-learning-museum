import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.weight_tensor = None
        self.gradient_tensor = None
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.gradient_tensor = gradient_tensor
        self.weight_tensor = weight_tensor - self.learning_rate * self.gradient_tensor

        return self.weight_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum: float):
        self.weight_tensor = None
        self.gradient_tensor = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.velocity = self.momentum * self.velocity - self.learning_rate * self.gradient_tensor

        self.weight_tensor = weight_tensor + self.velocity

        return self.weight_tensor


class Adam:
    def __init__(self, learning_rate: float, mu: float, rho: float):
        self.weight_tensor = None
        self.gradient_tensor = None
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.velocity = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.velocity = self.mu * self.velocity + (1 - self.mu) * self.gradient_tensor

        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(self.gradient_tensor,self.gradient_tensor)

        # Bias Correction
        velocity_corrected = self.velocity / (1 - self.mu ** self.k)
        r_corrected = self.r / (1 - self.rho ** self.k)

        # Adam update
        self.weight_tensor = weight_tensor - self.learning_rate * (velocity_corrected / np.sqrt(r_corrected) + np.finfo(np.float_).eps)

        return self.weight_tensor
