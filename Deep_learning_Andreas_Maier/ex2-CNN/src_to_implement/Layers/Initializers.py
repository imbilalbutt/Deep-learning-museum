import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.value = value
        self.fan_out = None
        self.fan_in = None
        self.weights_shape = None
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.full(weights_shape, self.value)
        return self.weights


class UniformRandom:

    def __init__(self):
        self.fan_out = None
        self.fan_in = None
        self.weights_shape = None
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weights = np.random.uniform(0, 1, weights_shape)
        return self.weights


class Xavier:
    def __init__(self):
        self.fan_out = None
        self.fan_in = None
        self.weights_shape = None
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        sigma = np.sqrt(2 / (fan_in + fan_out))
        self.weights = np.random.normal(loc=0.0, scale=sigma, size=weights_shape)
        return self.weights


class He:
    def __init__(self):
        self.fan_out = None
        self.fan_in = None
        self.weights_shape = None
        self.weights = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        sigma = np.sqrt(2 / fan_in)
        self.weights = np.random.normal(loc=0.0, scale=sigma, size=weights_shape)
        return self.weights
