import numpy as np
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, inpT: np.ndarray):
        if self.testing_phase:
            return inpT
        self.comparator = np.random.rand(*inpT.shape) < self.prob
        updates_nodes = np.multiply(inpT, self.comparator)
        updates_nodes = updates_nodes/self.prob
        return updates_nodes

    def backward(self, errT: np.ndarray):
        updates_nodes = np.multiply(errT, self.comparator)
        updates_nodes = updates_nodes/self.prob
        return updates_nodes