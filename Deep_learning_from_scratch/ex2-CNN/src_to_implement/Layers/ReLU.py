from .Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__int__()

        self.prediction = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.prediction = np.maximum(self.input_tensor, 0)

        return self.prediction

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        # self.error_tensor[error_tensor < 0] = 0

        self.error_tensor[self.input_tensor < 0] = 0

        return self.error_tensor
