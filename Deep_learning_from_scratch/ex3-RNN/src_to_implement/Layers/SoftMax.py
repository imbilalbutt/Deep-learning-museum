from .Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__int__()

        self.output_tensor = None
        self.prediction = None
        self.error_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Method 1
        input_tensor_adjusted = self.input_tensor - np.max(self.input_tensor)  #self.input_tensor - np.max(input_tensor, axis=1)
        exponential = np.exp(input_tensor_adjusted)


        # Method 1.1 --> wrong --> return a scalar single number
        # exponential_whole_sum = np.sum(exponential)

        #  Method 1.2.1 --> works
        # exponential_whole_sum = np.sum(exponential, axis=1, keepdims=True)

        # Method 1.2.2
        exponential_whole_sum = np.sum(exponential, axis=1).reshape(exponential.shape[0], 1)

        self.prediction = np.divide(exponential, exponential_whole_sum)

        # Method 2 --> works
        # self.prediction = (np.exp(self.input_tensor - np.max(self.input_tensor)) / np.exp(
        #     self.input_tensor - np.max(self.input_tensor)).sum(axis=1, keepdims=True))

        return self.prediction

    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        # Method 1 --> did not work
        # self.output_tensor = self.prediction * (self.error_tensor - np.sum(self.error_tensor * self.prediction, axis=1).T)

        # Method 2
        #self.output_tensor = self.prediction * (self.error_tensor - np.sum(self.error_tensor * self.prediction, axis=1, keepdims=True))

        # Method 3 --> does not work
        # self.output_tensor = np.dot(self.prediction, (
        #             self.error_tensor - np.sum(self.error_tensor * self.prediction, axis=1, keepdims=True)))

        # Method 4
        gradient = np.sum(self.error_tensor * self.prediction, axis=1)
        self.output_tensor = self.prediction * (self.error_tensor - gradient.reshape(gradient.shape[0], 1))

        return self.output_tensor
