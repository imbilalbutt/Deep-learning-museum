from .Base import BaseLayer
import numpy as np
import copy


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size + 1
        self.output_size = output_size
        self.gradient_weights = None
        self.gradient_bias = None
        self.trainable = True

        # dims of rows = output of input_layer --> going toward hidden_layer
        # dims of cols = input to hidden_layer <-- coming from input_layer
        self.weights = np.random.uniform(0, 1, (self.input_size, self.output_size))
        # self.bias = np.random.uniform(0, 1, (1, output_size))
        self._optimizer = None
        self.gradient_wrt_input_X = None
        self.gradient_wrt_weight_W = None
        self.error_tensor = None
        self.output_tensor = None
        self.input_tensor = None
        self.gradient_wrt_bias_B = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Method 1: half test case of FCNN passes
        # np.concatenate((self.input_tensor, np.ones((self.input_tensor.shape[0], 1))), axis=1).reshape(
        #     self.input_tensor.shape[0], self.input_tensor.shape[1])

        # Method 2: almost all test cases of FCNN passes
        self.input_tensor = np.concatenate((self.input_tensor, np.ones((self.input_tensor.shape[0], 1))), axis=1)

        self.output_tensor = np.dot(self.input_tensor, self.weights)  # + self.bias
        return self.output_tensor

    # dy = error tensor (gradient of loss w.r.t.y)
    def backward(self, error_tensor):
        self.error_tensor = error_tensor

        # self.gradient_bias = error_tensor
        # self.gradient_wrt_input_X = np.dot(self.error_tensor, self.weights)
        temp_weight = copy.deepcopy(self.weights)
        self.gradient_wrt_input_X = np.dot(self.error_tensor, temp_weight[:-1].T)  # np.dot(W.T, dy)
        self.gradient_weights = np.dot(self.input_tensor.T, self.error_tensor) # np.dot(dy, x.T)
        # self.gradient_wrt_weight_W = np.dot(self.error_tensor.T, self.input_tensor)  # np.dot(dy, x.T)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            # self.weights -= self._optimizer (learning_rate * self.gradient_wrt_weight_W)

        return self.gradient_wrt_input_X

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
