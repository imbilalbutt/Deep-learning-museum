from .Base import BaseLayer


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.input_tensor = None
        super().__int__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = None
        self.bias = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

    def backward(self, error_tensor):
        pass

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
