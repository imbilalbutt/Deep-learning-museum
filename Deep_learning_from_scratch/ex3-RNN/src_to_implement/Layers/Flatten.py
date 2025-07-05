from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__int__()
        self.input_tensor = None
        self.error_tensor = None

    #     [batch, channel, width, height

    def forward(self, input_tensor):

        # [:, channel, width, height]
        batch_size, channels, height, width = input_tensor.shape

        # Method 1
        self.input_tensor = input_tensor.reshape(batch_size, -1)

        # Method 2
        # self.input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

        return self.input_tensor

    def backward(self, error_tensor):

        # [:, channel, width, height]

        self.error_tensor = error_tensor.reshape(self.input_tensor.shape)

        return self.error_tensor
