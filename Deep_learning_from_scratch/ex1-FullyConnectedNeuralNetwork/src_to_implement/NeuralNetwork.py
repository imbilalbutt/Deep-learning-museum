import copy


class NeuralNetwork:

    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.error_tensor = None
        self.loss_tensor = None
        self.input_tensor = None
        self.label_tensor = None
        self.prediction = None
        self.input_data = None

    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()

        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        if self.input_tensor is not None:
            self.loss_tensor = self.loss_layer.forward(self.input_tensor, self.label_tensor)

        return self.loss_tensor

    def backward(self):

        self.error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def append_layer(self, layer):

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for itr in range(0, iterations):
            # print("iteration = ", itr, "\n")
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.input_data = input_tensor
        for layer in self.layers:
            self.prediction = layer.forward(self.input_data)
            self.input_data = self.prediction
        return self.prediction
