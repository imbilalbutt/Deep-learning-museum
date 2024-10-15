class TanH(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inpT):
        self.activation = np.tanh(inpT)
        return self.activation

    def backward(self, errT):
        derv = (1 - np.power(self.activation, 2))
        return derv*errT