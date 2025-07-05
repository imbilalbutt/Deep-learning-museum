
class Sgd:
    def __init__(self, learning_rate: float):
        self.weight_tensor = None
        self.gradient_tensor = None
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.gradient_tensor = gradient_tensor
        self.weight_tensor = weight_tensor - self.learning_rate * self.gradient_tensor

        return self.weight_tensor
