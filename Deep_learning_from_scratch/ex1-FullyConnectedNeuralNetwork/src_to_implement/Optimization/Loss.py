import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.error_tensor = None
        self.label_tensor = None
        self.loss = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor

        # print(np.finfo(np.float32).eps)
        self.loss = np.sum(- np.log(self.prediction_tensor[self.label_tensor ==1] + np.finfo(np.float_).eps))

        return self.loss

    def backward(self, label_tensor):

        self.label_tensor = label_tensor

        # self.error_tensor = -1 * self.label_tensor * np.log(self.prediction_tensor)

        self.error_tensor = -1 * (self.label_tensor / (self.prediction_tensor + np.finfo(np.float_).eps))

        return self.error_tensor
