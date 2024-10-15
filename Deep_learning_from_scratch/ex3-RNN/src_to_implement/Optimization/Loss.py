import numpy as np

class CrossEntropyLoss(object):
    def __init__(self):
        self.pred = None

    def forward(self, pred, label):
        self.pred = pred
        loss = label * np.log(pred + np.finfo(float).eps)
        return -np.sum(loss)

    def backward(self, label):
        tmp = -(1 / self.pred) * label
        return tmp