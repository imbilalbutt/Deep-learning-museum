import math
import numpy as np

from .Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, inpT: np.ndarray) -> np.ndarray:
        batch_sz, ch, h_in, w_in = inpT.shape
        h_p, w_p = self.pooling_shape

        h_out = math.floor((h_in - h_p) / self.stride_shape[0]) + 1
        w_out = math.floor((w_in - w_p) / self.stride_shape[1]) + 1

        windows = np.lib.stride_tricks.as_strided(
                                            inpT,
                                            shape=(batch_sz, ch, h_out, w_out, *self.pooling_shape),
                                            strides=(inpT.strides[0],
                                                inpT.strides[1],
                                                self.stride_shape[0] * inpT.strides[2],
                                                self.stride_shape[1] * inpT.strides[3],
                                                inpT.strides[2],
                                                inpT.strides[3])
                )
        out = np.max(windows, axis=(4,5))

        maxs_scaled = out.repeat(self.stride_shape[0], axis=2).repeat(self.stride_shape[1], axis=3)
        x_window = inpT[:, :, :h_out * self.stride_shape[0], :w_out * self.stride_shape[1]]
        self.mask_argmax = np.equal(x_window, maxs_scaled).astype(int)
        self.inpT = inpT
        return out

    def backward(self, errT: np.ndarray) -> np.ndarray:
        dA = errT.repeat(self.stride_shape[0], axis=2).repeat(self.stride_shape[1], axis=3)
        dA = np.multiply(dA, self.mask_argmax)
        pad = np.zeros(self.inpT.shape)
        pad[:, :, :dA.shape[2], :dA.shape[3]] = dA
        return pad