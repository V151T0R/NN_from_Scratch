import numpy as np
from Layer.LossLayer import LossLayer


class MSE(LossLayer):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(mse, mse_prime)


class BinaryCrossEntropy(LossLayer):
    def __init__(self):
        def binary_cross_entropy(y_true, y_pred):
            return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

        def binary_cross_entropy_prime(y_true, y_pred):
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

        super().__init__(binary_cross_entropy, binary_cross_entropy_prime)