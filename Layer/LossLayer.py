import numpy as np
from Layer.BaseLayer import Layer


class LossLayer(Layer):
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
        self.input = None
        self.target = None

    def forward(self, input, target=None):
        self.input = input
        self.target = target

        if self.target is None: raise ValueError("LossLayer.forward() requires a target value.")
            

        return self.loss(self.target, self.input)

    def backward(self, output_gradient=None, learning_rate=0.0):
        if self.target is None or self.input is None:raise ValueError("Call LossLayer.forward() before LossLayer.backward().")
            
        if output_gradient is not None:return output_gradient
            
        return self.loss_prime(self.target, self.input)
