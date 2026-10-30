import numpy as np
from Layer.BaseLayer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation             #activation function
        self.activation_prime = activation_prime #derivative of activation function

    def forward(self, input):  #input=Z
        self.input = input
        return self.activation(self.input) #output=Y

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))