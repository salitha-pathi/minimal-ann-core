import cupy as cp
from .layer import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = cp.random.randn(output_size, input_size)
        self.bias = cp.random.randn(output_size, 1)

    def forward(self, input):
        self.input = cp.array(input)
        return cp.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = cp.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return cp.dot(self.weights.T, output_gradient)
