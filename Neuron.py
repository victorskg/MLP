import math
import numpy as np

class Neuron(object):
    
    error = 1
    derivative_v = 1

    def __init__(self, att_number, learn_rate):
        self.learn_rate = learn_rate
        self.att_number = att_number
        self.weights = np.random.rand(att_number + 1)

    def output(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def update(self, inputs, error, guess):
        self.error = error
        self.derivative_v = self.derivative(guess)
        update = self.learn_rate * error * self.derivative_v
        self.weights[1:] += update * np.array(inputs)
        self.weights[0] += update

    def run_neuron(self, inputs):
        return self.logistic_sigmoid(self.output(inputs))

    @staticmethod
    def derivative(g):
        return g * (1 - g)

    @staticmethod
    def logistic_sigmoid(u):
        return 1.0 / (1.0 + math.exp(-u))

    def __repr__(self):
        return np.array2string(self.weights, separator=',', suppress_small=True)
