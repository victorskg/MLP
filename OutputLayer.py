from Neuron import Neuron

class OutputLayer(object):

    total_error = 0

    def __init__(self, att_number, learn_rate, n_neurons):
        self.n_neurons = n_neurons
        self.att_number = att_number
        self.learn_rate = learn_rate
        self.neurons = [Neuron(att_number, learn_rate) for _ in range(n_neurons)]

    def run_layer(self, inputs, desired):
        total_error = 0
        for i in range(self.n_neurons):
            guess = self.neurons[i].run_neuron(inputs)
            error = desired[i] - guess
            self.total_error += error
            self.neurons[i].update(inputs, error, guess)

    def run_test(self, inputs):
        outputs = [self.neurons[i].run_neuron(inputs) for i in range(self.n_neurons)]
        return outputs