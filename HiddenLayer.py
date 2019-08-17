from Neuron import Neuron

class HiddenLayer(object):
    outputs = []

    def __init__(self, att_number, learn_rate, n_neurons):
        self.n_neurons = n_neurons
        self.att_number = att_number
        self.learn_rate = learn_rate
        self.neurons = [Neuron(att_number, learn_rate) for _ in range(n_neurons)]

    def run_layer(self, inputs):
        self.outputs = [self.neurons[i].run_neuron(inputs) for i in range(self.n_neurons)]

    def update_layer(self, inputs, out_layer):
        for i in range(self.n_neurons):
            error_n = 0
            for n in range(out_layer.n_neurons):
                error_n += out_layer.neurons[n].weights[i + 1] * out_layer.neurons[n].error * out_layer.neurons[n].derivative_v
                
            self.neurons[i].update(inputs, error_n, self.outputs[i])
