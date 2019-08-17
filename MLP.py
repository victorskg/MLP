import numpy as np
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer

class MLP(object):
    def __init__(self, att_number, learn_rate, epochs, n_hidden_layer, n_output_layer):
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.att_number = att_number
        self.n_hidden_layer = n_hidden_layer
        self.n_output_layer = n_output_layer
        self.hidden_layer = HiddenLayer(att_number, learn_rate, n_hidden_layer)
        self.output_layer = OutputLayer(n_hidden_layer, learn_rate, n_output_layer)

    def feedforward(self, inputs, expected):
        self.hidden_layer.run_layer(inputs)
        self.output_layer.run_layer(self.hidden_layer.outputs, expected)
        self.hidden_layer.update_layer(inputs, self.output_layer)

    def train(self, train_data, att):
        for _ in range(self.epochs):
            np.random.shuffle(train_data)
            for d in train_data:
                selected_inputs, expected = self.inputs_and_expected(d, att)
                self.feedforward(selected_inputs, expected)

    def test(self, test_data, att):
        hits = 0
        for data in test_data:
            selected_inputs, expected = self.inputs_and_expected(data, att)
            self.hidden_layer.run_layer(selected_inputs)
            outputs = self.output_layer.run_test(self.hidden_layer.outputs)
            predict = self.predict(outputs)
            hits = hits + 1 if np.array_equal(predict, expected) else hits
            #print('Expected: {0}, Predict: {1}'.format(expected, predict))

        print((hits / len(test_data)) * 100)
        return (hits / len(test_data)) * 100

    @staticmethod
    def inputs_and_expected(d, att):
        expected = np.array(list(d[4])).astype(np.int)
        selected_inputs = [d[att[i]] for i in range(len(att))]
        return selected_inputs, expected

    @staticmethod
    def predict(outputs):
        predict = [1 if output == np.amax(outputs) else 0 for output in outputs]
        return predict
