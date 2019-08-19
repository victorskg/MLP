import math
import numpy as np

class ELM(object):
    def __init__ (self, hid_num, out_num):
        self.hid_num = hid_num
        self.out_num = out_num

    def train(self, dataset, att):
        X, y = np.array([self.inputs(d, att) for d in dataset]), [self.expected(d, att) for d in dataset]

        X = self.add_bias(X)

        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))
        
        _H = np.linalg.pinv(self.sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)
        
        return self

    def test(self, test_data, att):
        hits = 0
        X, d = np.array([self.inputs(d, att) for d in test_data]), [self.expected(d, att) for d in test_data]
        
        _H = self.sigmoid(np.dot(self.W, self.add_bias(X).T))
        
        y = np.dot(_H.T, self.beta)
        y_label = [self.predict(out) for out in y]
        self.y_label = y_label
        
        for i,j in zip(y_label, d):
            hits = hits + 1 if np.array_equal(i, j) else hits
        
        return (hits / len(test_data)) * 100

    @staticmethod
    def sigmoid(x):
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def add_bias(X):
        return np.c_[X, np.ones(X.shape[0])]

    @staticmethod
    def inputs(d, att):
        selected_inputs = [d[att[i]] for i in range(len(att))]
        return selected_inputs

    @staticmethod
    def expected(d, att):
        expected = np.array([d[len(d)-1]]) if (isinstance(d[len(d)-1], np.floating)) else np.array(list(d[len(d)-1])).astype(np.int)
        return expected

    @staticmethod
    def predict(outputs):
        predict = [1 if output == np.amax(outputs) else 0 for output in outputs]
        return predict
