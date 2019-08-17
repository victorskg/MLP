import numpy as np
from MLP import MLP
import DataUtils as ut

def main():
    att = [0, 1, 2, 3]
    mlp = MLP(4, 0.1, 500, 6, 3)
    d = ut.get_iris_data().to_numpy()
    
    np.random.shuffle(d)
    qt_training = int(0.8 * len(d))
    train_data, test_data = d[:qt_training], d[qt_training:]
    mlp.train(train_data, att)
    mlp.test(test_data, att)
    

if __name__ == '__main__':
    main()