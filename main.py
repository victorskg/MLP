import numpy as np
from MLP import MLP
import DataUtils as ut

def main():
    #run_iris()
    run_vertebral()

def run_iris():
    att = [0, 1, 2, 3]
    mlp = MLP(4, 0.1, 500, 6, 3)
    d = ut.get_iris_data().to_numpy()
    
    realization(mlp, d, att)

def run_vertebral():
    att = [0, 1, 2, 3, 4, 5]
    mlp = MLP(6, 0.1, 500, 12, 3)
    d = ut.get_column_data().to_numpy()

    realization(mlp, d, att)

def realization(mlp, d, att):
    np.random.shuffle(d)
    qt_training = int(0.8 * len(d))
    train_data, test_data = d[:qt_training], d[qt_training:]
    mlp.train(train_data, att)
    mlp.test(test_data, att)

if __name__ == '__main__':
    main()