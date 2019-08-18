import numpy as np
from MLP import MLP
import DataUtils as ut

def main():
    #run_iris()
    #run_vertebral()
    run_dermatology()
    #run_artificial()

def run_iris():
    acs = []
    att = [0, 1, 2, 3]
    d = ut.get_iris_data().to_numpy()
    
    for i in range(20):
        mlp = MLP(4, 0.1, 500, 6, 3)
        acs.append(realization(mlp, d, att))

        print('Realização: {0}, Acurácia: {1}%'.format(i, acs[i]))
    
    print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(calc_accuracy(acs), np.std(acs)))

def run_vertebral():
    acs = []
    att = [0, 1, 2, 3, 4, 5]
    d = ut.get_column_data().to_numpy()

    for i in range(20):
        mlp = MLP(6, 0.1, 500, 12, 3)
        acs.append(realization(mlp, d, att))

        print('Realização: {0}, Acurácia: {1}%'.format(i, acs[i]))
    
    print('Acurácia após 20 realizações: {0}%, Desvio padrão: {1}'.format(calc_accuracy(acs), np.std(acs)))

def run_dermatology():
    att = [x for x in range(34)]
    d = ut.get_dermatology().to_numpy()
    mlp = MLP(34, 0.1, 500, 12, 6)
    print(realization(mlp, d, att))

def run_artificial():
    att = [0]
    mlp = MLP(1, 0.1, 500, 1, 1)
    d = ut.get_artificial()
    
    realization_regression(mlp, d, att)

def realization(mlp, d, att):
    np.random.shuffle(d)
    qt_training = int(0.8 * len(d))
    train_data, test_data = d[:qt_training], d[qt_training:]
    mlp.train(train_data, att)
    
    return mlp.test(test_data, att)

def realization_regression(mlp, d, att):
    np.random.shuffle(d)
    qt_training = int(0.8 * len(d))
    train_data, test_data = d[:qt_training], d[qt_training:]
    mlp.train(train_data, att)
    ut.plot(mlp, test_data)

def calc_accuracy(array):
    return sum(array) / len(array) 

if __name__ == '__main__':
    main()