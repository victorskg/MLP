import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_iris_data():
    dataset = pd.read_csv("datasets/iris.data", header=None)
    dataset.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    dataset['class'] = dataset['class'].replace('Iris-setosa', '100')
    dataset['class'] = dataset['class'].replace('Iris-versicolor', '010')
    dataset['class'] = dataset['class'].replace('Iris-virginica', '001')

    return dataset

def get_column_data():
    dataset = pd.read_csv("datasets/column_3C.data", header=None)
    dataset.columns = ['0', '1', '2', '3', '4', '5', 'class']
    dataset['class'] = dataset['class'].replace('DH', '100')
    dataset['class'] = dataset['class'].replace('SL', '010')
    dataset['class'] = dataset['class'].replace('NO', '001')

    dataset[['0', '1', '2', '3', '4', '5']] = dataset[['0', '1', '2', '3', '4', '5']].apply(normalize)

    return dataset

def get_dermatology():
    dataset = pd.read_csv("datasets/dermatology.data", header=None)
    columns = [str(x) for x in range(34)]
    columns.append('class')
    dataset.columns = columns
    dataset['class'] = dataset['class'].replace(1, '100000')
    dataset['class'] = dataset['class'].replace(2, '010000')
    dataset['class'] = dataset['class'].replace(3, '001000')
    dataset['class'] = dataset['class'].replace(4, '000100')
    dataset['class'] = dataset['class'].replace(5, '000010')
    dataset['class'] = dataset['class'].replace(6, '000001')

    dataset[columns[:len(columns)-1]] = dataset[columns[:len(columns)-1]].apply(normalize)

    return dataset


def get_artificial():
    X = np.linspace(0, 10, 500)
    Y = [artificial(x) + np.random.uniform(-1, 1) for x in X]
    dataset = np.array([[i, j] for i, j in zip(X, Y)])
    plt.scatter(dataset[:, 0], dataset[:, 1], s=3, c='r')
    plt.show()
    
    return dataset

def plot(mlp, dataset):
    y = []
    for data in dataset:
       outs = mlp.hidden_layer.run_layer(data[0])
       y.append(mlp.output_layer.run_test(outs))
    
    plt.scatter(dataset[:, 0], dataset[:, 1], s=3, c='r')
    plt.plot(dataset[:, 0], y, color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Result after training')
    plt.show()

def artificial(x):
    return (3 * math.sin(x)) + 1

def normalize(df):
    return (df-df.min())/(df.max()-df.min())
