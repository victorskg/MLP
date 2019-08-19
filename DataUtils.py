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

def get_cancer():
    dataset = pd.read_csv("datasets/breast-cancer-wisconsin.data", header=None)
    columns = [str(x) for x in range(10)]
    columns.append('class')
    dataset.columns = columns
    dataset['class'] = dataset['class'].replace(2, '10')
    dataset['class'] = dataset['class'].replace(4, '01')

    dataset[columns[:len(columns)-1]] = dataset[columns[:len(columns)-1]].apply(normalize)

    return dataset

def get_xor():
    data = create_points([0,0], '10')
    data = data.append(create_points([1,1], '10'), ignore_index=True)
    data = data.append(create_points([0,1], '01'), ignore_index=True)
    data = data.append(create_points([1,0], '01'),  ignore_index=True)
    
    return data

def create_points(source, _class):
    points = []
    for _ in range(50):
        coords = [source[i] + np.random.random() * 0.09 for i in range(2)]
        coords.append(_class)
        points.append(coords)          
    return pd.DataFrame(data=points, columns=['0', '1', 'class'])

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

def plot_decision_surface_mlp(mlp, test_data, inputs):
    x1_colunm, x2_colunm = test_data[:, inputs[0]], test_data[:, inputs[1]]
    x1_max, x1_min = np.amax(x1_colunm) + 0.5, np.amin(x1_colunm) - 0.5
    x2_max, x2_min = np.amax(x2_colunm) + 0.5, np.amin(x2_colunm) - 0.5

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.07), np.arange(x2_min, x2_max, 0.07))
    Z = np.array([xx1.ravel(), xx2.ravel()]).T

    fig, ax = plt.subplots()
    ax.set_facecolor((0.97, 0.97, 0.97))
    for x1, x2 in Z:
        guess = mlp.get_predict([x1, x2])
        if np.array_equal(guess, [1, 0]):
            ax.scatter(x1, x2, c='red', s=1.5, marker='o')
        elif np.array_equal(guess, [0, 1]):
            ax.scatter(x1, x2, c='blue', s=1.5, marker='o')

    for row in test_data:
        expected = np.array(list(row[len(row) - 1])).astype(np.int)
        if np.array_equal(expected, [1, 0]):
            ax.scatter(row[inputs[0]], row[inputs[1]], c='red', marker='v')
        elif np.array_equal(expected, [0, 1]):
            ax.scatter(row[inputs[0]], row[inputs[1]], c='blue', marker='*')
    
    plt.show()

def plot_decision_surface_elm(elm, test_data, inputs):
    x1_colunm, x2_colunm = test_data[:, inputs[0]], test_data[:, inputs[1]]
    x1_max, x1_min = np.amax(x1_colunm) + 0.5, np.amin(x1_colunm) - 0.5
    x2_max, x2_min = np.amax(x2_colunm) + 0.5, np.amin(x2_colunm) - 0.5

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.07), np.arange(x2_min, x2_max, 0.07))
    Z = np.array([xx1.ravel(), xx2.ravel()]).T

    fig, ax = plt.subplots()
    ax.set_facecolor((0.97, 0.97, 0.97))
    elm.test(Z, inputs)
    guesses = elm.y_label
    for index in range(len(guesses)):
        if np.array_equal(guesses[index], [1, 0]):
            ax.scatter(Z[index][0], Z[index][1], c='red', s=1.5, marker='o')
        elif np.array_equal(guesses[index], [0, 1]):
            ax.scatter(Z[index][0], Z[index][1], c='blue', s=1.5, marker='o')

    for row in test_data:
        expected = np.array(list(row[len(row) - 1])).astype(np.int)
        if np.array_equal(expected, [1, 0]):
            ax.scatter(row[inputs[0]], row[inputs[1]], c='red', marker='v')
        elif np.array_equal(expected, [0, 1]):
            ax.scatter(row[inputs[0]], row[inputs[1]], c='blue', marker='*')
    
    plt.show()

def artificial(x):
    return (3 * math.sin(x)) + 1

def normalize(df):
    return (df-df.min())/(df.max()-df.min())
