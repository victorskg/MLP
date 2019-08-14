import pandas as pd


def get_iris_data():
    dataset = pd.read_csv("datasets/iris.data", header=None)
    dataset.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    dataset['class'] = dataset['class'].replace('Iris-setosa', '100')
    dataset['class'] = dataset['class'].replace('Iris-versicolor', '010')
    dataset['class'] = dataset['class'].replace('Iris-virginica', '001')

    return dataset
