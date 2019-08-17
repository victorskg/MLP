import pandas as pd


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

def normalize(df):
    return (df-df.min())/(df.max()-df.min())
