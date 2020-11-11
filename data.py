import pandas as pd
import os

path = os.getcwd()
dataset = os.path.join(path, 'dataset')


def get_wine_dataset():
    location = os.path.join(dataset, 'wine', 'winequality-red.csv')
    data = pd.read_csv(location)
    return data


def get_heart_dataset():
    location = os.path.join(dataset, 'heart_UCI', 'heart.csv')
    data = pd.read_csv(location)
    data.head()
    return data
