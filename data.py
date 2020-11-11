import pandas as pd
import os

path = os.getcwd()
dataset = os.path.join(path, 'dataset')


def get_wine_dataset():
    location = os.path.join(dataset, 'winequality-red.csv')
    data = pd.read_csv(location)
    return data
