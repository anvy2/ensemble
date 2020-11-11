import matplotlib.pyplot as plt
from matplotlib.pyplot import annotate
import seaborn as sns


def wine_dataset_analytics(data):
    print(data.describe())
    corelation = data.corr()
    return corelation
