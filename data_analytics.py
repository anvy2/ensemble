import matplotlib.pyplot as plt
def dataset_analytics(data):
    print(data.describe())
    corelation = data.corr()
    return corelation
