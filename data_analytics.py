import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COLORS10 = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


def dataset_analytics(data):
    print(data.describe())
    corelation = data.corr()
    return corelation


def pca_plot(X, y):
    pca = PCA(n_components=2)
    X_pc = pca.fit_transform(X)

    fig, ax = plt.subplots()
    mask = y == 0
    ax.scatter(X_pc[mask, 0], X_pc[mask, 1], color=COLORS10[0],
               label='Class 0', alpha=0.5, s=20)
    ax.scatter(X_pc[~mask, 0], X_pc[~mask, 1],
               color=COLORS10[1], label='Class 1', alpha=0.5, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(loc='best')
    return fig
