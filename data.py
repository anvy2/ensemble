from os import sep
import pandas as pd
import os
import numpy as np

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


def get_titanic_dataset():
    location_train = os.path.join(dataset, 'titanic', 'train.csv')
    location_test = os.path.join(dataset, 'titanic', 'test.csv')
    train_data = pd.read_csv(location_train)
    test_data = pd.read_csv(location_test)
    return [train_data, test_data]


def get_proteomics():
    location = os.path.join(dataset, 'Harmonizome',
                            'gene_attribute_matrix_cleaned.txt.gz')
    proteome = pd.read_csv(location,
                           sep='\t', compression='gzip', skiprows=2)
    # Load HGNC gene family
    location_gene_family = os.path.join(
        dataset, 'Harmonizome', 'HGNC_gene_family.txt')
    gene_family = pd.read_csv(location_gene_family, sep='\t')
    # Left join with proteome data
    gene_family = gene_family[['Approved symbol', 'Group name']]
    gene_family.set_index('Approved symbol', inplace=True)

    proteome = proteome.drop(
        ['UniprotAcc', 'GeneID/Brenda Tissue Ontology BTO:'], axis=1)
    proteome.set_index('GeneSym', inplace=True)

    proteome = proteome.merge(
        gene_family, left_index=True, right_index=True, how='inner')

    # Split X and y
    z = pd.read_csv(location_gene_family, sep='\t')
    X = proteome.drop(['Group name'], axis=1)
    X = X.values
    y = proteome['Group name']
    y = map(lambda x: 'kinase' in x.lower(), y)
    y = list(y)
    y = np.array(y, dtype=np.int64)
    return [X, y, z]
