from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd


def make_prediction(network, data):
    x = network.predict(data.drop('species', axis = 0)
    df = pd.DataFrame(iris.data)
    df['actual'] = iris.target
    df['pred'] = x
    return df

Test = make_prediction(network, data)

Test.to_csv('predSpecies.csv')
print('Done')


