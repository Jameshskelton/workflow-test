from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

iris = load_iris()
df = iris.data

network = pickle.load(open('fte72i7f1w/models/mosygl1eabx1z7o/finalized_model.pkl', 'rb'))

def make_prediction(network, df):
    x = network.predict(df)
    out = pd.DataFrame(x, columns = ['pred'])
    out['actual'] = iris.target
    return out

Test = make_prediction(network, df)

Test.to_csv('predSpecies1.csv')
print('Done')


