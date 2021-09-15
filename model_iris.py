from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

iris = load_iris()
df = iris.data

network = pickle.load(open('mosygl1eabx1z7o', 'rb'))

def make_prediction(network, df):
    x = network.predict(df)
    out = pd.DataFrame(x, columns = ['pred'])
    out['actual'] = iris.target
    return out

Test = make_prediction(network, df)

Test.to_csv('predSpecies1.csv')
print('Done')


