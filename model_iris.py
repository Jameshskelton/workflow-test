from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
from gradient import sdk_client

iris = load_iris()
df = iris.data

network = pickle.load(open('save/save.p', 'rb'))

def make_prediction(network, data):
    x = network.predict(df)
    out = pd.DataFrame(x, columns = ['pred'])
    out['actual'] = iris.target
    return out

Test = make_prediction(network, df)

Test.to_csv('predSpecies1.csv')
print('Done')


