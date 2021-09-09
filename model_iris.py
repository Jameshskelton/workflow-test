from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd


iris = load_iris()
df = pd.read_csv('https://raw.githubusercontent.com/Jameshskelton/workflow-test/main/iris.csv')

def make_prediction(network, df):
    x = network.predict(df)
    out = pd.DataFrame(x, columns = ['pred'])
    out['actual'] = iris.target
    return out

Test = make_prediction(network, data)

Test.to_csv('predSpecies.csv')
print('Done')


