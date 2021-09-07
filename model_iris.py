from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

def make_prediction(model, data):
    x = model.predict(data)
    df = pd.DataFrame(iris.data)
    df['actual'] = iris.target
    df['pred'] = x
    df.to_csv('iris_pred.csv')


