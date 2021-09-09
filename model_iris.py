from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

def make_prediction(model, data):
    x = model.predict(data.drop('species', axis = 0)
    df = pd.DataFrame(iris.data)
    df['actual'] = iris.target
    df['pred'] = x
    df.to_csv('predSpecies.csv')


def main(model, data):
    make_prediction(model, data)

if __name__ == "__main__":
	main()