from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pickle

iris = load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

pickle.dump( clf, open( "save.p", "wb" ) )