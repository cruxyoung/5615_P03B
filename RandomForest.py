from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics

fruits = pd.read_csv("Wholesale customers data.csv")

feature_names_fruits = ['Fresh', 'Milk', 'Frozen', 'Delicassen']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['Channel']


X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)
clf = RandomForestClassifier(n_jobs=2)


clf.fit(X_train, y_train)

print('R-squared score (training): {:.3f}'
     .format(clf.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(clf.score(X_test, y_test)))
