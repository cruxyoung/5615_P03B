import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
'''
The normal process of a marchine learning including:
1. transforming the data
2. train the model using knn
3. output the train score
4. test the training result and output test score
'''



fruits = pd.read_table('fruit_data.txt')

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']


X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler
# examine the origin data and transform features to a given range
scaler = MinMaxScaler()
# fit_transfrom  = fit first and then transform
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# n_neighbors stands for the number of closest neighbors to find 
knn = KNeighborsClassifier(n_neighbors = 5)
# input the transformed dataset
knn.fit(X_train_scaled, y_train)
# {:.2f} output	2 decimal places
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print('Predicted fruit type for ', example_fruit, ' is ', 
          target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])