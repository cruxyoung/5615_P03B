import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# fruits = pd.read_table('fruit_data.txt')

# feature_names_fruits = ['height', 'width', 'mass', 'color_score']
# X_fruits = fruits[feature_names_fruits]
# y_fruits = fruits['fruit_label']
# target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

# X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# # we must apply the scaling to the test set that we computed for the training set
# X_test_scaled = scaler.transform(X_test)

X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
                                                   random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))