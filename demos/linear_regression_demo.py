import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

fruits = pd.read_csv('Wholesale customers data.csv')

feature_names_fruits = ['Fresh', 'Milk', 'Frozen', 'Delicassen']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['Channel']
target_names_fruits = ['1', '2', '3']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)


# X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1,
#                                                    random_state = 0)
# linreg = GaussianNB().fit(X_train, y_train)
linreg = RandomForestClassifier(max_depth=2, random_state=0)
linreg.fit(X_train, y_train)

# plt.scatter(X_train.values[:,0], X_train.values[:,1], marker= 'o', s=50)
# plt.show()

# print('linear model coeff (w): {}'
#      .format(linreg.coef_))
# print('linear model intercept (b): {:.3f}'
#      .format(linreg.intercept_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))

print(linreg.predict([[fruits.values[1][2],fruits.values[1][3]	,	fruits.values[1][5]	,	fruits.values[1][7]]]))