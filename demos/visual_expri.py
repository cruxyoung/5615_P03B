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
h = .02
x_min, x_max = X_train.min() - 1, X_train.max() + 1
y_min, y_max = y_train.min() - 1, y_train.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()